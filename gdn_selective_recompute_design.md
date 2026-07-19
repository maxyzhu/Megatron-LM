# GDN Selective Recompute — Implementation Plan

Roadmap item: **GDN selective recompute (in_proj, conv1d, gated_delta_rule) (P1)** — MoE Roadmap #4815.
Branch: `gdn-selective-recompute` (independent, forked from latest `upstream/main`).
Profiler (PR #5662) is a **measurement tool only** — do NOT import it into the implementation. Use it locally to quantify savings; keep the two branches code-decoupled.

---

## 0. Verification status (source-confirmed, not assumed)

All confirmed against real upstream source:

- Recompute whitelist lives in `transformer_config` as `allowed_modules`; currently contains `gdn_norm_out` but NOT the three new modules.
- Existing GDN recompute (`gdn_norm_out`) uses `CheckpointWithoutOutput` (output-discarding), applied to the gated-norm + a2a segment, with the discard hook attached to the final `out` (out_proj output).
- Two checkpoint mechanisms exist and are NOT interchangeable:
  - `CheckpointWithoutOutput` (class): checkpoint + optional two-step output-discarding.
  - `checkpoint(function, distribute_saved_activations, *args)` (function): standard checkpoint via `CheckpointFunction.apply`.
- **Both** mechanisms save/restore RNG state (`_get_all_rng_states` / `_set_all_rng_states`). Neither controls FLA kernel non-determinism (that comes from float reduction order, not torch RNG).
- `torch_chunk_gated_delta_rule` is the deterministic path (docstring: "Need this because FLA is not deterministic").

---

## 1. Skeleton scope

Goal of the skeleton: **structure stands up, runs, switches work.** NOT speed, NOT max memory savings. Correctness first, optimization later.

Skeleton acceptance criteria:

1. `--recompute-modules gdn_in_proj` (and the other two) are accepted, not rejected by config validation.
2. When a switch is on, the corresponding segment actually takes the checkpoint branch (verify with a log/print first).
3. On **single GPU + deterministic mode**, enabling any one segment's recompute produces a loss curve matching the no-recompute run (coarse correctness screen; gradcheck is the later fine check).

---

## 2. Changes

### Change 1 — `transformer_config.py` whitelist + guard

Append three modules to `allowed_modules`:

```python
allowed_modules = {
    "core_attn", "moe_act", "layernorm", "mla_up_proj",
    "mlp", "moe", "shared_experts", "gdn_norm_out",
    "gdn_in_proj", "gdn_conv1d", "gdn_gated_delta_rule",   # new
}
```

Add a guard covering the new modules (minimal-change style: keep the existing `gdn_norm_out` guard untouched, add a separate guard for the three new ones):

```python
gdn_new_modules = {"gdn_in_proj", "gdn_conv1d", "gdn_gated_delta_rule"}
if (gdn_new_modules & set(self.recompute_modules)
        and self.experimental_attention_variant != "gated_delta_net"):
    raise ValueError(
        "gdn_in_proj / gdn_conv1d / gdn_gated_delta_rule in recompute_modules "
        "are only supported with experimental_attention_variant='gated_delta_net'."
    )
```

### Change 2 — `gated_delta_net.py` `__init__` flags

Mirror the existing `recompute_norm_out` pattern:

```python
# existing (leave untouched):
#   self.recompute_norm_out = False
#   self.norm_out_checkpoint = None
#   if self.config.recompute_granularity == "selective":
#       self.recompute_norm_out = "gdn_norm_out" in self.config.recompute_modules

self.recompute_in_proj = False
self.recompute_conv1d = False
self.recompute_gated_delta_rule = False
if self.config.recompute_granularity == "selective":
    self.recompute_in_proj = "gdn_in_proj" in self.config.recompute_modules
    self.recompute_conv1d = "gdn_conv1d" in self.config.recompute_modules
    self.recompute_gated_delta_rule = (
        "gdn_gated_delta_rule" in self.config.recompute_modules
    )
```

### Change 3 — `forward()` three segments

**Mechanism selection for skeleton: all three use the standard function-style `checkpoint`.**
Rationale: correctness-equivalent to `CheckpointWithoutOutput` (both save inputs, restore RNG, recompute); lowest complexity; no discard-hook timing to reason about; no crossing with the existing `gdn_norm_out` output-discarding hook.

#### ① in_proj

```python
nvtx_range_push(suffix="in_proj")
if self.recompute_in_proj:
    def _in_proj_fn(hs):
        out, _bias = self.in_proj(hs)
        return out
    qkvzba = tensor_parallel.checkpoint(_in_proj_fn, False, hidden_states)
else:
    qkvzba, _ = self.in_proj(hidden_states)
nvtx_range_pop(suffix="in_proj")
```

Boundary decision: wrap ONLY the linear, boundary before CP a2a. Do not recompute the all-to-all in v1.

#### ② conv1d

```python
nvtx_range_push(suffix="conv1d")
def _conv1d_fn(qkv_in):
    # Move the entire existing conv1d block (deterministic F.conv1d branch
    # + FLA causal_conv1d branch) in here verbatim.
    # IMPORTANT: the CP-local weight/bias slicing (get_parameter_local_cp)
    # must also be recomputed inside this closure, or recompute gets the
    # wrong local weights.
    ...
    return qkv_processed

if self.recompute_conv1d:
    qkv = tensor_parallel.checkpoint(_conv1d_fn, False, qkv)
else:
    qkv = _conv1d_fn(qkv)
nvtx_range_pop(suffix="conv1d")
```

#### ③ gated_delta_rule

```python
nvtx_range_push(suffix="gated_delta_rule")
if self.recompute_gated_delta_rule:
    # FLA chunk_gated_delta_rule is non-deterministic; recompute would break
    # gradients. Runtime guard (config validation should already block non-det):
    assert self.config.deterministic_mode, (
        "gdn_gated_delta_rule recompute currently requires deterministic_mode "
        "(FLA kernel is non-deterministic; recompute would break gradients)."
    )
    def _gdr_fn(q, k, v, g_, beta_):
        out, _state = self.gated_delta_rule(
            q, k, v, g=g_, beta=beta_,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False, cu_seqlens=cu_seqlens_q,
        )
        return out
    core_attn_out = tensor_parallel.checkpoint(
        _gdr_fn, False, query, key, value, g, beta
    )
    last_recurrent_state = None
else:
    core_attn_out, last_recurrent_state = self.gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, cu_seqlens=cu_seqlens_q,
    )
nvtx_range_pop(suffix="gated_delta_rule")
```

---

## 3. Pre-code check

Before writing the first line, confirm the export path of the function-style checkpoint:

```
python -c "from megatron.core import tensor_parallel; print(tensor_parallel.checkpoint)"
```

If it's not exposed as `tensor_parallel.checkpoint`, use the actual path (e.g. `tensor_parallel.random.checkpoint`).

---

## 4. GPU validation points (cannot be settled from source — must test)

1. **in_proj recompute × delayed wgrad timing.** `backward_dw` / `_backward_in_proj` delays weight-grad; standard checkpoint recompute may conflict. Test first WITHOUT `--delay-wgrad-compute`.
2. **conv1d closure CP-local weight recompute.** Confirm `get_parameter_local_cp` gets correct local slices on recompute, especially TP/CP > 1.
3. **conv1d dual-path consistency.** deterministic → `F.conv1d`; non-det → FLA `causal_conv1d`. Recompute must take the same path. Validate consistency under deterministic first.
4. **(Optimization phase, not blocking v1) gated_delta_rule → output-discarding upgrade.** `core_attn_out`'s downstream (`_gated_norm_and_a2a`) saves it for backward — this MEETS the `CheckpointWithoutOutput` docstring condition, so output-discarding may pay off. But: must measure `core_attn_out` storage with the profiler first, and must reason about backward-hook ordering vs the existing `gdn_norm_out` discard hook to confirm no crossing. Do NOT do this in the skeleton.

---

## 5. Validation config (3:1 hybrid awareness)

- GDN : gated-attention layers are ~3:1 in Qwen3-Next / Qwen3.5-style models. Confirm the exact layer alternation pattern and total layer count for the target config before building the validation setup.
- Report per-layer (single GDN layer) savings as the CORE metric; report whole-model activation-memory impact as an end-to-end secondary figure (GDN is ~75% of layers, so the whole-model number is diluted — state the denominator explicitly).
- With the profiler, statistics MUST separate GDN layers from attention layers. Layers alternate statically (not token-conditional like MoE routing), so group by layer type — do NOT average across mixed layer types.
- Verify segment isolation: recompute on GDN layers must not affect attention layers (clean boundaries, no cross-talk).
- Each of the three segments must be INDEPENDENTLY switchable via `recompute_modules` — the deliverable trade-off table (each segment alone / pairwise / all three: memory saved vs compute overhead) is the PR's core value and what only you can produce (you have the profiler).

---

## 6. Landing order (risk low → high)

1. **in_proj** — cleanest (pure linear, no state, no randomness). Validates the whole `checkpoint` API understanding.
2. **conv1d** — watch deterministic/FLA dual path + CP-local weights.
3. **gated_delta_rule** — restrict to / focus-validate deterministic path; gradcheck must pass; measure whether the two chunk loops are worth recomputing.

Then: independent PR, honest scope doc (what's included, what's deferred, GPU-validated reasoning). Merge before any follow-on.

---

## 7. Coordination reminder

Before heavy investment: proactively claim the P1 item with Connor-XY, and ask whether the P0.5 fine-grained activation offloading has an internal owner (排掉内部抢先风险). Bundle into one coordinated ask.