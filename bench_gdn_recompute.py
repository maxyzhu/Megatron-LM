import os
import time

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# --- knobs (override via env) ---
TP = int(os.environ.get("TP", "1"))
CP = int(os.environ.get("CP", "1"))
SP = os.environ.get("SP", "0") == "1"          # sequence_parallel (exercises in_proj SP all-gather)
SEQ_LEN = int(os.environ.get("SEQ_LEN", "8192"))   # GLOBAL seq len
BATCH = int(os.environ.get("BATCH", "2"))
WARMUP = int(os.environ.get("WARMUP", "5"))
ITERS = int(os.environ.get("ITERS", "20"))


def make_config(deterministic):
    return TransformerConfig(
        hidden_size=2048,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        num_layers=1,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        num_attention_heads=16,
        num_query_groups=2,
        activation_func=F.silu,
        bf16=True,
        deterministic_mode=deterministic,
        tensor_model_parallel_size=TP,
        context_parallel_size=CP,
        sequence_parallel=SP,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )


def build_gdn(config, pg_collection):
    submod = get_experimental_attention_variant_module_spec(config=config).submodules
    gdn = GatedDeltaNet(
        config,
        submodules=submod,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg_collection,
    )
    return gdn.cuda().bfloat16()


def bench(deterministic, recompute_module, pg_collection, sp_size):
    config = make_config(deterministic)
    if recompute_module is not None:
        config.recompute_granularity = "selective"
        config.recompute_modules = [recompute_module]

    model_parallel_cuda_manual_seed(42)
    torch.manual_seed(42)
    gdn = build_gdn(config, pg_collection)

    local_seq = SEQ_LEN // sp_size // CP
    assert local_seq * sp_size * CP == SEQ_LEN, (
        f"SEQ_LEN={SEQ_LEN} must be divisible by sp_size*CP={sp_size * CP}"
    )
    hs = torch.randn(
        (local_seq, BATCH, config.hidden_size),
        device=torch.cuda.current_device(),
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def step():
        gdn.zero_grad(set_to_none=True)
        if hs.grad is not None:
            hs.grad = None
        out, _ = gdn(hs, None)
        out.float().sum().backward()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS * 1e3  # ms/iter
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB (per-rank)

    del gdn, hs
    torch.cuda.empty_cache()
    return dt, peak


def main():
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=1,
        context_parallel_size=CP,
    )
    model_parallel_cuda_manual_seed(123)
    sp_size = TP if SP else 1

    tp = parallel_state.get_tensor_model_parallel_group()
    cp = parallel_state.get_context_parallel_group()
    pg = ProcessGroupCollection(tp=tp, cp=cp)

    rank0 = torch.distributed.get_rank() == 0

    runs = [
        ("baseline (non-det)", False, None),
        ("gdn_in_proj (non-det)", False, "gdn_in_proj"),
        ("gdn_conv1d (non-det)", False, "gdn_conv1d"),
        ("baseline (det)", True, None),
        ("gdn_gated_delta_rule (det)", True, "gdn_gated_delta_rule"),
    ]

    results = {}
    for label, det, mod in runs:
        dt, peak = bench(det, mod, pg, sp_size)
        results[label] = (dt, peak)
        if rank0:
            print(f"{label:32s}  {dt:8.2f} ms/iter   {peak:9.1f} MB peak (rank0)")

    if rank0:
        def pct(a, b):
            return (a - b) / b * 100.0

        base_nd_t, base_nd_m = results["baseline (non-det)"]
        base_d_t, base_d_m = results["baseline (det)"]

        print(f"\n=== config: TP={TP} CP={CP} SP={SP} SEQ_LEN={SEQ_LEN} BATCH={BATCH} ===")
        print(f"{'segment':28s} {'mem saved %':>12s} {'time overhead %':>16s}")
        for label, base_t, base_m in [
            ("gdn_in_proj (non-det)", base_nd_t, base_nd_m),
            ("gdn_conv1d (non-det)", base_nd_t, base_nd_m),
            ("gdn_gated_delta_rule (det)", base_d_t, base_d_m),
        ]:
            t, m = results[label]
            print(f"{label:28s} {(-pct(m, base_m)):12.2f} {pct(t, base_t):16.2f}")

    Utils.destroy_model_parallel()


if __name__ == "__main__":
    main()