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

SEQ_LEN, BATCH, WARMUP, ITERS = 8192, 2, 5, 20


def make_config():
    return TransformerConfig(
        hidden_size=2048, linear_conv_kernel_dim=4,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_num_key_heads=16, linear_num_value_heads=32,
        num_layers=1, normalization="RMSNorm", layernorm_zero_centered_gamma=True,
        num_attention_heads=16, num_query_groups=2, activation_func=F.silu, bf16=True,
        deterministic_mode=False,  # -> FLA (non-deterministic) gdr kernel selected at init
        experimental_attention_variant="gated_delta_net", linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )


def build(recompute, pg):
    cfg = make_config()
    if recompute:
        cfg.recompute_granularity = "selective"
        cfg.recompute_modules = ["gdn_gated_delta_rule"]
    model_parallel_cuda_manual_seed(42)
    torch.manual_seed(42)
    submod = get_experimental_attention_variant_module_spec(config=cfg).submodules
    gdn = GatedDeltaNet(
        cfg, submodules=submod, layer_number=1, bias=False, conv_bias=False,
        conv_init=1.0, use_qk_l2norm=True, A_init_range=(1, 16), pg_collection=pg,
    ).cuda().bfloat16()
    # MEASUREMENT HACK: FLA (non-det) gdr kernel is already selected at __init__ (built with
    # deterministic_mode=False). Flip the flag only to pass forward's deterministic assert so we
    # can measure NON-DET gdr recompute. Both baseline and recompute runs do this identically,
    # so conv1d's path (F.conv1d) cancels in the delta.
    gdn.config.deterministic_mode = True
    return gdn


def bench(recompute, pg):
    gdn = build(recompute, pg)
    hs = torch.randn(
        (SEQ_LEN, BATCH, 2048), device=torch.cuda.current_device(),
        dtype=torch.bfloat16, requires_grad=True,
    )

    def fwd_bwd(measure_act=False):
        gdn.zero_grad(set_to_none=True)
        if hs.grad is not None:
            hs.grad = None
        act = None
        if measure_act:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
        out, _ = gdn(hs, None)
        if measure_act:
            torch.cuda.synchronize()
            act = (torch.cuda.memory_allocated() - before) / (1024 ** 2)
        out.float().sum().backward()
        return act

    for _ in range(WARMUP):
        fwd_bwd()
    torch.cuda.synchronize()
    act = fwd_bwd(measure_act=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fwd_bwd()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS * 1e3
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    del gdn, hs
    torch.cuda.empty_cache()
    return dt, peak, act


def main():
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(123)
    pg = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    bdt, bpeak, bact = bench(False, pg)
    rdt, rpeak, ract = bench(True, pg)
    print(f"baseline (non-det gdr)      {bdt:8.2f} ms  peak {bpeak:8.1f}MB  act {bact:8.1f}MB")
    print(f"gdr recompute (non-det)     {rdt:8.2f} ms  peak {rpeak:8.1f}MB  act {ract:8.1f}MB")
    print(f"\nact saved %  : {(bact - ract) / bact * 100:6.2f}")
    print(f"peak saved % : {(bpeak - rpeak) / bpeak * 100:6.2f}")
    print(f"time ovhd %  : {(rdt - bdt) / bdt * 100:6.2f}")
    Utils.destroy_model_parallel()


if __name__ == "__main__":
    main()