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

N_LAYERS = int(os.environ.get("N_LAYERS", "20"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "2048"))
BATCH = int(os.environ.get("BATCH", "1"))
WARMUP, ITERS = 3, 10


def make_config(recompute):
    cfg = TransformerConfig(
        hidden_size=2048, linear_conv_kernel_dim=4,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_num_key_heads=16, linear_num_value_heads=32,
        num_layers=N_LAYERS, normalization="RMSNorm", layernorm_zero_centered_gamma=True,
        num_attention_heads=16, num_query_groups=2, activation_func=F.silu, bf16=True,
        deterministic_mode=False,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1] * N_LAYERS,
        transformer_impl="transformer_engine",
    )
    if recompute:
        cfg.recompute_granularity = "selective"
        cfg.recompute_modules = ["gdn_in_proj_conv"]
    return cfg


def build_stack(cfg, pg):
    submod = get_experimental_attention_variant_module_spec(config=cfg).submodules
    layers = []
    for i in range(N_LAYERS):
        gdn = GatedDeltaNet(
            cfg, submodules=submod, layer_number=i + 1, bias=False, conv_bias=False,
            conv_init=1.0, use_qk_l2norm=True, A_init_range=(1, 16), pg_collection=pg,
        )
        layers.append(gdn.cuda().bfloat16())
    return torch.nn.ModuleList(layers)


def bench(recompute, pg):
    model_parallel_cuda_manual_seed(42)
    torch.manual_seed(42)
    cfg = make_config(recompute)
    layers = build_stack(cfg, pg)
    assert all(l.recompute_in_proj_conv is recompute for l in layers)

    hs = torch.randn(
        (SEQ_LEN, BATCH, cfg.hidden_size),
        device=torch.cuda.current_device(), dtype=torch.bfloat16, requires_grad=True,
    )

    def step(measure_act=False):
        for l in layers:
            l.zero_grad(set_to_none=True)
        if hs.grad is not None:
            hs.grad = None
        x = hs
        if measure_act:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
        for l in layers:
            out, _ = l(x, None)
            x = x + out
        act = None
        if measure_act:
            torch.cuda.synchronize()
            act = (torch.cuda.memory_allocated() - before) / (1024 ** 2)
        x.float().sum().backward()
        return act

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    act = step(measure_act=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS * 1e3
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    del layers, hs
    torch.cuda.empty_cache()
    return dt, peak, act


def main():
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    pg = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    bdt, bpeak, bact = bench(False, pg)
    rdt, rpeak, ract = bench(True, pg)
    print(f"\n=== {N_LAYERS}-layer GDN stack  SEQ={SEQ_LEN} BATCH={BATCH} ===")
    print(f"baseline        {bdt:8.1f} ms  peak {bpeak:9.1f}MB  act {bact:9.1f}MB")
    print(f"gdn_in_proj_conv {rdt:8.1f} ms  peak {rpeak:9.1f}MB  act {ract:9.1f}MB")
    print(f"\npeak saved % : {(bpeak - rpeak) / bpeak * 100:6.2f}")
    print(f"act  saved % : {(bact - ract) / bact * 100:6.2f}")
    print(f"time ovhd %  : {(rdt - bdt) / bdt * 100:6.2f}")
    Utils.destroy_model_parallel()


if __name__ == "__main__":
    main()