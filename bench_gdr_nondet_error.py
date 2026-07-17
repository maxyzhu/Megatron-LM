import torch

from megatron.core import tensor_parallel
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

# shapes matching a GDN layer's gdr inputs (after prepare/repeat): [B, T, H, D]
B, T, H, K, V = 2, 4096, 32, 128, 128
DTYPE = torch.bfloat16
DEV = "cuda"


def make_inputs(seed=0):
    g_cpu = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=g_cpu, device=DEV, dtype=DTYPE)
    k = torch.randn(B, T, H, K, generator=g_cpu, device=DEV, dtype=DTYPE)
    v = torch.randn(B, T, H, V, generator=g_cpu, device=DEV, dtype=DTYPE)
    # g (decay) in log space, negative; beta in (0,1)
    g = -torch.rand(B, T, H, generator=g_cpu, device=DEV, dtype=torch.float32).abs()
    beta = torch.rand(B, T, H, generator=g_cpu, device=DEV, dtype=DTYPE)
    return q, k, v, g, beta


def leaves(inputs):
    return [x.detach().clone().requires_grad_(True) for x in inputs]


def kernel(q, k, v, g, beta):
    out, _ = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, cu_seqlens=None,
    )
    return out


def run(inputs, grad_out, recompute):
    xs = leaves(inputs)
    if recompute:
        out = tensor_parallel.checkpoint(kernel, False, *xs)
    else:
        out = kernel(*xs)
    out.backward(grad_out)
    grads = [x.grad.detach().float() for x in xs]
    return out.detach().float(), grads


def rel(a, b):
    # relative L2 error ||a-b|| / ||b||
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def main():
    torch.cuda.set_device(0)
    inputs = make_inputs(seed=0)

    # fixed upstream grad so we measure the Jacobian-vector product, not a loss that
    # itself varies with the (non-deterministic) forward output.
    gg = torch.Generator(device=DEV).manual_seed(123)
    grad_out = torch.randn(B, T, H, V, generator=gg, device=DEV, dtype=DTYPE)

    names = ["dq", "dk", "dv", "dg", "dbeta"]

    # reference: no recompute
    out_ref, g_ref = run(inputs, grad_out, recompute=False)
    # second no-recompute run: the kernel's inherent run-to-run noise floor
    out_ref2, g_ref2 = run(inputs, grad_out, recompute=False)
    # recompute (checkpoint re-runs the non-deterministic kernel in backward)
    out_rec, g_rec = run(inputs, grad_out, recompute=True)

    print(f"forward out  run-to-run rel diff : {rel(out_ref2, out_ref):.3e}")
    print(f"forward out  recompute vs ref    : {rel(out_rec, out_ref):.3e}")
    print()
    print(f"{'grad':6s} {'noise floor (2nd run)':>22s} {'recompute vs ref':>20s}")
    for n, a_noise, a_rec, base in zip(names, g_ref2, g_rec, g_ref):
        print(f"{n:6s} {rel(a_noise, base):22.3e} {rel(a_rec, base):20.3e}")

    print("\nInterpretation: if 'recompute vs ref' ~ 'noise floor', non-det gdr recompute "
          "adds no error beyond the kernel's own non-determinism -> safe to allow.")


if __name__ == "__main__":
    main()