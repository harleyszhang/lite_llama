"""The fused Triton grouped-topk router vs the torch reference.

Grouped routing is launch-bound on the batches it actually serves: the torch
reference spends ~a dozen CUDA launches on tiny tensors per call (score,
biased group top-2, group top-k, scatter the group mask, expert top-k,
gather the originals, renormalise, scale), while the fused kernel is a
single launch with every intermediate in registers. This benchmark times
both on the DeepSeek-V3 routing geometry (256 experts, sigmoid scoring,
noaux_tc bias, 8 experts per token) from single-token decode to 8k-token
prefill, after verifying the kernel reproduces the reference on every
geometry the wrapper dispatches.

Usage:
    python benchmarks/kernels/bench_grouped_topk.py
"""

from __future__ import annotations

import torch
import triton
from microbench import run_perf_report

from lite_llama.kernels import grouped_topk, grouped_topk_torch

# --------------------------------------------------------------------------- #
# Routing geometries: the checkpoints the wrapper actually serves.
# --------------------------------------------------------------------------- #
#: DeepSeek-V3: 256 routed experts, sigmoid scoring with the noaux_tc bias.
#: V3-4layers ships the same family fields (8 groups, top-4 groups, sigmoid,
#: the 2.5 routed scale) on a trimmed expert table.
V3 = {
    "num_experts": 256,
    "top_k": 8,
    "renormalize": True,
    "num_expert_group": 8,
    "topk_group": 4,
    "scoring_func": "sigmoid",
    "routed_scaling_factor": 2.5,
    "bias": True,
}
#: The V2 family: the same group structure with softmax scoring and no bias.
V2 = {**V3, "scoring_func": "softmax", "bias": False}
#: A non-power-of-two table (20 experts per group): the kernel pads to 256
#: lanes and a padding lane must never outscore a real expert.
PADDED = {**V3, "num_experts": 160}


def _route_kwargs(geometry: dict) -> dict:
    """The router kwargs of a geometry, minus the fields that shape tensors."""
    return {k: v for k, v in geometry.items() if k not in ("num_experts", "bias")}


def _make_inputs(geometry: dict, num_tokens: int, device: str = "cuda"):
    logits = torch.randn(num_tokens, geometry["num_experts"], device=device, dtype=torch.float32)
    bias = (
        torch.randn(geometry["num_experts"], device=device, dtype=torch.float32)
        if geometry["bias"]
        else None
    )
    return logits, bias


#: The timed implementations. Both consume the same ``(logits, bias)`` pair.
PROVIDERS: dict[str, callable] = {
    "torch": lambda logits, bias: grouped_topk_torch(
        logits, e_score_correction_bias=bias, **_route_kwargs(V3)
    ),
    "triton": lambda logits, bias: grouped_topk(
        logits, e_score_correction_bias=bias, **_route_kwargs(V3)
    ),
}


def verify(device: str = "cuda") -> None:
    """Check the kernel against the reference on every geometry, before timing.

    ``torch.topk(..., sorted=False)`` leaves the reference's column order
    unspecified (the kernel emits its picks by descending biased score), so
    the comparison is order-free: identical expert sets, paired weights
    matching to fp32 reduction-order tolerance.
    """
    torch.manual_seed(0)
    for name, geometry in (("DeepSeek-V3", V3), ("DeepSeek-V2", V2), ("padded-160", PADDED)):
        logits, bias = _make_inputs(geometry, num_tokens=256, device=device)
        kwargs = _route_kwargs(geometry)
        ref_weights, ref_ids = grouped_topk_torch(logits, e_score_correction_bias=bias, **kwargs)
        weights, ids = grouped_topk(logits, e_score_correction_bias=bias, **kwargs)
        order = ids.sort(dim=-1)
        ref_order = ref_ids.sort(dim=-1)
        assert torch.equal(order.values, ref_order.values), (
            f"{name}: the kernel picked different experts"
        )
        max_dw = (
            (weights.gather(1, order.indices) - ref_weights.gather(1, ref_order.indices))
            .abs()
            .max()
            .item()
        )
        print(f"  {name:12s} same experts, max|Δweights| = {max_dw:.3e}")
        assert max_dw < 1e-5, f"{name}: weights diverged ({max_dw})"


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[2**i for i in range(14)],
        line_arg="provider",
        line_vals=list(PROVIDERS),
        line_names=[name.capitalize() for name in PROVIDERS],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="µs (lower is better)",
        plot_name="grouped-topk-performance",
        # decode sends one token, a long prefill sends thousands — the span
        # is three orders of magnitude, so the axis is logarithmic.
        x_log=True,
        args={},
    )
)
def benchmark_grouped_topk(num_tokens: int, provider: str) -> float:
    """Router latency in µs on the DeepSeek-V3 geometry."""
    logits, bias = _make_inputs(V3, num_tokens)
    route = PROVIDERS[provider]
    ms = triton.testing.do_bench(lambda: route(logits, bias))
    return ms * 1e3


if __name__ == "__main__":
    run_perf_report(
        benchmark_grouped_topk,
        verify,
        verify_msg="Verifying the kernel against the torch reference:",
        run_msg="Running benchmark (DeepSeek-V3 geometry)",
    )
