"""The MoE router GEMM: fp32 SGEMM path vs the bf16 tensor-core tier-4 path.

The router computes ``x @ gate_weight.T`` into per-token expert logits, which
feed topk selection. The logits must land in fp32 — a bf16 output can flip a
topk pick on near-ties, and a wrong expert costs far more than the precision.
Two ways to get there:

* fp32 SGEMM (old): widen ``x`` and the gate weight to fp32, then a scalar
  ``simt`` SGEMM. Profiler shows three kernels per call — two bf16->fp32 copy
  kernels for the operands, a non-tensor-core SGEMM, and a split-K reduce —
  plus a persistent fp32 copy of the gate weight.
* bf16 tensor-core (tier-4): ``torch.mm(x, w.T, out_dtype=fp32)`` keeps both
  operands bf16 and lets cuBLAS accumulate and emit fp32 inside one tensor-op
  GEMM epilogue. Profiler shows a single ``nvjet`` kernel, no copies, no
  widened-weight buffer. This is vllm's router GateLinear tier-4 path.

Both produce fp32 logits. This benchmark first verifies the topk picks are
identical between the two (the substitution must not move routing), then times
them on the Qwen3-30B-A3B router geometry (hidden 2048, 128 experts) from
single-token decode to 8k-token prefill.

Usage:
    python benchmarks/kernels/bench_router_gemm.py
"""

from __future__ import annotations

import torch
import triton
from microbench import run_perf_report

# --------------------------------------------------------------------------- #
# Router geometries: [num_experts, hidden] gate weights the checkpoints ship.
# --------------------------------------------------------------------------- #
GEOMETRIES: dict[str, dict] = {
    # Qwen3-30B-A3B-Instruct: the MoE model in the decode A/B log.
    "qwen3_30b_a3b": {"hidden": 2048, "num_experts": 128},
    # DeepSeek-V2-Lite: 160 routed experts on a 2048 residual stream.
    "deepseek_v2_lite": {"hidden": 2048, "num_experts": 160},
}

#: The timed geometry — the MoE checkpoint the decode A/B log measured.
GEOM = GEOMETRIES["qwen3_30b_a3b"]
TOP_K = 8


def _make_inputs(geom: dict, num_tokens: int, device: str = "cuda"):
    """A bf16 gate weight ``[E, H]`` and bf16 activations ``[T, H]``."""
    w = (torch.randn(geom["num_experts"], geom["hidden"], device=device) * 0.02).to(torch.bfloat16)
    x = torch.randn(num_tokens, geom["hidden"], device=device).to(torch.bfloat16)
    return x, w


def _fp32_sgemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The old path: widen both operands, scalar fp32 SGEMM."""
    return torch.nn.functional.linear(x.float(), w.float())


def _tier4_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The tier-4 path: bf16 operands, cuBLAS fp32 accumulate + fp32 output."""
    return torch.mm(x, w.t(), out_dtype=torch.float32)


#: The timed implementations. Both consume the same ``(x, w)`` pair.
PROVIDERS: dict[str, callable] = {"fp32_sgemm": _fp32_sgemm, "tier4_bf16": _tier4_bf16}


def verify(device: str = "cuda") -> None:
    """The two paths must pick the same experts before either is timed.

    Order-free: both emit their picks by descending softmax weight, so the
    comparison sorts the ids and requires identical expert sets. A flip here
    would mean the tier-4 substitution moved routing, which it must not.
    """
    torch.manual_seed(0)
    for name, geom in GEOMETRIES.items():
        x, w = _make_inputs(geom, num_tokens=256, device=device)
        lo, ln = _fp32_sgemm(x, w), _tier4_bf16(x, w)
        assert lo.dtype == ln.dtype == torch.float32, "both paths must emit fp32 logits"
        _, ids_o = torch.topk(torch.softmax(lo, dim=-1), TOP_K, dim=-1)
        _, ids_n = torch.topk(torch.softmax(ln, dim=-1), TOP_K, dim=-1)
        assert torch.equal(ids_o.sort(-1).values, ids_n.sort(-1).values), f"{name}: topk flipped"
        max_dlogit = (lo - ln).abs().max().item()
        print(f"  {name:18s} same top-{TOP_K} experts, max|Δlogit| = {max_dlogit:.3e}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[2**i for i in range(14)],
        line_arg="provider",
        line_vals=list(PROVIDERS),
        line_names=["fp32 SGEMM", "bf16 tier-4"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="µs (lower is better)",
        plot_name="router-gemm-performance",
        # decode sends one token, a long prefill sends thousands — three orders
        # of magnitude, so the axis is logarithmic.
        x_log=True,
        args={},
    )
)
def benchmark_router_gemm(num_tokens: int, provider: str) -> float:
    """Router-GEMM latency in µs on the Qwen3-30B-A3B geometry."""
    x, w = _make_inputs(GEOM, num_tokens)
    ms = triton.testing.do_bench(lambda: PROVIDERS[provider](x, w))
    return ms * 1e3


if __name__ == "__main__":
    run_perf_report(
        benchmark_router_gemm,
        verify,
        verify_msg="Verifying topk parity between the two router-GEMM paths:",
        run_msg=f"Running benchmark ({GEOM})",
    )
