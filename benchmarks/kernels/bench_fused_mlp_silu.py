"""Fused SwiGLU MLP: silu(x @ w1) * (x @ w2) @ w3 — three GEMMs, one kernel.

The Triton kernel fuses the up/gate GEMMs with the silu-multiply
epilogue; ``_check`` diffs against the eager chain before anything is
timed.

Usage:
    python benchmarks/kernels/bench_fused_mlp_silu.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from microbench import Row, Work, bench, metadata, report, verify

from lite_llama.kernels.ops.activation.swiglu import swiglu_forward

#: Grouped-pid tile sizes, shared by every launch below (bumping these is part
#: of tuning the kernel, not of running the benchmark).
BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 64, 64, 128, 8


@triton.jit
def mlp_kernel(
    a_ptr,
    w1_ptr,
    w2_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_w1k,
    stride_w1n,
    stride_w2k,
    stride_w2n,
    stride_cm,
    stride_cn,
    FUSE_SILU: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """One grouped GEMM. FUSE_SILU folds silu(a@w1) * (a@w2) into the epilogue.

    A has shape (M, K), the weights (K, N), C (M, N). With FUSE_SILU the kernel
    walks w1 and w2 down K side by side and writes silu(acc1) * acc2; without it
    w2 is unused and the fp32 accumulator is cast to fp16 for the store — the
    down-projection in :func:`mlp_silu`.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    if FUSE_SILU:
        w2_ptrs = w2_ptr + (offs_k[:, None] * stride_w2k + offs_bn[None, :] * stride_w2n)

    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if FUSE_SILU:
        acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w1 = tl.load(w1_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc1 += tl.dot(a, w1)
        if FUSE_SILU:
            w2 = tl.load(w2_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            acc2 += tl.dot(a, w2)
            w2_ptrs += BLOCK_SIZE_K * stride_w2k
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k

    if FUSE_SILU:
        c = (acc1 * tl.sigmoid(acc1)) * acc2
    else:
        c = acc1.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _launch(
    a: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, out: torch.Tensor, fuse_silu: bool
) -> torch.Tensor:
    """Launch :func:`mlp_kernel` on flat 2D operands; shapes must be contiguous."""
    M, K = a.shape
    N = w1.shape[1]
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    mlp_kernel[grid](
        a,
        w1,
        w2,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        w1.stride(0),
        w1.stride(1),
        w2.stride(0),
        w2.stride(1),
        out.stride(0),
        out.stride(1),
        FUSE_SILU=fuse_silu,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=2,
        num_warps=4,
    )
    return out


def _fused_silu_up(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """gate/up projections + silu-mul epilogue in one kernel: (M, K) -> (M, N)."""
    M = x.shape[0]
    return _launch(x, w1, w2, torch.empty((M, w1.shape[1]), device=x.device, dtype=x.dtype), True)


def _check(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> tuple[int, int, int]:
    assert x.shape[-1] == w1.shape[0] == w2.shape[0], "Incompatible dimensions"
    assert w1.shape == w2.shape, "Incompatible dimensions"
    assert x.is_contiguous() and w1.is_contiguous() and w2.is_contiguous(), "Must be contiguous"
    return x.numel() // x.shape[-1], w1.shape[0], w1.shape[1]  # M, K, N


def mlp_silu(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """All-Triton: fused gate/up GEMM, then the down projection as a plain GEMM."""
    batch, seq_len, _ = x.shape
    M, K, _ = _check(x, w1, w2)
    x = x.view(M, K)
    up = _fused_silu_up(x, w1, w2)
    out = _launch(up, w3, w3, torch.empty((M, w3.shape[1]), device=x.device, dtype=x.dtype), False)
    return out.view(batch, seq_len, -1)


def triton_torch_mlp_silu(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor
) -> torch.Tensor:
    """Triton for the fused gate/up GEMM, cuBLAS for the down projection."""
    batch, seq_len, _ = x.shape
    M, K, _ = _check(x, w1, w2)
    up = _fused_silu_up(x.view(M, K), w1, w2)
    return torch.mm(up, w3).view(batch, seq_len, -1)


def torch_mlp(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor
) -> torch.Tensor:
    """Eager reference: three ``torch.mm`` plus the package's fused swiglu."""
    batch, seq_len, dim = x.shape
    M = batch * seq_len
    x = x.view(M, dim)
    out = swiglu_forward(torch.mm(x, w1), torch.mm(x, w2))
    return torch.mm(out, w3).view(batch, seq_len, -1)


class FusedMLP(nn.Module):
    """The eager path dressed as a module — kept as a correctness reference only,
    since its timing would be the same data point as ``torch_mlp``."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu_forward(self.gate_proj(x), self.up_proj(x)))


#: Implementations compared by the benchmark. Add a row to register a new one.
PROVIDERS: dict[
    str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
] = {
    "torch": torch_mlp,
    "fused-triton": mlp_silu,
    "hybrid": triton_torch_mlp_silu,
}

#: (label, batch, seq_len) — Qwen2.5-7B-sized MLP (hidden 3584, intermediate 18944).
CASES = [
    ("b4_s256", 4, 256),
    ("b8_s1024", 8, 1024),
]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")
    print(metadata())

    hidden_size, intermediate_size = 3584, 18944
    torch.manual_seed(0)
    w1 = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16) * 0.01
    w2 = torch.randn_like(w1) * 0.01
    w3 = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=torch.float16) * 0.01
    # The Triton kernels index weights as (K, N); torch.mm wants (K, N) too.
    w1, w2, w3 = w1.t().contiguous(), w2.t().contiguous(), w3.t().contiguous()

    module = FusedMLP(hidden_size, intermediate_size).cuda()
    rows: list[Row] = []
    for label, batch, seq_len in CASES:
        x = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=torch.float16)
        reference = torch_mlp(x, w1, w2, w3)

        print(f"\n{label}: verifying against torch_mlp before timing anything")
        for name, fn in PROVIDERS.items():
            verify(name, fn(x, w1, w2, w3), reference, rtol=1e-2, atol=1e-2)
        module.load_state_dict(
            {"gate_proj.weight": w1.t(), "up_proj.weight": w2.t(), "down_proj.weight": w3.t()}
        )
        verify("nn.module", module(x), reference, rtol=1e-2, atol=1e-2)

        M, K, N = batch * seq_len, hidden_size, intermediate_size
        # Three GEMMs of 2*M*N*K FLOP each; every byte read once, output written once.
        work = Work(flops=6 * M * N * K, moved=2 * (2 * M * K + 3 * N * K))
        for name, fn in PROVIDERS.items():
            rows.append(Row(name, label, bench(lambda fn=fn, x=x: fn(x, w1, w2, w3)), work))

    print()
    report(rows)


if __name__ == "__main__":
    main()
