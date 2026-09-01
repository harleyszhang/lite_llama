"""Two Triton row-wise softmax kernels vs ``torch.softmax``.

The one-pass naive kernel is compared with a split variant (logsumexp,
then combine) on wide rows; correctness is checked before any timing
loop starts.

Usage:
    python benchmarks/kernels/bench_softmax.py
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable row-wise softmax in plain PyTorch (reference).

    Subtracting the row max avoids overflow; softmax is invariant to the shift.
    Memory traffic is read ``5MN + 2M`` and write ``3MN + 2M`` elements — the
    round trips through HBM that a fused kernel removes by keeping every
    intermediate in SRAM.
    """
    x_max = x.max(dim=1, keepdim=True).values  # read MN, write M
    numerator = torch.exp(x - x_max)  # read MN + M, write MN
    return numerator / numerator.sum(dim=1, keepdim=True)  # read MN + M, write MN


# --------------------------------------------------------------------------- #
# Native: one program per row.
# --------------------------------------------------------------------------- #
@triton.jit
def _softmax_kernel_fwd(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    # Padding lanes load as -inf: they cannot become the row max, and exp(-inf)=0
    # keeps them out of the sum. Loading 0.0 would corrupt an all-negative row.
    row_ptr = input_ptr + row_id * input_row_stride + col_offsets
    x = tl.load(row_ptr, mask=mask, other=-float("inf")).to(tl.float32)

    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    softmax_out = numerator / tl.sum(numerator, axis=0)

    out_ptr = output_ptr + row_id * output_row_stride + col_offsets
    tl.store(out_ptr, softmax_out.to(output_ptr.dtype.element_ty), mask=mask)


def _num_warps(block_size: int) -> int:
    """More warps for wider rows, so a long row still saturates the SMs."""
    if block_size >= 32768:
        return 32
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    return 4


@torch.no_grad()
def softmax_native_fwd(x: torch.Tensor) -> torch.Tensor:
    """Triton row-wise softmax over ``dim=1``; one program per row."""
    assert x.ndim == 2, "softmax_native_fwd only accepts a 2D [rows, cols] tensor"
    rows, cols = x.shape
    block_size = triton.next_power_of_2(cols)
    out = torch.empty_like(x)
    _softmax_kernel_fwd[(rows,)](
        out,
        x,
        x.stride(0),
        out.stride(0),
        cols,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return out


# --------------------------------------------------------------------------- #
# Split: partial log-sum-exp per tile, combined per row, then normalise.
# --------------------------------------------------------------------------- #
@triton.jit
def _logsumexp_kernel(out_ptr, in_ptr, M, N, TILE_N: tl.constexpr):
    pid_n = tl.program_id(0)
    num_programs_n = tl.num_programs(0)
    pid_m = tl.program_id(1)

    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    offset = pid_m * N + n_offsets
    inp = tl.load(in_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(inp, 0)
    z = tl.sum(tl.exp(inp - m), 0)
    logz = m + tl.log(z)

    tl.store(out_ptr + pid_m * num_programs_n + pid_n, logz.to(out_ptr.dtype.element_ty))


@triton.jit
def _combine_logsumexp_kernel(out_ptr, inp_ptr, M, N, TILE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N
    logzs = tl.load(inp_ptr + pid_m * N + n_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(logzs, 0)
    z = tl.sum(tl.exp(logzs - m), 0)
    tl.store(out_ptr + pid_m, (m + tl.log(z)).to(out_ptr.dtype.element_ty))


@triton.jit
def _softmax_split_kernel(out_ptr, in_ptr, logz_ptr, M, N, TILE_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    offset = pid_m * N + n_offsets
    mask = n_offsets < N
    inp = tl.load(in_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
    logz = tl.load(logz_ptr + pid_m).to(tl.float32)
    tl.store(out_ptr + offset, tl.exp(inp - logz).to(out_ptr.dtype.element_ty), mask=mask)


@torch.no_grad()
def softmax_split(x: torch.Tensor) -> torch.Tensor:
    """Triton row-wise softmax over ``dim=1``; a long row is tiled across programs."""
    assert x.ndim == 2, "softmax_split only accepts a 2D [rows, cols] tensor"
    M, N = x.shape

    tile_n = min(4096, triton.next_power_of_2(N))
    num_tiles_n = triton.cdiv(N, tile_n)

    logz = torch.empty((M, num_tiles_n), dtype=x.dtype, device=x.device)
    _logsumexp_kernel[(num_tiles_n, M, 1)](logz, x, M, N, tile_n)

    combined_logz = torch.empty((M,), dtype=x.dtype, device=x.device)
    _combine_logsumexp_kernel[(M, 1, 1)](
        combined_logz, logz, M, num_tiles_n, triton.next_power_of_2(num_tiles_n)
    )

    out = torch.empty_like(x)
    _softmax_split_kernel[(num_tiles_n, M, 1)](out, x, combined_logz, M, N, tile_n)
    return out


#: Implementations compared by the benchmark. Add a row to register a new one.
PROVIDERS: dict[str, callable] = {
    "torch": lambda x: torch.softmax(x, dim=-1),
    "naive": naive_softmax,
    "triton": softmax_native_fwd,
    "split": softmax_split,
}


def verify(rows: int = 4, cols: int = 2048, device: str = "cuda") -> None:
    """Check every provider against ``torch.softmax`` before timing it."""
    x = torch.randn(rows, cols, device=device, dtype=torch.float32)
    reference = torch.softmax(x, dim=-1)
    for name, fn in PROVIDERS.items():
        max_err = (fn(x).float() - reference).abs().max().item()
        print(f"  {name:8s} max|Δ| vs torch = {max_err:.3e}")
        assert max_err < 1e-5, f"{name} softmax diverged from torch: {max_err}"

    # All-negative row: the padding-lane bug (loading 0.0) would surface here.
    neg = torch.full((2, 1536), -5.0, device=device, dtype=torch.float32)
    expected = torch.softmax(neg, dim=-1)
    for name in ("triton", "split"):
        assert torch.allclose(PROVIDERS[name](neg), expected, atol=1e-6), name
    print("  edge case (all-negative row) OK")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=list(PROVIDERS),
        line_names=[name.capitalize() for name in PROVIDERS],
        styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("purple", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 2048},
    )
)
def benchmark_softmax(M: int, N: int, provider: str) -> float:
    """Effective HBM bandwidth (GB/s): softmax reads and writes ``MN`` elements."""
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    ms = triton.testing.do_bench(lambda: PROVIDERS[provider](x))
    return 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")
    save_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../images/benchmark_result")
    )
    os.makedirs(save_path, exist_ok=True)

    print("Verifying correctness against torch.softmax:")
    verify()
    print(f"\nRunning benchmark, saving plot to {save_path}")
    benchmark_softmax.run(print_data=True, save_path=save_path)
