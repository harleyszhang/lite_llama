"""Micro cost-model probe for TBO: does halving the batch double the GEMM cost?

For decode (small M) the row-parallel GEMMs are weight-memory-bound: the time
is dominated by reading the weight matrix, not by the M rows. If that holds,
splitting batch M into two halves of M/2 makes each GEMM re-read the full
weight shard, so the *pair* of half GEMMs costs ~2x the single full GEMM —
which is exactly the penalty TBO pays before it ever hides an all-reduce.

This probe measures, on one A10, for the real Qwen2.5-1.5B TP2 shapes:
  * o_proj / down_proj GEMM time at M and at M/2 (x2),
  * the all-reduce wire time for the [M, hidden] partial on TP2,
so the TBO trade-off (2x compute vs hidden AR) can be read off directly.

Usage:
    python benchmarks/probe_tbo_cost_model.py --batch 16
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

HIDDEN = 1536
INTER = 8960
N_HEADS = 12
HEAD_DIM = 128
N_LAYERS = 28
TP = 2


def time_gemm(weight: torch.Tensor, x: torch.Tensor, iters: int = 200) -> float:
    """Median us of ``x @ weight`` (bf16), sync-bounded."""
    for _ in range(20):
        x @ weight
    torch.cuda.synchronize()
    lat = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x @ weight
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(lat)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    M = args.batch

    dev = "cuda"
    # Per-rank weight shards for the two row-parallel GEMMs (TP2).
    # o_proj: contraction dim = n_heads*head_dim, split by TP.
    o_in = N_HEADS * HEAD_DIM // TP  # 768
    w_o = torch.randn(o_in, HIDDEN, device=dev, dtype=torch.bfloat16)
    # down_proj: contraction dim = intermediate, split by TP.
    d_in = INTER // TP  # 4480
    w_d = torch.randn(d_in, HIDDEN, device=dev, dtype=torch.bfloat16)
    # Column-parallel GEMMs also double under a split (no AR, but same 2x read).
    # qkv_proj contraction = hidden (not split on input); gate_up contraction = hidden.
    w_qkv = torch.randn(HIDDEN, (N_HEADS + 2 * 2) * HEAD_DIM // TP, device=dev, dtype=torch.bfloat16)
    w_gu = torch.randn(HIDDEN, 2 * INTER // TP, device=dev, dtype=torch.bfloat16)

    print(f"=== TBO cost-model probe, Qwen2.5-1.5B TP2 shapes, batch M={M} ===\n")

    rows = []
    for name, w, full_in in [
        ("o_proj (row-par)", w_o, o_in),
        ("down_proj (row-par)", w_d, d_in),
        ("qkv_proj (col-par)", w_qkv, HIDDEN),
        ("gate_up (col-par)", w_gu, HIDDEN),
    ]:
        x_full = torch.randn(M, full_in, device=dev, dtype=torch.bfloat16)
        x_half = torch.randn(M // 2, full_in, device=dev, dtype=torch.bfloat16)
        t_full = time_gemm(w, x_full)
        t_half = time_gemm(w, x_half)
        t_two_halves = 2 * t_half
        ratio = t_two_halves / t_full
        rows.append((name, t_full, t_half, t_two_halves, ratio))
        print(
            f"{name:22s} M={M:3d}: {t_full:7.2f} us | "
            f"M={M//2:3d}: {t_half:7.2f} us | 2x halves: {t_two_halves:7.2f} us "
            f"({ratio:.2f}x of full)"
        )

    total_full = sum(r[1] for r in rows)
    total_halves = sum(r[3] for r in rows)
    print(
        f"\nper-layer GEMM total: full {total_full:.2f} us vs 2x halves "
        f"{total_halves:.2f} us -> {total_halves/total_full:.2f}x"
    )
    print(
        f"whole model ({N_LAYERS} layers) compute: full {total_full*N_LAYERS/1000:.2f} ms "
        f"vs 2x halves {total_halves*N_LAYERS/1000:.2f} ms "
        f"(+{(total_halves-total_full)*N_LAYERS/1000:.2f} ms of doubled weight reads)"
    )

    # All-reduce wire time for the [M, hidden] partial, single-GPU loopback is
    # not representative; report the payload size so the AR fraction can be
    # reasoned about (real AR needs the TP2 engine, measured in bench_overlap_l2).
    payload = M * HIDDEN * 2  # bf16
    n_ar = 2 * N_LAYERS
    print(
        f"\nAR payload per reduce: {payload/1024:.1f} KiB; {n_ar} reduces/step; "
        f"total AR bytes/step {payload*n_ar/1024/1024:.2f} MiB"
    )
    print(
        "NOTE: AR wire time must be measured on the TP2 engine (NCCL), not here."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
