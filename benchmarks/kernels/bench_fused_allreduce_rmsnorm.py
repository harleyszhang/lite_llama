"""O11 communication-RMSNorm fusion kernel benchmark (TP=2 required).

Compares:
- Baseline: all-reduce + fused_add_rmsnorm (two separate ops)
- O11 fusion: fused_allreduce_rmsnorm (reduce-scatter + norm + all-gather)

Usage:
    torchrun --nproc_per_node=2 benchmarks/kernels/bench_fused_allreduce_rmsnorm.py [--json path]

Note:
    On TP=2 A10, O11 fusion is 2.5x-9.4x SLOWER than baseline due to:
    1. Two communication ops (reduce-scatter + all-gather) vs one all-reduce
    2. No async communication-computation overlap in current implementation
    3. Small message sizes (4KB-1MB) where latency dominates

    O11 may be beneficial with TP>2, larger messages (prefill), or async overlap.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.distributed as dist

from rapid_llm.kernels.ops.layernorm.skip_rmsnorm import (
    fused_add_rmsnorm,
    fused_allreduce_rmsnorm,
)


@contextlib.contextmanager
def _skip_allreduce():
    """Context to skip all-reduce in row-parallel linear (for O11 fusion)."""
    from rapid_llm.batch_overlap import comm_overlap as _co
    token = _co._skip_allreduce.set(True)
    try:
        yield
    finally:
        _co._skip_allreduce.reset(token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None, help="Output JSON path")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)

    # Simple distributed init (avoid init_tensor_parallel which creates extra groups)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Manually set TP state
    from rapid_llm.distributed import parallel_state as ps
    ps._TP_RANK = rank
    ps._TP_WORLD_SIZE = world_size
    ps._TP_GROUP = dist.group.WORLD

    # Skip shapes that trigger fallback (total_tokens < world_size or not divisible)
    configs = [
        (4, 2048),    # 4 tokens, 4 % 2 = 0
        (16, 4096),   # 16 tokens, 16 % 2 = 0
        (32, 4096),   # 32 tokens, 32 % 2 = 0
        (64, 8192),   # 64 tokens, 64 % 2 = 0
    ]

    if rank == 0:
        print(f"O11 Benchmark (TP={world_size}, {torch.cuda.get_device_name()})")
        print("=" * 75)
        print("Baseline: all-reduce + fused_add_rmsnorm")
        print("O11:      reduce-scatter + RMSNorm + all-gather")
        print("=" * 75)
        print(f"{'shape':<20} {'baseline(μs)':<15} {'O11(μs)':<15} {'speedup':<10}")
        print("-" * 75)

    results = []
    for batch_seq, hidden in configs:
        shape = (batch_seq, hidden)
        dtype = torch.float16
        device = f"cuda:{local_rank}"

        partial = torch.randn(shape, dtype=dtype, device=device)
        residual = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.ones(hidden, dtype=dtype, device=device)
        eps = 1e-5

        # Warmup
        for _ in range(10):
            _ = fused_add_rmsnorm(partial.clone(), residual.clone(), weight, eps)
            with _skip_allreduce():
                _ = fused_allreduce_rmsnorm(partial.clone(), residual.clone(), weight, eps)
        torch.cuda.synchronize()

        n_iters = 200

        # --- Baseline: all-reduce + fused_add_rmsnorm ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            x = partial.clone()
            r = residual.clone()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            y, _ = fused_add_rmsnorm(x, r, weight, eps)
        torch.cuda.synchronize()
        time_baseline = (time.perf_counter() - t0) / n_iters * 1e6

        # --- O11 fusion: fused_allreduce_rmsnorm ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            x = partial.clone()
            r = residual.clone()
            with _skip_allreduce():
                y, _ = fused_allreduce_rmsnorm(x, r, weight, eps)
        torch.cuda.synchronize()
        time_fused = (time.perf_counter() - t0) / n_iters * 1e6

        speedup = time_baseline / time_fused

        if rank == 0:
            results.append({
                "shape": f"({batch_seq}, {hidden})",
                "batch_seq": batch_seq,
                "hidden": hidden,
                "baseline_us": round(time_baseline, 3),
                "fused_us": round(time_fused, 3),
                "speedup": round(speedup, 3),
            })
            print(
                f"{str(shape):<20} {time_baseline:<15.3f} {time_fused:<15.3f} {speedup:<10.3f}x"
            )

    dist.destroy_process_group()

    if rank == 0:
        print("-" * 75)
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        max_speedup = max(r["speedup"] for r in results)
        min_speedup = min(r["speedup"] for r in results)
        print(f"Average speedup: {avg_speedup:.3f}x")
        print(f"Max speedup: {max_speedup:.3f}x")
        print(f"Min speedup: {min_speedup:.3f}x")

        output = {
            "benchmark": "fused_allreduce_rmsnorm_o11",
            "description": "O11 fusion (reduce-scatter+norm+all-gather) vs baseline (all-reduce+add+norm) on TP=2",
            "tp_size": world_size,
            "gpu": torch.cuda.get_device_name(),
            "results": results,
            "summary": {
                "avg_speedup": round(avg_speedup, 3),
                "max_speedup": round(max_speedup, 3),
                "min_speedup": round(min_speedup, 3),
            },
            "note": "O11 is slower on TP=2 due to 2x communication ops and no async overlap",
        }

        if args.json:
            with open(args.json, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
