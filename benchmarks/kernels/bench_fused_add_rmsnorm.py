"""O11 communication-RMSNorm fusion kernel benchmark.

Compares fused_add_rmsnorm vs separate skip_rmsnorm on decode shapes.
The fused kernel eliminates one HBM read of the residual tensor.

Usage:
    python benchmarks/kernels/bench_fused_add_rmsnorm.py --json <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import triton

from rapid_llm.kernels.ops.layernorm.skip_rmsnorm import (
    fused_add_rmsnorm,
    skip_rmsnorm,
)


def bench_fused_add_rmsnorm():
    """Benchmark fused_add_rmsnorm vs skip_rmsnorm on decode shapes."""
    results = []
    
    # Decode shapes: (batch * seq_len, hidden_size)
    # Typical decode: batch=1-64, seq_len=1, hidden=2048-8192
    configs = [
        (1, 2048),    # batch=1, hidden=2048
        (4, 2048),    # batch=4
        (16, 4096),   # batch=16, hidden=4096
        (32, 4096),   # batch=32
        (64, 8192),   # batch=64, hidden=8192
    ]
    
    for batch_seq, hidden in configs:
        shape = (batch_seq, hidden)
        dtype = torch.float16
        device = "cuda"
        
        x = torch.randn(shape, dtype=dtype, device=device)
        residual = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.ones(hidden, dtype=dtype, device=device)
        eps = 1e-5
        
        # Warmup
        for _ in range(10):
            _ = skip_rmsnorm(x, residual, weight, eps)
            _ = fused_add_rmsnorm(x, residual, weight, eps)
        torch.cuda.synchronize()
        
        # Benchmark skip_rmsnorm (separate path)
        n_iters = 1000
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = skip_rmsnorm(x, residual, weight, eps)
        torch.cuda.synchronize()
        time_skip = (time.perf_counter() - t0) / n_iters * 1e6  # μs
        
        # Benchmark fused_add_rmsnorm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = fused_add_rmsnorm(x, residual, weight, eps)
        torch.cuda.synchronize()
        time_fused = (time.perf_counter() - t0) / n_iters * 1e6  # μs
        
        speedup = time_skip / time_fused
        
        results.append({
            "shape": f"({batch_seq}, {hidden})",
            "batch_seq": batch_seq,
            "hidden": hidden,
            "skip_rmsnorm_us": round(time_skip, 3),
            "fused_add_rmsnorm_us": round(time_fused, 3),
            "speedup": round(speedup, 3),
        })
        
        print(f"shape={shape}: skip={time_skip:.3f}μs, fused={time_fused:.3f}μs, speedup={speedup:.3f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="O11 fused_add_rmsnorm kernel benchmark")
    parser.add_argument("--json", default=None, help="Output JSON path")
    args = parser.parse_args()
    
    print("Benchmarking O11 communication-RMSNorm fusion...")
    print("=" * 70)
    
    results = bench_fused_add_rmsnorm()
    
    output = {
        "benchmark": "fused_add_rmsnorm_o11",
        "description": "fused_add_rmsnorm vs skip_rmsnorm on decode shapes",
        "results": results,
        "summary": {
            "avg_speedup": round(sum(r["speedup"] for r in results) / len(results), 3),
            "max_speedup": round(max(r["speedup"] for r in results), 3),
            "min_speedup": round(min(r["speedup"] for r in results), 3),
        },
    }
    
    print("\n" + "=" * 70)
    print(f"Average speedup: {output['summary']['avg_speedup']:.3f}x")
    print(f"Max speedup: {output['summary']['max_speedup']:.3f}x")
    print(f"Min speedup: {output['summary']['min_speedup']:.3f}x")
    
    if args.json:
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
