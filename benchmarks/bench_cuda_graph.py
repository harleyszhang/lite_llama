#!/usr/bin/env python
"""Benchmark eager vs CUDA Graph decode, and assert the outputs match.

Prints a small table comparing decode throughput. The comparison isolates the
decode phase (where graphs help) by using a short prompt and a longer generation.

Usage:
    python benchmarks/bench_cuda_graph.py --model-dir my_weight/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import time

import torch
from common import LiteBackend

from lite_llama import SamplingParams, TextGenerator


def _run(generator: TextGenerator, prompt: str, params: SamplingParams, iters: int) -> float:
    """Return median seconds per generate() call over ``iters`` runs (after warmup)."""
    generator.generate([prompt], params)  # warmup / one-time allocations
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        generator.generate([prompt], params)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-gen-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    prompt = "The capital of France is"
    params = SamplingParams(temperature=0.0, max_gen_len=args.max_gen_len)

    # 同进程先后建两个后端:必须显式限制 KV blocks。自动预算会吃到 90% 显存,
    # 且第一个生成器销毁后显存不被引擎回收,第二个生成器会拿不到 KV 空间。
    # 单 prompt + max_seq_len 的场景 8192 个 token block 绰绰有余。
    kv_blocks = 8192
    eager_be = LiteBackend(
        args.model_dir,
        use_cuda_graph=False,
        max_seq_len=args.max_seq_len,
        max_gpu_num_blocks=kv_blocks,
        device="cuda",
    )
    eager = eager_be.generator
    eager_out = eager.generate([prompt], params)[0]
    eager_dt = _run(eager, prompt, params, args.iters)
    eager_be.close()

    graph_be = LiteBackend(
        args.model_dir,
        use_cuda_graph=True,
        max_seq_len=args.max_seq_len,
        max_gpu_num_blocks=kv_blocks,
        device="cuda",
    )
    graph = graph_be.generator
    graph_out = graph.generate([prompt], params)[0]
    graph_dt = _run(graph, prompt, params, args.iters)
    graph_be.close()

    tokens = args.max_gen_len
    print("\n" + "=" * 68)
    print(f"model         : {args.model_dir}")
    print(f"generated     : {tokens} tokens, greedy")
    print(f"outputs match : {eager_out == graph_out}")
    print("-" * 68)
    print(f"{'mode':<14}{'latency (s)':>16}{'tokens/s':>16}{'speedup':>16}")
    print(f"{'eager':<14}{eager_dt:>16.4f}{tokens / eager_dt:>16.1f}{'1.00x':>16}")
    print(
        f"{'cuda-graph':<14}{graph_dt:>16.4f}{tokens / graph_dt:>16.1f}"
        f"{f'{eager_dt / graph_dt:.2f}x':>16}"
    )
    print("=" * 68)

    if eager_out != graph_out:
        print("\nERROR: CUDA Graph output diverged from eager!")
        print("  eager:", repr(eager_out))
        print("  graph:", repr(graph_out))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
