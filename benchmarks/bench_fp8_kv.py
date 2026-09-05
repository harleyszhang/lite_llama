"""O14 fp8 KV cache end-to-end benchmark.

Compares fp8 KV vs fp16 KV on:
1. KV capacity (bytes per token)
2. Generation agreement (greedy token match rate)
3. Throughput

Usage:
    python benchmarks/bench_fp8_kv.py --model-dir <ckpt> --json <path>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.lib.backends import LiteBackend
from benchmarks.lib.workloads import sampling_params


def compare_generations(
    model_dir: str,
    prompts: list[str],
    max_gen_len: int = 32,
) -> dict:
    """Generate with fp16 KV and fp8 KV, compare outputs."""
    results = {}

    # fp16 KV baseline
    backend_fp16 = LiteBackend(model_dir, use_cuda_graph=True, kv_cache_dtype="auto")
    t0 = time.perf_counter()
    step_texts_fp16: list[list[str]] = []
    for deltas in backend_fp16.generator.stream(prompts, sampling_params(max_gen_len)):
        step_texts_fp16.append(list(deltas))
    time_fp16 = time.perf_counter() - t0
    outputs_fp16 = ["".join(step[i] for step in step_texts_fp16) for i in range(len(prompts))]
    backend_fp16.close()
    del backend_fp16
    torch.cuda.empty_cache()

    # fp8 KV
    backend_fp8 = LiteBackend(model_dir, use_cuda_graph=True, kv_cache_dtype="fp8")
    t0 = time.perf_counter()
    step_texts_fp8: list[list[str]] = []
    for deltas in backend_fp8.generator.stream(prompts, sampling_params(max_gen_len)):
        step_texts_fp8.append(list(deltas))
    time_fp8 = time.perf_counter() - t0
    outputs_fp8 = ["".join(step[i] for step in step_texts_fp8) for i in range(len(prompts))]
    backend_fp8.close()
    del backend_fp8
    torch.cuda.empty_cache()

    # Compare
    token_match = 0
    total_tokens = 0
    for a, b in zip(outputs_fp16, outputs_fp8, strict=True):
        toks_a = a.split()
        toks_b = b.split()
        for ta, tb in zip(toks_a, toks_b, strict=False):
            if ta == tb:
                token_match += 1
            total_tokens += 1
        total_tokens += abs(len(toks_a) - len(toks_b))

    results["time_fp16_s"] = round(time_fp16, 3)
    results["time_fp8_s"] = round(time_fp8, 3)
    results["token_match_rate"] = round(token_match / max(total_tokens, 1), 4)
    results["num_prompts"] = len(prompts)
    results["max_gen_len"] = max_gen_len
    return results


def main():
    parser = argparse.ArgumentParser(description="O14 fp8 KV benchmark")
    parser.add_argument("--model-dir", required=True, help="Model checkpoint path")
    parser.add_argument("--json", default=None, help="Output JSON path")
    parser.add_argument("--max-gen-len", type=int, default=32)
    args = parser.parse_args()

    prompts = [
        "The capital of France is",
        "In a universe where magic exists,",
        "def fibonacci(n):\n    if n <= 1:",
        "Q: What is 2+2?\nA:",
        "Once upon a time in a land far away,",
    ]

    print(f"Model: {args.model_dir}")
    print(f"Prompts: {len(prompts)}, max_gen_len: {args.max_gen_len}")
    print()

    results = compare_generations(args.model_dir, prompts, args.max_gen_len)

    # KV capacity comparison (analytical)
    results["kv_bytes_ratio"] = 0.5  # fp8 is half of fp16
    results["kv_capacity_multiplier"] = 2.0  # fp8 doubles capacity

    print(f"fp16 time: {results['time_fp16_s']:.3f}s")
    print(f"fp8  time: {results['time_fp8_s']:.3f}s")
    print(f"Token match rate: {results['token_match_rate']:.2%}")
    print("KV capacity: fp8 = 2x fp16")

    if args.json:
        from benchmarks.lib import write_json_log
        write_json_log(args.json, results, {"benchmark": "fp8_kv"})
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
