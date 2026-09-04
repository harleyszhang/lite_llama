"""Benchmark ngram speculative decoding (O5).

Measures the throughput impact of ngram-based speculative decoding on a
repetitive workload using the ContinuousBatchingEngine directly.

Usage:
    python benchmarks/bench_speculative.py --model-dir my_weight/Qwen3-0.6B
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch


def _repetitive_prompts() -> list[str]:
    """Prompts with repeated structure that ngram speculation can exploit."""
    base = "Write a Python function that calculates the sum of two numbers. "
    repeat = "def add(a, b): return a + b. "
    return [base + repeat * 5 for _ in range(4)]


def _run_engine(
    model_dir: str,
    prompts: list[str],
    max_gen_len: int = 64,
    speculate: bool = False,
) -> dict:
    """Run the ContinuousBatchingEngine with or without speculation."""
    from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
    from rapid_llm.engine.sampler import SamplingParams

    if speculate:
        os.environ["LITE_LLAMA_SPECULATE"] = "1"
    else:
        os.environ.pop("LITE_LLAMA_SPECULATE", None)

    engine = ContinuousBatchingEngine.from_pretrained(model_dir)
    params = SamplingParams(temperature=0.0, max_gen_len=max_gen_len)

    # Warm up.
    warmup_prompts = ["Hello"]
    engine.generate(warmup_prompts, SamplingParams(temperature=0.0, max_gen_len=4))

    # Measure.
    for p in prompts:
        engine.add_request(p, params)

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    total_tokens = 0
    step_count = 0

    while engine.has_unfinished_requests():
        requests = engine.step()
        for req in requests:
            if req.delta:
                total_tokens += 1
        step_count += 1

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start

    engine.shutdown()
    os.environ.pop("LITE_LLAMA_SPECULATE", None)

    return {
        "total_tokens": total_tokens,
        "total_steps": step_count,
        "total_time_s": round(total_time, 4),
        "tokens_per_second": round(total_tokens / max(total_time, 1e-6), 1),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark ngram speculative decoding (O5)")
    parser.add_argument("--model-dir", type=str, default="my_weight/Qwen3-0.6B")
    parser.add_argument("--max-gen-len", type=int, default=64)
    parser.add_argument("--json", type=str, default=None, help="Output JSON log path")
    args = parser.parse_args()

    prompts = _repetitive_prompts()
    print(f"Model: {args.model_dir}")
    print(f"Prompts: {len(prompts)}, max_gen_len: {args.max_gen_len}")
    print()

    # Baseline: no speculation.
    print("Running baseline (no speculation)...")
    baseline = _run_engine(args.model_dir, prompts, args.max_gen_len, speculate=False)
    print(f"  {baseline['tokens_per_second']} tok/s, "
          f"{baseline['total_steps']} steps, "
          f"{baseline['total_time_s']}s")

    # Speculative: ngram proposal + verify.
    print("Running speculative (ngram)...")
    speculative = _run_engine(args.model_dir, prompts, args.max_gen_len, speculate=True)
    print(f"  {speculative['tokens_per_second']} tok/s, "
          f"{speculative['total_steps']} steps, "
          f"{speculative['total_time_s']}s")

    speedup = baseline["total_time_s"] / max(speculative["total_time_s"], 1e-6)
    step_reduction = 1 - speculative["total_steps"] / max(baseline["total_steps"], 1)

    print()
    print("=== Results ===")
    print(f"Baseline:    {baseline['tokens_per_second']} tok/s "
          f"({baseline['total_steps']} steps, {baseline['total_time_s']}s)")
    print(f"Speculative: {speculative['tokens_per_second']} tok/s "
          f"({speculative['total_steps']} steps, {speculative['total_time_s']}s)")
    print(f"Wall speedup: {speedup:.2f}x")
    print(f"Step reduction: {step_reduction:.1%}")

    # Save results.
    if args.json:
        log_path = Path(args.json)
    else:
        log_dir = Path(__file__).parent.parent / "docs" / "benchmark_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"speculative_o5_{ts}.json"

    log_path.write_text(json.dumps({
        "config": {
            "model_dir": args.model_dir,
            "max_gen_len": args.max_gen_len,
            "num_prompts": len(prompts),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "results": {
            "baseline": baseline,
            "speculative": speculative,
            "speedup": round(speedup, 2),
            "step_reduction": round(step_reduction, 4),
        },
    }, indent=2))
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
