"""Quantization speed + precision + memory benchmark.

Compares lite_llama quantization schemes against HuggingFace fp16 as baseline.
Measures: TTFT, TPOT, TPS, peak GPU memory, greedy token match rate.

Usage:
    python benchmarks/bench_quant.py --model-dir /data/shared/llm_weights/Qwen3-0.6B
    python benchmarks/bench_quant.py --model-dir /data/shared/llm_weights/Qwen3-0.6B --json results.json
    python benchmarks/bench_quant.py --all  # run all representative models
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Ensure the benchmarks package is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.common import (
    BenchResult,
    HFBackend,
    LiteBackend,
    PROMPTS,
    expand_prompts,
    print_table,
)

_MAX_GEN = 64
_BATCH = 4


@dataclass
class QuantBenchResult:
    """Extended result including memory and precision metrics."""

    config_label: str
    model: str
    speed: BenchResult
    peak_mem_gb: float
    model_mem_gb: float = 0.0  # Model weights only
    kv_cache_tokens: int = 0   # KV cache capacity in tokens
    # Precision (vs HF fp16 greedy): fraction of generated tokens matching.
    token_match_rate: float | None = None


def _peak_mem_gb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**3)


def _measure_lite(
    model_dir: str,
    quantization: str | None = None,
    kv_cache_dtype: str = "auto",
    tensor_parallel_size: int = 1,
) -> tuple[BenchResult, float, float, int, list[str]]:
    """Run lite_llama and return (result, peak_mem_gb, model_mem_gb, kv_tokens, texts)."""
    torch.cuda.reset_peak_memory_stats()
    kwargs: dict = {}
    if quantization:
        kwargs["quantization"] = quantization
    if kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = kv_cache_dtype
    if tensor_parallel_size > 1:
        kwargs["tensor_parallel_size"] = tensor_parallel_size

    backend = LiteBackend(model_dir, use_cuda_graph=True, **kwargs)
    prompts = expand_prompts(PROMPTS, _BATCH)

    # Measure greedy speed
    result = backend.measure(prompts, _MAX_GEN, greedy=True)
    peak = _peak_mem_gb()

    # Extract model weight memory and KV cache capacity
    try:
        runner = backend.generator.engine.model_runner
        model_bytes = sum(p.numel() * p.element_size() for p in runner.model.parameters())
        model_mem_gb = model_bytes / (1024**3)
        kv_tokens = runner.kv_cache_manager.gpu_kv_buffer[0].shape[0]
    except Exception:
        model_mem_gb = 0.0
        kv_tokens = 0

    # Collect generated texts for precision comparison
    from lite_llama import SamplingParams

    gen = backend.generator
    texts = gen.generate(prompts, SamplingParams(temperature=0.0, max_gen_len=_MAX_GEN))

    backend.close()
    return result, peak, model_mem_gb, kv_tokens, texts


def _measure_hf(model_dir: str) -> tuple[BenchResult, float, list[str]]:
    """Run HF transformers fp16 baseline."""
    torch.cuda.reset_peak_memory_stats()
    backend = HFBackend(model_dir)
    prompts = expand_prompts(PROMPTS, _BATCH)
    result = backend.measure(prompts, _MAX_GEN, greedy=True)
    peak = _peak_mem_gb()

    # Get generated texts
    texts_tensor = backend._last_gen
    texts = [backend.tokenizer.decode(t, skip_special_tokens=True) for t in texts_tensor]

    del backend
    gc.collect()
    torch.cuda.empty_cache()
    return result, peak, texts


def _token_match_rate(lite_texts: list[str], hf_texts: list[str]) -> float:
    """Compute fraction of tokens matching between two text lists."""
    total, match = 0, 0
    for lt, ht in zip(lite_texts, hf_texts):
        lt_toks = lt.split()
        ht_toks = ht.split()
        n = min(len(lt_toks), len(ht_toks))
        total += n
        match += sum(1 for a, b in zip(lt_toks[:n], ht_toks[:n]) if a == b)
    return match / total if total > 0 else 0.0


def benchmark_model(
    model_dir: str,
    schemes: list[str | None],
    tp: int = 1,
    skip_hf: bool = False,
) -> list[QuantBenchResult]:
    """Run all requested schemes on one model."""
    model_name = Path(model_dir).name
    results: list[QuantBenchResult] = []

    # HF baseline
    hf_texts: list[str] | None = None
    if not skip_hf:
        print(f"\n{'='*60}")
        print(f"  {model_name} — HF fp16 baseline")
        print(f"{'='*60}")
        hf_res, hf_mem, hf_texts = _measure_hf(model_dir)
        results.append(QuantBenchResult(
            config_label="HF fp16",
            model=model_name,
            speed=hf_res,
            peak_mem_gb=hf_mem,
        ))
        print(f"  Peak mem: {hf_mem:.2f} GB | TPS: {hf_res.tps:.1f}")

    for scheme in schemes:
        label = f"lite {'fp16' if scheme is None else scheme}"
        print(f"\n  {model_name} — {label} (TP={tp})")
        try:
            lite_res, lite_mem, model_mem, kv_tokens, lite_texts = _measure_lite(
                model_dir, quantization=scheme, tensor_parallel_size=tp
            )
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        match_rate = None
        if hf_texts is not None:
            match_rate = _token_match_rate(lite_texts, hf_texts)

        results.append(QuantBenchResult(
            config_label=label,
            model=model_name,
            speed=lite_res,
            peak_mem_gb=lite_mem,
            model_mem_gb=model_mem,
            kv_cache_tokens=kv_tokens,
            token_match_rate=match_rate,
        ))
        print(
            f"  Model: {model_mem:.2f} GB | KV: {kv_tokens} tok | TPS: {lite_res.tps:.1f}" +
            (f" | Match: {match_rate*100:.1f}%" if match_rate is not None else "")
        )

    return results


def render_markdown_table(results: list[QuantBenchResult]) -> str:
    """Render results as a markdown table."""
    lines = [
        "| Model | Config | Model Mem | KV Tokens | TTFT (ms) | TPOT (ms) | TPS |",
        "|-------|--------|-----------|-----------|-----------|-----------|-----|",
    ]
    for r in results:
        model_mem = f"{r.model_mem_gb:.2f} GB" if r.model_mem_gb > 0 else r.peak_mem_gb and f"{r.peak_mem_gb:.2f} GB" or "N/A"
        kv = f"{r.kv_cache_tokens:,}" if r.kv_cache_tokens > 0 else "N/A"
        lines.append(
            f"| {r.model} | {r.config_label} | {model_mem} | "
            f"{kv} | {r.speed.ttft_ms:.1f} | {r.speed.tpot_ms:.2f} | "
            f"{r.speed.tps:.1f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Quantization benchmark")
    parser.add_argument("--model-dir", type=str, help="Single model to benchmark")
    parser.add_argument("--schemes", nargs="*", default=None,
                        help="Quantization schemes (None=fp16, int8, fp8, int4, smoothquant)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--json", type=str, help="Output JSON path (default: docs/benchmark_logs/)")
    parser.add_argument("--all", action="store_true", help="Run representative subset")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HF baseline")
    args = parser.parse_args()

    all_results: list[QuantBenchResult] = []

    if args.all:
        # Representative subset from the plan
        configs = [
            ("/data/shared/llm_weights/Qwen3-0.6B", [None, "int8", "fp8", "int4"], 1, False),
            ("/data/shared/llm_weights/Qwen3-0.6B-FP8", [None], 1, False),
            ("/data/shared/llm_weights/Qwen3-VL-4B-Instruct", [None, "int8"], 1, False),
            ("/data/shared/llm_weights/Qwen3-30B-A3B-Instruct-2507-FP8", [None], 2, True),
        ]
        for model_dir, schemes, tp, skip_hf in configs:
            if not Path(model_dir).exists():
                print(f"SKIP (not found): {model_dir}")
                continue
            all_results.extend(benchmark_model(model_dir, schemes, tp, skip_hf))
    elif args.model_dir:
        schemes = args.schemes if args.schemes else [None, "int8", "fp8"]
        # Convert "None" string to actual None
        schemes = [None if s in ("None", "none", "fp16") else s for s in schemes]
        all_results = benchmark_model(args.model_dir, schemes, args.tp, args.skip_hf)
    else:
        parser.print_help()
        return

    # Print summary
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    print(render_markdown_table(all_results))

    if args.json:
        out = [
            {
                "model": r.model,
                "config": r.config_label,
                "peak_mem_gb": r.peak_mem_gb,
                "model_mem_gb": r.model_mem_gb,
                "kv_cache_tokens": r.kv_cache_tokens,
                "token_match_rate": r.token_match_rate,
                **r.speed.as_dict(),
            }
            for r in all_results
        ]
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nJSON saved to {args.json}")


if __name__ == "__main__":
    main()
