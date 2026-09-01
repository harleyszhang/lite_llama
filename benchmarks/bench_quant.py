"""Quantisation benchmark: speed, memory and accuracy in one run.

Each scheme runs the same prompts (``_run_lite``); peak memory,
tokens/s and the token-match rate against HF fp16 land in one markdown
table so a scheme choice can cite all three.

Usage:
    python benchmarks/bench_quant.py --model-dir <ckpt> --schemes w8a8_int8
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure the benchmarks package is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import (
    PROMPTS,
    BenchResult,
    HFBackend,
    describe_footprint,
    expand_prompts,
    make_backend,
    peak_mem_gb,
    reset_peak_mem,
    write_json_log,
)

_MAX_GEN = 64
_BATCH = 4


@dataclass
class QuantRow:
    """一个方案的测量结果:速度 + 显存 + 精度。"""

    label: str
    model: str
    speed: BenchResult
    peak_mem_gb: float
    weights_gb: float = 0.0
    kv_cache_tokens: int = 0
    #: 与 HF fp16 基线逐 token 比对的一致率;跳过基线时为 None。
    token_match_rate: float | None = None

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "config": self.label,
            "peak_mem_gb": self.peak_mem_gb,
            "weights_gb": self.weights_gb,
            "kv_cache_tokens": self.kv_cache_tokens,
            "token_match_rate": self.token_match_rate,
            **self.speed.as_dict(),
        }


def _run_lite(
    model_dir: str, scheme: str | None, tp: int, kv_cache_dtype: str, image: str
) -> tuple[BenchResult, float, float, int, list[str]]:
    """一个量化方案:返回 (速度, 显存峰值 GiB, 权重 GiB, KV 容量 token, 输出文本)。"""
    reset_peak_mem()
    backend = make_backend(
        model_dir,
        tensor_parallel_size=tp,
        image_path=image,
        max_seq_len=2048,
        quantization=scheme,
        kv_cache_dtype=kv_cache_dtype,
    )
    try:
        speed = backend.measure(expand_prompts(PROMPTS, _BATCH), _MAX_GEN, greedy=True)
        weights_gb, kv_tokens = describe_footprint(backend.runner, tp)
        return speed, peak_mem_gb(), weights_gb, kv_tokens, backend.texts()
    finally:
        backend.close()


def _run_hf(model_dir: str) -> tuple[BenchResult, float, list[str], object]:
    """HF fp16 基线;tokenizer 一并返回,供逐 token 比对复用同一套分词。"""
    reset_peak_mem()
    backend = HFBackend(model_dir)
    try:
        speed = backend.measure(expand_prompts(PROMPTS, _BATCH), _MAX_GEN, greedy=True)
        return speed, peak_mem_gb(), backend.texts(), backend.tokenizer
    finally:
        backend.close()


def _token_match_rate(lite_texts: list[str], hf_texts: list[str], tokenizer) -> float:
    """两侧输出重分词后的逐 token 一致率。

    greedy 下第一个分歧点之后两条轨迹会一路散开,所以这个数字量的是"多久才分歧",
    不是数值精度——逐元素精度看 tests/golden 的 max_abs_diff。
    """
    total = match = 0
    for lite, hf in zip(lite_texts, hf_texts, strict=False):
        lite_ids = tokenizer.encode(lite, add_special_tokens=False)
        hf_ids = tokenizer.encode(hf, add_special_tokens=False)
        n = min(len(lite_ids), len(hf_ids))
        total += n
        match += sum(1 for a, b in zip(lite_ids[:n], hf_ids[:n], strict=True) if a == b)
    return match / total if total else 0.0


def benchmark_model(
    model_dir: str,
    schemes: list[str | None],
    tp: int = 1,
    skip_hf: bool = False,
    kv_cache_dtype: str = "auto",
    image: str = "examples/assets/vision_bench.jpg",
) -> list[QuantRow]:
    """一个 checkpoint 上跑完所有请求的方案(基线在前,方案在后)。"""
    model_name = Path(model_dir).name
    rows: list[QuantRow] = []

    hf_texts: list[str] | None = None
    tokenizer = None
    if not skip_hf:
        print(f"\n{'=' * 60}\n  {model_name} — HF fp16 baseline\n{'=' * 60}")
        speed, peak, hf_texts, tokenizer = _run_hf(model_dir)
        rows.append(QuantRow("HF fp16", model_name, speed, peak))
        print(f"  Peak mem: {peak:.2f} GB | TPS: {speed.tps:.1f}")

    for scheme in schemes:
        label = f"lite {scheme or 'fp16'}"
        print(f"\n  {model_name} — {label} (TP={tp})")
        try:
            speed, peak, weights_gb, kv_tokens, texts = _run_lite(
                model_dir, scheme, tp, kv_cache_dtype, image
            )
        except Exception as exc:  # 方案不支持这个 checkpoint 时跳过,不中断整轮
            print(f"  SKIP: {exc}")
            continue

        match = _token_match_rate(texts, hf_texts, tokenizer) if hf_texts else None
        rows.append(QuantRow(label, model_name, speed, peak, weights_gb, kv_tokens, match))
        print(
            f"  Weights: {weights_gb:.2f} GB | KV: {kv_tokens} tok | TPS: {speed.tps:.1f}"
            + (f" | Match: {match * 100:.1f}%" if match is not None else "")
        )

    return rows


def render_markdown_table(rows: list[QuantRow]) -> str:
    lines = [
        "| Model | Config | Weights | KV Tokens | TTFT (ms) | TPOT (ms) | TPS | Match |",
        "|-------|--------|---------|-----------|-----------|-----------|-----|-------|",
    ]
    for r in rows:
        # 基线不报权重占用(HF 侧没有 ModelRunner 可问),退回显存峰值。
        weights = f"{r.weights_gb:.2f} GB" if r.weights_gb else f"{r.peak_mem_gb:.2f} GB peak"
        kv = f"{r.kv_cache_tokens:,}" if r.kv_cache_tokens else "N/A"
        match = "—" if r.token_match_rate is None else f"{r.token_match_rate * 100:.0f}%"
        lines.append(
            f"| {r.model} | {r.label} | {weights} | {kv} | {r.speed.ttft_ms:.1f} | "
            f"{r.speed.tpot_ms:.2f} | {r.speed.tps:.1f} | {match} |"
        )
    return "\n".join(lines)


#: ``--all`` 的代表性组合:(checkpoint, 方案, TP, 是否跳过 HF 基线)。
#: 30B MoE 的 fp16 基线反量化后放不进两张卡,只测 lite 侧。
_ALL_CONFIGS = [
    ("/data/shared/llm_weights/Qwen3-0.6B", [None, "int8", "fp8", "int4"], 1, False),
    ("/data/shared/llm_weights/Qwen3-0.6B-FP8", [None], 1, False),
    ("/data/shared/llm_weights/Qwen3-VL-4B-Instruct", [None, "int8"], 1, False),
    ("/data/shared/llm_weights/Qwen3-30B-A3B-Instruct-2507-FP8", [None], 2, True),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization benchmark")
    parser.add_argument("--model-dir", type=str, help="Single model to benchmark")
    parser.add_argument(
        "--schemes",
        nargs="*",
        default=None,
        help="Quantization schemes (fp16, int8, fp8, int4, smoothquant)",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--kv-cache-dtype", default="auto", help="auto (fp16) or fp8")
    parser.add_argument(
        "--image",
        default="examples/assets/vision_bench.jpg",
        help="Image fed to vision-language checkpoints (ignored for text models)",
    )
    parser.add_argument("--json", type=str, help="Output JSON path")
    parser.add_argument("--all", action="store_true", help="Run representative subset")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HF baseline")
    args = parser.parse_args()

    rows: list[QuantRow] = []
    if args.all:
        for model_dir, schemes, tp, skip_hf in _ALL_CONFIGS:
            if not Path(model_dir).exists():
                print(f"SKIP (not found): {model_dir}")
                continue
            rows.extend(
                benchmark_model(model_dir, schemes, tp, skip_hf, args.kv_cache_dtype, args.image)
            )
    elif args.model_dir:
        # "fp16" 是"不量化"的用户拼写,内部用 None 表示。
        requested = args.schemes or ["fp16", "int8", "fp8"]
        schemes = [None if s in ("None", "none", "fp16") else s for s in requested]
        rows = benchmark_model(
            args.model_dir, schemes, args.tp, args.skip_hf, args.kv_cache_dtype, args.image
        )
    else:
        parser.print_help()
        return

    print(f"\n{'=' * 60}\n  RESULTS\n{'=' * 60}")
    print(render_markdown_table(rows))

    if args.json:
        write_json_log(args.json, vars(args), [r.as_dict() for r in rows])


if __name__ == "__main__":
    main()
