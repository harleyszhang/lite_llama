"""Latency/throughput benchmark: lite_llama vs HuggingFace transformers.

Reports the metrics used by vLLM / SGLang serving benchmarks, per engine:

* **TTFT** (time to first token, s) — prefill latency, measured as the wall-clock
  time of a ``max_new_tokens=1`` generation over the batch (prefill + 1 token).
* **TPOT** (time per output token, ms) — steady-state decode latency, defined
  exactly as vLLM does: ``(latency - ttft) / (output_len - 1)``.
* **TGS** (token generation speed, tokens/s) — aggregate output throughput,
  ``total_output_tokens / latency``.

Methodology notes (why earlier numbers were not trustworthy):

* Both engines decode **greedily** (``temperature=0`` / ``do_sample=False``) and
  stop **naturally at EOS** — identical stopping policy, unlike the old script,
  which forced transformers to ignore EOS (``eos_token_id=None``) while lite_llama
  stopped early, so the two never ran the same workload.
* Output tokens are counted by re-tokenising the generated text with the *same*
  tokenizer for both engines (this is exactly what vLLM's ``benchmark_serving``
  does; it may inflate counts slightly but is consistent, hence fair).
* Every timed region is wrapped in ``torch.cuda.synchronize()``; a warmup pass is
  excluded; ``--iters`` runs are aggregated by median.

Results (and the full config) are printed and saved to ``--log-dir`` as JSON.

Run from the repository root:
    python examples/benchmark.py --model my_weight/Qwen2.5-1.5B-Instruct \
        --batch-size 8 --gen-len 128 --iters 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lite_llama.engine import SamplingParams, TextGenerator

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

_PROMPTS: list[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
    "A Complete Introduction to the History of the American Civil War",
    "Python is a good programming language, how to learn it?",
    "Please introduce llama model architecture and give implement cuda code.",
    "Please introduce Qwen2.5 model structure and give cuda implement code.",
]


@dataclass
class Metrics:
    """Per-engine measurement for one (batch_size, gen_len) configuration."""

    engine: str
    batch_size: int
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    tpot_ms: float
    tgs: float
    latency_s: float

    @classmethod
    def from_runs(cls, engine, batch_size, prompt_tokens, ttfts, latencies, out_tokens):
        ttft = statistics.median(ttfts)
        latency = statistics.median(latencies)
        output_tokens = round(statistics.median(out_tokens))
        avg_out = output_tokens / batch_size
        tpot_ms = (latency - ttft) / (avg_out - 1) * 1000 if avg_out > 1 else float("nan")
        tgs = output_tokens / latency if latency > 0 else float("nan")
        return cls(engine, batch_size, prompt_tokens, output_tokens, ttft, tpot_ms, tgs, latency)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_tokens(texts: list[str], tokenizer) -> int:
    """Re-tokenise generated text to count output tokens (vLLM's own method)."""
    return sum(len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts)


def _timed(fn) -> tuple[object, float]:
    _sync()
    start = time.perf_counter()
    out = fn()
    _sync()
    return out, time.perf_counter() - start


def bench_lite_llama(model_dir, prompts, gen_len, iters, device, max_gpu_num_blocks=None) -> Metrics:
    gen = TextGenerator(
        checkpoints_dir=model_dir, max_seq_len=2048, device=device,
        max_gpu_num_blocks=max_gpu_num_blocks,
    )
    greedy = dict(temperature=0.0, top_p=1.0, repetition_penalty=1.0, stop_on_repeat=False)

    gen.generate(["Hello world"] * len(prompts), SamplingParams(max_gen_len=8, **greedy))  # warmup

    ttfts, latencies, out_tokens = [], [], []
    texts: list[str] = []
    for _ in range(iters):
        _, ttft = _timed(lambda: gen.generate(prompts, SamplingParams(max_gen_len=1, **greedy)))
        texts, latency = _timed(
            lambda: gen.generate(prompts, SamplingParams(max_gen_len=gen_len, **greedy))
        )
        ttfts.append(ttft)
        latencies.append(latency)
        out_tokens.append(count_tokens(texts, gen.tokenizer))

    prompt_tokens = count_tokens(prompts, gen.tokenizer)
    del gen
    torch.cuda.empty_cache()
    return Metrics.from_runs("lite_llama", len(prompts), prompt_tokens, ttfts, latencies, out_tokens)


def bench_transformers(model_dir, prompts, gen_len, iters, device, dtype="fp16") -> Metrics:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # correct for decoder-only batched generation
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "auto": "auto"}[dtype],
        device_map=device,
    ).eval()

    def generate(max_new_tokens):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0
            )
        gen_ids = out[:, inputs.input_ids.size(-1) :]
        return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    generate(8)  # warmup

    ttfts, latencies, out_tokens = [], [], []
    for _ in range(iters):
        _, ttft = _timed(lambda: generate(1))
        texts, latency = _timed(lambda: generate(gen_len))
        ttfts.append(ttft)
        latencies.append(latency)
        out_tokens.append(count_tokens(texts, tokenizer))

    prompt_tokens = count_tokens(prompts, tokenizer)
    del model
    torch.cuda.empty_cache()
    return Metrics.from_runs("transformers", len(prompts), prompt_tokens, ttfts, latencies, out_tokens)


def _print_report(cfg: dict, lite: Metrics, hf: Metrics) -> None:
    print(f"\n{'=' * 68}\n{cfg['model']}  |  batch={cfg['batch_size']}  gen_len={cfg['gen_len']}"
          f"  iters={cfg['iters']}  gpu={cfg['gpu']}\n{'=' * 68}")
    row = "{:<14}{:>12}{:>12}{:>14}{:>14}"
    print(row.format("engine", "TTFT (s)", "TPOT (ms)", "TGS (tok/s)", "out_tokens"))
    for m in (lite, hf):
        if m is None:
            continue
        print(row.format(m.engine, f"{m.ttft_s:.4f}", f"{m.tpot_ms:.3f}",
                         f"{m.tgs:.2f}", m.output_tokens))
    if hf is not None and lite is not None and hf.tpot_ms and lite.tpot_ms:
        print(f"\nspeedup  TGS {lite.tgs / hf.tgs:.2f}x   "
              f"TPOT {hf.tpot_ms / lite.tpot_ms:.2f}x   TTFT {hf.ttft_s / lite.ttft_s:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", default="my_weight/Qwen2.5-1.5B-Instruct",
                        help="Checkpoint dir (shared by lite_llama and transformers)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=2, help="Timed repeats (median reported)")
    parser.add_argument(
        "--hf-dtype", choices=["fp16", "bf16", "auto"], default="fp16",
        help="dtype the transformers baseline loads weights in; use auto for fp8 "
             "checkpoints (on GPUs without native fp8 it dequantises to the config dtype)",
    )
    parser.add_argument(
        "--engine", choices=["both", "lite_llama", "transformers"], default="both",
        help="which side to measure; use lite_llama one-sided for checkpoints whose "
             "quantisation transformers cannot load here (AWQ needs gptqmodel/autoawq)",
    )
    parser.add_argument(
        "--max-gpu-num-blocks", type=int, default=None,
        help="KV pool size in tokens for lite_llama; profile-based when omitted. "
             "Shrink for checkpoints near the device budget (e.g. 16384 for 8B bf16 "
             "on a 22 GiB card, where profiling leaves the pool too small for graph capture)",
    )
    parser.add_argument("--log-dir", default="benchmark_logs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    prompts = (_PROMPTS * (args.batch_size // len(_PROMPTS) + 1))[: args.batch_size]

    lite = hf = None
    if args.engine in ("both", "lite_llama"):
        lite = bench_lite_llama(
            args.model, prompts, args.gen_len, args.iters, device, args.max_gpu_num_blocks
        )
    if args.engine in ("both", "transformers"):
        hf = bench_transformers(args.model, prompts, args.gen_len, args.iters, device, args.hf_dtype)

    cfg = dict(model=args.model, batch_size=args.batch_size, gen_len=args.gen_len,
               iters=args.iters, gpu=gpu, timestamp=datetime.now().isoformat(timespec="seconds"))
    _print_report(cfg, lite, hf)

    log_dir = Path(args.log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = Path(args.model).name
    log_path = log_dir / f"bench_{tag}_b{args.batch_size}_g{args.gen_len}_{stamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"config": cfg, "lite_llama": asdict(lite) if lite else None,
                    "transformers": asdict(hf) if hf else None}, indent=2)
    )
    print(f"\nsaved log -> {log_path}")


if __name__ == "__main__":
    main()
