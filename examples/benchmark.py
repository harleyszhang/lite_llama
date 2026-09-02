"""Latency/throughput benchmark: lite_llama vs HuggingFace transformers vs vLLM.

``bench_lite_llama``, ``bench_transformers`` and ``bench_vllm`` time the same
prompts on the same device, then ``_print_report`` diffs per-token latency and
throughput — one table, not a wall of logs.

Usage:
    python examples/benchmark.py --model <ckpt> --batch-size 16
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


def bench_lite_llama(
    model_dir, prompts, gen_len, iters, device, max_gpu_num_blocks=None,
    tensor_parallel_size=1, hf_overrides=None,
) -> Metrics:
    """Measure lite_llama on one (batch_size, gen_len) configuration.

    ``tensor_parallel_size`` above 1 routes through the continuous-batching
    engine: it is the only path whose executor broadcasts each step's plan to
    follower ranks, which is what a sharded forward needs. Decode then runs
    eager (NCCL collectives cannot live inside a captured graph).
    """
    greedy = dict(temperature=0.0, top_p=1.0, repetition_penalty=1.0, stop_on_repeat=False)

    if tensor_parallel_size > 1:
        from lite_llama.engine import ContinuousBatchingEngine

        engine = ContinuousBatchingEngine.from_pretrained(
            model_dir,
            max_seq_len=2048,
            max_gpu_num_blocks=max_gpu_num_blocks,
            tensor_parallel_size=tensor_parallel_size,
            hf_overrides=hf_overrides,
        )

        def generate(params):
            return [output.text for output in engine.generate(prompts, params)]

        tokenizer = engine.tokenizer
    else:
        gen = TextGenerator(
            checkpoints_dir=model_dir, max_seq_len=2048, device=device,
            max_gpu_num_blocks=max_gpu_num_blocks, hf_overrides=hf_overrides,
        )

        def generate(params):
            return gen.generate(prompts, params)

        tokenizer = gen.tokenizer

    generate(SamplingParams(max_gen_len=8, **greedy))  # warmup

    ttfts, latencies, out_tokens = [], [], []
    texts: list[str] = []
    for _ in range(iters):
        _, ttft = _timed(lambda: generate(SamplingParams(max_gen_len=1, **greedy)))
        texts, latency = _timed(lambda: generate(SamplingParams(max_gen_len=gen_len, **greedy)))
        ttfts.append(ttft)
        latencies.append(latency)
        out_tokens.append(count_tokens(texts, tokenizer))

    prompt_tokens = count_tokens(prompts, tokenizer)
    if tensor_parallel_size > 1:
        engine.shutdown()
    else:
        del gen
    gc.collect()
    torch.cuda.empty_cache()
    return Metrics.from_runs("lite_llama", len(prompts), prompt_tokens, ttfts, latencies, out_tokens)


def bench_transformers(model_dir, prompts, gen_len, iters, device, dtype="fp16",
                      hf_overrides=None) -> Metrics:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # correct for decoder-only batched generation
    config = AutoConfig.from_pretrained(model_dir)
    # The same override the lite_llama side runs under, so a trimmed stack is
    # measured on identical arithmetic both sides — the point of layer-local
    # comparisons.
    for field, value in (hf_overrides or {}).items():
        setattr(config, field, value)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
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


def bench_vllm(model_dir, prompts, gen_len, iters, hf_overrides=None) -> Metrics:
    """Measure vLLM's offline ``LLM`` on one (batch_size, gen_len) configuration.

    Production defaults stay on (chunked prefill, CUDA graphs, torch.compile)
    except prefix caching, which lite_llama also leaves off by default — the
    warmup would otherwise serve every timed run from cache and the measured
    TTFT would describe the cache, not the engine.
    """
    from vllm import LLM as VllmLLM
    from vllm import SamplingParams as VllmParams

    llm = VllmLLM(
        model=model_dir,
        max_model_len=2048,  # the ceiling the lite_llama side runs under
        dtype="bfloat16",
        enable_prefix_caching=False,
        hf_overrides=hf_overrides or {},
    )
    tokenizer = llm.get_tokenizer()

    def generate(max_tokens):
        params = VllmParams(temperature=0.0, max_tokens=max_tokens)
        return [out.outputs[0].text for out in llm.generate(prompts, params)]

    generate(8)  # warmup: also pays vLLM's one-off compile

    ttfts, latencies, out_tokens = [], [], []
    for _ in range(iters):
        _, ttft = _timed(lambda: generate(1))
        texts, latency = _timed(lambda: generate(gen_len))
        ttfts.append(ttft)
        latencies.append(latency)
        out_tokens.append(count_tokens(texts, tokenizer))

    prompt_tokens = count_tokens(prompts, tokenizer)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return Metrics.from_runs("vllm", len(prompts), prompt_tokens, ttfts, latencies, out_tokens)


def _print_report(cfg: dict, results: list[Metrics]) -> None:
    print(f"\n{'=' * 68}\n{cfg['model']}  |  batch={cfg['batch_size']}  gen_len={cfg['gen_len']}"
          f"  iters={cfg['iters']}  gpu={cfg['gpu']}\n{'=' * 68}")
    if cfg.get("hf_overrides"):
        print(f"hf_overrides={cfg['hf_overrides']}")
    row = "{:>14}{:>12}{:>12}{:>14}{:>14}"
    print(row.format("engine", "TTFT (s)", "TPOT (ms)", "TGS (tok/s)", "out_tokens"))
    for m in results:
        print(row.format(m.engine, f"{m.ttft_s:.4f}", f"{m.tpot_ms:.3f}",
                         f"{m.tgs:.2f}", m.output_tokens))
    by_engine = {m.engine: m for m in results}
    lite, hf = by_engine.get("lite_llama"), by_engine.get("transformers")
    if hf is not None and lite is not None and hf.tpot_ms and lite.tpot_ms:
        print(f"\nspeedup vs transformers  TGS {lite.tgs / hf.tgs:.2f}x   "
              f"TPOT {hf.tpot_ms / lite.tpot_ms:.2f}x   TTFT {hf.ttft_s / lite.ttft_s:.2f}x")
    vllm = by_engine.get("vllm")
    if vllm is not None and lite is not None and vllm.tpot_ms and lite.tpot_ms:
        print(f"speedup vs vllm          TGS {lite.tgs / vllm.tgs:.2f}x   "
              f"TPOT {vllm.tpot_ms / lite.tpot_ms:.2f}x   TTFT {vllm.ttft_s / lite.ttft_s:.2f}x")


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
        "--engine", choices=["both", "lite_llama", "transformers", "vllm", "all"],
        default="both",
        help="which side to measure; use lite_llama one-sided for checkpoints whose "
             "quantisation transformers cannot load here (AWQ needs gptqmodel/autoawq); "
             "vllm runs in whichever interpreter has it installed (its own venv, not "
             "this project's), so it is typically a separate invocation",
    )
    parser.add_argument(
        "--hf-overrides", default=None,
        help="JSON applied over the checkpoint's config on every engine (vLLM "
             "--hf-overrides semantics), e.g. '{\"num_hidden_layers\": 1}' to run a "
             "trimmed stack — the layer-local comparison",
    )
    parser.add_argument(
        "--max-gpu-num-blocks", type=int, default=None,
        help="KV pool size in tokens for lite_llama; profile-based when omitted. "
             "Shrink for checkpoints near the device budget (e.g. 16384 for 8B bf16 "
             "on a 22 GiB card, where profiling leaves the pool too small for graph capture)",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="GPUs to split the lite_llama replica's weights over (e.g. 2 to fit "
             "an 8B bf16 checkpoint's b16 budget on two 22 GiB cards). Decode "
             "runs eager under TP: the sharded layers' NCCL all-reduce cannot be "
             "captured inside a CUDA graph. The transformers baseline then uses "
             "device_map=auto over the same GPUs",
    )
    parser.add_argument("--log-dir", default="docs/benchmark_logs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    prompts = (_PROMPTS * (args.batch_size // len(_PROMPTS) + 1))[: args.batch_size]
    hf_overrides = json.loads(args.hf_overrides) if args.hf_overrides else None

    results: list[Metrics] = []
    if args.engine in ("both", "all", "lite_llama"):
        results.append(bench_lite_llama(
            args.model, prompts, args.gen_len, args.iters, device,
            args.max_gpu_num_blocks, args.tensor_parallel_size, hf_overrides,
        ))
    if args.engine in ("both", "all", "transformers"):
        # TP runs spread the baseline's layers across the same GPUs (model
        # parallelism, transformers' device_map=auto), keeping both sides on
        # identical hardware.
        hf_device = "auto" if args.tensor_parallel_size > 1 else device
        results.append(bench_transformers(
            args.model, prompts, args.gen_len, args.iters, hf_device,
            args.hf_dtype, hf_overrides,
        ))
    if args.engine in ("all", "vllm"):
        results.append(bench_vllm(args.model, prompts, args.gen_len, args.iters, hf_overrides))

    cfg = dict(model=args.model, batch_size=args.batch_size, gen_len=args.gen_len,
               iters=args.iters, tensor_parallel_size=args.tensor_parallel_size,
               hf_overrides=hf_overrides, gpu=gpu,
               timestamp=datetime.now().isoformat(timespec="seconds"))
    _print_report(cfg, results)

    log_dir = Path(args.log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = Path(args.model).name
    tp_tag = f"_tp{args.tensor_parallel_size}" if args.tensor_parallel_size > 1 else ""
    log_path = log_dir / f"bench_{tag}_b{args.batch_size}_g{args.gen_len}{tp_tag}_{stamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    engines = {m.engine: asdict(m) for m in results}
    log_path.write_text(json.dumps({"config": cfg, **engines}, indent=2))
    print(f"\nsaved log -> {log_path}")


if __name__ == "__main__":
    main()
