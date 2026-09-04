"""Vision benchmark: rapid_llm vs HF transformers at a fixed image size.

Both backends run the same image + question prompts with the generation
length pinned, so the comparison isolates engine overhead from the
vision encoder's cost.

Usage:
    python examples/benchmark_vision.py --model <ckpt> --image <jpg>
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
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from rapid_llm.engine import SamplingParams, VisionGenerator

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

_QUESTIONS: list[str] = [
    "Describe this image in detail.",
    "What objects can you see in this picture?",
    "What is the main subject and what is it doing?",
    "What colors dominate this image?",
    "Where was this photo most likely taken?",
    "Write a short caption for this image.",
    "What time of day does this picture suggest?",
    "Is there anything unusual about this image?",
]


@dataclass
class Metrics:
    """Per-engine measurement for one serial multimodal pass."""

    engine: str
    num_requests: int
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    tpot_ms: float
    tgs: float
    latency_s: float

    @classmethod
    def from_runs(cls, engine, num_requests, prompt_tokens, ttfts, latencies, out_tokens):
        """Per-request view of a serial multimodal pass.

        ``ttfts`` / ``latencies`` are totals over the ``num_requests`` loop:
        TTFT and latency are reported per request (divided by the loop
        width), TPOT is the serial wall-clock per output token, and TGS is
        the aggregate throughput of the serial loop.
        """
        n = num_requests
        ttft = statistics.median(ttfts) / n
        latency_total = statistics.median(latencies)
        latency = latency_total / n
        output_tokens = round(statistics.median(out_tokens))
        avg_out = output_tokens / n
        tpot_ms = (latency - ttft) / (avg_out - 1) * 1000 if avg_out > 1 else float("nan")
        tgs = output_tokens / latency_total if latency_total > 0 else float("nan")
        return cls(engine, n, prompt_tokens, output_tokens, ttft, tpot_ms, tgs, latency)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_tokens(texts: list[str], tokenizer) -> int:
    return sum(len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts)


def _timed(fn) -> tuple[object, float]:
    _sync()
    start = time.perf_counter()
    out = fn()
    _sync()
    return out, time.perf_counter() - start


def load_image(path: str, size: int) -> Image.Image:
    """Square-resized RGB image — pins the vision-token budget of dynamic-
    resolution towers (Qwen3-VL) so runs are comparable."""
    img = Image.open(path).convert("RGB")
    return img.resize((size, size), Image.BICUBIC)


def build_prompts(model_dir: str, questions: list[str]) -> list[str]:
    """LLaVA wants the explicit vicuna turn + ``<image>`` marker; Qwen3-VL's
    preparer (rapid_llm) and chat template (both engines) take a plain
    question and add the vision placeholders themselves."""
    from rapid_llm.models.config import read_model_type

    if read_model_type(model_dir) == "llava":
        return [f"USER: <image>\n{q} ASSISTANT:" for q in questions]
    return list(questions)


def bench_lite(model_dir, prompts, image, gen_len, iters, device) -> Metrics:
    gen = VisionGenerator(checkpoints_dir=model_dir, max_seq_len=2048, device=device)
    greedy = dict(temperature=0.0, top_p=1.0, repetition_penalty=1.0, stop_on_repeat=False)

    gen.generate(prompts[0], [image], SamplingParams(max_gen_len=8, **greedy))  # warmup

    ttfts, latencies, out_tokens = [], [], []
    for _ in range(iters):
        _, ttft = _timed(
            lambda: [gen.generate(p, [image], SamplingParams(max_gen_len=1, **greedy)) for p in prompts]
        )
        texts, latency = _timed(
            lambda: [gen.generate(p, [image], SamplingParams(max_gen_len=gen_len, **greedy)) for p in prompts]
        )
        ttfts.append(ttft)
        latencies.append(latency)
        out_tokens.append(count_tokens(texts, gen.engine.tokenizer))

    prompt_tokens = count_tokens(prompts, gen.engine.tokenizer)
    del gen
    gc.collect()
    torch.cuda.empty_cache()
    return Metrics.from_runs("rapid_llm", len(prompts), prompt_tokens, ttfts, latencies, out_tokens)


def bench_hf(model_dir, prompts, image, gen_len, iters, device, dtype="fp16") -> Metrics:
    processor = AutoProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer
    if not getattr(tokenizer, "pad_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen3-VL needs the chat template to insert the vision placeholders
    # (rapid_llm's preparer applies the same template on its side).
    texts = prompts
    if getattr(processor, "chat_template", None) and "<image>" not in prompts[0]:
        texts = [
            processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "auto": "auto"}[dtype],
        device_map=device,
    ).eval()

    def generate(text, max_new_tokens):
        inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0
            )
        gen_ids = out[:, inputs["input_ids"].size(-1):]
        return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    generate(texts[0], 8)  # warmup

    ttfts, latencies, out_tokens = [], [], []
    for _ in range(iters):
        _, ttft = _timed(lambda: [generate(t, 1) for t in texts])
        outs, latency = _timed(lambda: [generate(t, gen_len) for t in texts])
        ttfts.append(ttft)
        latencies.append(latency)
        out_tokens.append(count_tokens([o[0] for o in outs], tokenizer))

    prompt_tokens = count_tokens(texts, tokenizer)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return Metrics.from_runs("transformers", len(prompts), prompt_tokens, ttfts, latencies, out_tokens)


def _print_report(cfg: dict, lite: Metrics, hf: Metrics) -> None:
    print(f"\n{'=' * 68}\n{cfg['model']}  |  requests={cfg['num_requests']}  gen_len={cfg['gen_len']}"
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
    parser.add_argument("--model", default="my_weight/Qwen3-VL-4B-Instruct",
                        help="Vision-language checkpoint dir (llava / qwen3_vl)")
    parser.add_argument("--image", default="examples/assets/vision_bench.jpg")
    parser.add_argument("--image-size", type=int, default=672,
                        help="Square resize before preprocessing; pins Qwen3-VL's dynamic "
                             "vision-token count (672 -> ~576 tokens, on par with LLaVA's 576)")
    parser.add_argument("--num-requests", type=int, default=8,
                        help="Serial (image, prompt) pairs per iteration")
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=2, help="Timed repeats (median reported)")
    parser.add_argument("--hf-dtype", choices=["fp16", "bf16", "auto"], default="fp16")
    parser.add_argument("--engine", choices=["both", "rapid_llm", "transformers"], default="both")
    parser.add_argument("--log-dir", default="docs/benchmark_logs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    image = load_image(args.image, args.image_size)
    prompts = build_prompts(args.model, _QUESTIONS[: args.num_requests])

    lite = hf = None
    if args.engine in ("both", "rapid_llm"):
        lite = bench_lite(args.model, prompts, image, args.gen_len, args.iters, device)
    if args.engine in ("both", "transformers"):
        hf = bench_hf(args.model, prompts, image, args.gen_len, args.iters, device, args.hf_dtype)

    cfg = dict(model=args.model, num_requests=args.num_requests, gen_len=args.gen_len,
               iters=args.iters, image_size=args.image_size, gpu=gpu,
               timestamp=datetime.now().isoformat(timespec="seconds"))
    _print_report(cfg, lite, hf)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model).name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"bench_v_{tag}_g{args.gen_len}_{stamp}.json"
    with open(log_path, "w") as f:
        json.dump({"config": cfg,
                   "rapid_llm": asdict(lite) if lite else None,
                   "transformers": asdict(hf) if hf else None}, f, indent=2)
    print(f"-> {log_path}")


if __name__ == "__main__":
    main()
