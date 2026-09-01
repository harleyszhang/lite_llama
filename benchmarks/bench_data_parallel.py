#!/usr/bin/env python
"""Data-parallel throughput scaling — and a sanity check on the outputs.

``--scaling weak`` grows the total batch with the replica count (the
serving question, where DP earns near-linear gains); ``strong`` splits a
fixed batch. Answers are diffed so speed never hides wrongness.

Usage:
    python benchmarks/bench_data_parallel.py --model <ckpt> --dp 2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import (
    PROMPTS,
    expand_prompts,
    free_gpu,
    measure_generate,
    report_agreement,
    require_gpus,
    timestamped_log_path,
    write_json_log,
)
from lite_llama import LLM, DataParallelEngine
from lite_llama.engine.dp_load_balancer import LOAD_BALANCERS


@dataclass
class DPResult:
    """One configuration's measurement. ``tps`` is the number DP is judged on."""

    label: str
    replicas: int
    batch: int
    latency_s: float
    gen_tokens: int

    @property
    def tps(self) -> float:
        return self.gen_tokens / self.latency_s if self.latency_s else 0.0

    @property
    def tps_per_gpu(self) -> float:
        return self.tps / self.replicas if self.replicas else 0.0

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "tps": round(self.tps, 1),
            "tps_per_gpu": round(self.tps_per_gpu, 1),
        }


def bench_single_process(model, prompts, gen_len, iters, **kw) -> tuple[DPResult, list[str]]:
    """Reference row: one in-process ``LLM``, no coordinator, no IPC."""
    llm = LLM(model=model, **kw)
    try:
        latency, tokens, texts = measure_generate(
            llm.generate, prompts, gen_len=gen_len, iters=iters, tokenizer=llm.tokenizer
        )
    finally:
        del llm
        free_gpu()
    return DPResult("LLM (in-process)", 1, len(prompts), latency, tokens), texts


def bench_data_parallel(
    model, replicas, prompts, gen_len, iters, load_balancer, max_num_seqs, **kw
) -> tuple[DPResult, list[str]]:
    """One row per replica count, through the DP coordinator.

    ``max_num_seqs`` is the replica's concurrency ceiling, and it has to be stated:
    a replica hosts a *resident* engine, so unlike the one-shot ``LLM`` row it cannot
    size itself to the batch it is handed. Leaving it at the serving default while the
    reference row decodes the whole batch at once would attribute a difference in
    concurrency to a difference in parallelism.
    """
    with DataParallelEngine(
        model=model,
        data_parallel_size=replicas,
        load_balancer=load_balancer,
        max_num_seqs=max_num_seqs,
        **kw,
    ) as engine:
        latency, tokens, texts = measure_generate(
            engine.generate, prompts, gen_len=gen_len, iters=iters, tokenizer=engine.tokenizer
        )
    free_gpu()
    return DPResult(
        f"DataParallelEngine dp={replicas}", replicas, len(prompts), latency, tokens
    ), texts


def print_table(results: list[DPResult], baseline: DPResult, scaling: str) -> None:
    row = "{:<28}{:>9}{:>7}{:>12}{:>12}{:>11}{:>12}"
    print(f"\n{'─' * 91}")
    print(row.format("config", "replicas", "batch", "latency (s)", "gen tokens", "TPS", "speedup"))
    print(f"{'─' * 91}")
    for r in results:
        speedup = f"{r.tps / baseline.tps:.2f}x" if baseline.tps else "—"
        print(
            row.format(
                r.label,
                r.replicas,
                r.batch,
                f"{r.latency_s:.2f}",
                r.gen_tokens,
                f"{r.tps:.1f}",
                speedup,
            )
        )
    print(f"{'─' * 91}")
    scaled = [r for r in results if r.replicas > 1]
    if not scaled or not baseline.tps:
        return
    best = max(scaled, key=lambda r: r.tps)
    efficiency = (best.tps / baseline.tps) / best.replicas * 100
    print(
        f"{scaling} scaling at dp={best.replicas}: {best.tps / baseline.tps:.2f}x "
        f"= {efficiency:.0f}% of linear "
        f"({best.tps_per_gpu:.0f} tok/s per GPU vs {baseline.tps:.0f} on one)"
    )
    if scaling == "strong" and efficiency < 70:
        # Not a defect: a batch this small leaves one GPU bandwidth-bound, so the
        # per-replica batch costs nearly the same step time as the whole batch did.
        print(
            "  the fixed batch is too small to saturate one GPU — rerun with a larger "
            "--batch-size, or with --scaling weak to measure serving throughput"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", default="my_weight/Qwen2.5-0.5B")
    parser.add_argument("--dp", type=int, default=2, help="Largest replica count (rows for 1..dp)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Prompts: total across replicas (strong) or per replica (weak)",
    )
    parser.add_argument("--scaling", default="weak", choices=["strong", "weak"])
    # Taken from the registry rather than spelled out, so a new policy is benchmarkable
    # the moment it is registered.
    parser.add_argument("--load-balancer", default="round_robin", choices=list(LOAD_BALANCERS))
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help="Replica concurrency ceiling; 0 sizes it to the prompts each replica gets",
    )
    parser.add_argument("--iters", type=int, default=2, help="Timed repeats (median reported)")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--max-gpu-num-blocks",
        type=int,
        default=None,
        help="KV cache tokens per replica; profiled when omitted",
    )
    parser.add_argument("--quantization", default=None, choices=["int8", "int4", "smoothquant"])
    parser.add_argument("--log-dir", default=None, help="Write a JSON log here")
    args = parser.parse_args()

    visible = require_gpus(1)
    if args.dp > visible:
        print(
            f"--dp {args.dp} needs {args.dp} GPUs, only {visible} visible; capping at {visible}",
            file=sys.stderr,
        )
        args.dp = visible

    kw = {
        "max_seq_len": args.max_seq_len,
        "max_gpu_num_blocks": args.max_gpu_num_blocks,
        "quantization": args.quantization,
    }

    def prompts_for(replicas: int) -> list[str]:
        total = args.batch_size * replicas if args.scaling == "weak" else args.batch_size
        return expand_prompts(PROMPTS, total)

    print(f"\n{'=' * 91}")
    print(
        f"{args.model}  |  {args.scaling} scaling  batch={args.batch_size}"
        f"{' per replica' if args.scaling == 'weak' else ' total'}  gen_len={args.gen_len}  "
        f"iters={args.iters}  lb={args.load_balancer}  "
        f"max_num_seqs={args.max_num_seqs or 'per-replica batch'}"
    )
    print(f"gpu={torch.cuda.get_device_name(0)} x {visible}  quant={args.quantization or 'fp16'}")
    print(f"{'=' * 91}")

    reference, reference_texts = bench_single_process(
        args.model, prompts_for(1), args.gen_len, args.iters, **kw
    )
    results = [reference]
    agreements: list[tuple[str, list[str]]] = []
    for replicas in range(1, args.dp + 1):
        prompts = prompts_for(replicas)
        # Sized to the share one replica receives, so every row decodes as wide a batch
        # as the reference ``LLM`` row does and the comparison is of replica counts.
        per_replica = -(-len(prompts) // replicas)
        result, texts = bench_data_parallel(
            args.model,
            replicas,
            prompts,
            args.gen_len,
            args.iters,
            args.load_balancer,
            args.max_num_seqs or per_replica,
            **kw,
        )
        results.append(result)
        if len(texts) == len(reference_texts):  # comparable only on the same workload
            agreements.append((result.label, texts))

    baseline = next(r for r in results if r.label.endswith("dp=1"))
    print_table(results, baseline, args.scaling)
    print()
    report_agreement(reference_texts, agreements)

    if args.log_dir:
        path = timestamped_log_path(
            args.log_dir, f"bench_dp_{Path(args.model).name}_b{args.batch_size}"
        )
        write_json_log(
            path,
            {
                "model": args.model,
                "gpu": torch.cuda.get_device_name(0),
                "n_gpus": visible,
                "scaling": args.scaling,
                "batch_size": args.batch_size,
                "gen_len": args.gen_len,
                "iters": args.iters,
                "load_balancer": args.load_balancer,
                "quantization": args.quantization,
            },
            [r.as_dict() for r in results],
        )


if __name__ == "__main__":
    main()
