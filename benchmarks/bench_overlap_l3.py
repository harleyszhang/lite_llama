"""L3 chunked all-reduce: a TP=2 prefill-heavy workload, on/off with timeline proof.

``measure`` A/B-runs the same workload with the chunked all-reduce enabled and
disabled — the only difference between arms is ``LITE_LLAMA_COMM_OVERLAP``.
Prefill is where L3 earns: chunked prompts keep every row-parallel GEMM above
the ``min_rows`` threshold, so each chunk's reduce overlaps the next chunk's
GEMM; decode steps (a handful of rows) stay on the blocking path by design.

Usage:
    python benchmarks/bench_overlap_l3.py --model-dir <ckpt> --timeline
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import (
    PROMPTS,
    BenchResult,
    expand_prompts,
    make_backend,
    report_agreement,
    require_gpus,
    write_json_log,
)

CKPT = "my_weight/Qwen2.5-1.5B-Instruct"

OVERLAP_ENV = "LITE_LLAMA_COMM_OVERLAP"
TIMELINE_ENV = "LITE_LLAMA_OVERLAP_TIMELINE"


def measure(
    model_dir: str,
    prompts: list[str],
    max_gen_len: int,
    l3: bool,
    max_num_batched_tokens: int,
) -> tuple[BenchResult, list[str]]:
    """一个 arm:唯一差异是 L3 开关。TP=2 下 decode 走 eager(graph 本就禁用)。"""
    os.environ[OVERLAP_ENV] = "1" if l3 else "0"
    from lite_llama.batch_overlap.comm_overlap import reset_comm_overlap_policy

    reset_comm_overlap_policy()  # rank0 进程跨 arm 复用;followers 每臂重建
    backend = make_backend(
        model_dir,
        tensor_parallel_size=2,
        use_cuda_graph=False,
        max_seq_len=2048,
        max_num_seqs=16,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    try:
        return backend.measure(prompts, max_gen_len, greedy=True), backend.texts()
    finally:
        backend.close()


def timeline_evidence(model_dir: str, prompts: list[str], max_num_batched_tokens: int) -> str:
    """开 timeline 跑一小轮,汇总 comm region 与 GEMM region 的重叠证据。"""
    os.environ[OVERLAP_ENV] = "1"
    os.environ[TIMELINE_ENV] = "1"
    from lite_llama.batch_overlap.comm_overlap import CommStreamPool, reset_comm_overlap_policy

    reset_comm_overlap_policy()
    CommStreamPool.reset()
    backend = make_backend(
        model_dir,
        tensor_parallel_size=2,
        use_cuda_graph=False,
        max_seq_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    try:
        backend.measure(prompts[:4], 8, greedy=True)
        records = CommStreamPool.for_device("cuda").timeline.collect()
        comm = [r for r in records if r.stream == "comm"]
        gemms = [r for r in records if r.name.startswith("l3.gemm")]
        pairs = 0
        overlap_ms = 0.0
        for reduce in comm:
            for gemm in gemms:
                span = min(reduce.end_ms, gemm.end_ms) - max(reduce.start_ms, gemm.start_ms)
                if span > 0:
                    pairs += 1
                    overlap_ms += span
        return (
            f"comm regions {len(comm)}, l3 gemm regions {len(gemms)}; "
            f"{pairs} overlapping pairs totalling {overlap_ms:.2f} ms of overlap"
        )
    finally:
        backend.close()
        os.environ.pop(TIMELINE_ENV, None)


def long_prompts(batch: int) -> list[str]:
    """拉长到几百 token 的 prompt:chunked prefill 才能把 row-parallel GEMM 顶过阈值。"""
    base = expand_prompts(PROMPTS, batch)
    return [" ".join([prompt] * (18 + 6 * (i % 5))) for i, prompt in enumerate(base)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=CKPT)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-gen-len", type=int, default=16, help="生成短,prefill 占绝对主导")
    ap.add_argument("--max-num-batched-tokens", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=2, help="每 arm 重复次数,报最好一次")
    ap.add_argument("--timeline", action="store_true", help="额外跑一轮 timeline 重叠证据")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    require_gpus(2)
    prompts = long_prompts(args.batch)
    results: dict[str, BenchResult] = {}
    texts: dict[str, list[str]] = {}
    for l3 in (False, True):
        label = "l3_on" if l3 else "l3_off"
        runs = [
            measure(args.model_dir, prompts, args.max_gen_len, l3, args.max_num_batched_tokens)
            for _ in range(args.repeat)
        ]
        results[label], texts[label] = min(runs, key=lambda r: r[0].total_s)

    print(f"\n{args.model_dir}  TP=2  batch={args.batch}  chunk={args.max_num_batched_tokens}")
    for label, result in results.items():
        print(result.row(label))

    off, on = results["l3_off"], results["l3_on"]
    for name, getter in (("TTFT", lambda r: r.ttft_ms), ("wall", lambda r: r.total_s)):
        delta = getter(off) - getter(on)
        print(f"-> L3 cuts {name} by {delta:.1f} ms ({delta / getter(off):.1%})")

    report_agreement(texts["l3_off"], [("l3_on", texts["l3_on"])])

    evidence = ""
    if args.timeline:
        print("\n=== timeline: comm 与 GEMM region 的重叠 ===")
        evidence = timeline_evidence(args.model_dir, prompts, args.max_num_batched_tokens)
        print(evidence)

    if args.json:
        write_json_log(
            args.json,
            vars(args),
            {
                "ttft_ms": {k: v.ttft_ms for k, v in results.items()},
                "tpot_ms": {k: v.tpot_ms for k, v in results.items()},
                "total_s": {k: v.total_s for k, v in results.items()},
                "timeline": evidence,
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
