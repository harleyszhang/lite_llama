"""SBO A/B (eager): V2-Lite EP2 with single-batch overlap on/off.

Answers one question: does moving the shared MLP onto a side stream pay while
the routed path's dispatch exchange is on the wire? Each arm re-launches the
engine, so the only difference is ``LITE_LLAMA_SBO``, and the timeline round
at the end counts the pairs that actually intersected on the device clock.

Both arms here run eager, so their TPOTs sit on the Python launch floor and the
exchange SBO hides is a small share of a CPU-bound step — the overlap is real
(the timeline proves it) but the gain does not show up in TPOT. EP now keeps
its CUDA graphs, so ``bench_overlap_sbo_graph.py`` re-runs the same two arms
launch-floor-free, where the overlap finally pays; this eager script stays as
the timeline-evidence arm (a captured graph bakes the events, so the overlap
regions can only be collected eager).

Usage:
    python benchmarks/bench_overlap_sbo.py --json docs/benchmark_logs/overlap_sbo_<ts>.json
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.lib import (
    PROMPTS,
    BenchResult,
    expand_prompts,
    make_backend,
    report_agreement,
    write_json_log,
)

MODEL = "my_weight/DeepSeek-V2-Lite"
SBO_ENV = "LITE_LLAMA_SBO"
TIMELINE_ENV = "LITE_LLAMA_OVERLAP_TIMELINE"


def arm(
    batch: int,
    sbo: bool,
    *,
    ep: bool = True,
    use_cuda_graph: bool = False,
    kv_blocks: int | None = None,
):
    """One arm; the only difference between the two SBO arms is the switch."""
    os.environ[SBO_ENV] = "1" if sbo else "0"
    from lite_llama.batch_overlap.single_batch_overlap import reset_sbo_policy

    reset_sbo_policy()
    backend = make_backend(
        MODEL,
        tensor_parallel_size=2,
        use_cuda_graph=use_cuda_graph,
        # graph capture walks the num-seqs ladder; the reference only needs to
        # cover the bench batches, so it skips the long ladder to 512
        max_num_seqs=64 if use_cuda_graph else 512,
        max_seq_len=2048,
        max_gpu_num_blocks=kv_blocks,
        enable_expert_parallel=ep,
    )
    try:
        prompts = [p[:64] for p in expand_prompts(PROMPTS, batch)]
        result = backend.measure(prompts, 64, greedy=True)
        texts = backend.texts()
    finally:
        backend.close()
    return result, texts


def timeline_evidence(batch: int, kv_blocks: int | None = None) -> str:
    """One round with the timeline on: count the shared-MLP/exchange overlaps."""
    os.environ[SBO_ENV] = "1"
    os.environ[TIMELINE_ENV] = "1"
    from lite_llama.batch_overlap.comm_overlap import CommStreamPool
    from lite_llama.batch_overlap.single_batch_overlap import reset_sbo_policy

    reset_sbo_policy()
    CommStreamPool.reset()
    backend = make_backend(
        MODEL,
        tensor_parallel_size=2,
        use_cuda_graph=False,
        max_num_seqs=512,
        max_seq_len=2048,
        max_gpu_num_blocks=kv_blocks,
        enable_expert_parallel=True,
    )
    try:
        prompts = [p[:64] for p in expand_prompts(PROMPTS, batch)]
        # A short run: the first decode steps are the cold start (NCCL channel
        # setup and Triton JIT), which is exactly what the unit test drops.
        backend.measure(prompts, 16, greedy=True)
        records = CommStreamPool.for_device("cuda").timeline.collect()
        shared = [r for r in records if r.name == "sbo.shared_mlp"]
        dispatch = [r for r in records if r.stream == "comm" and r.name.startswith("ep.dispatch")]
        pairs = 0
        overlap_ms = 0.0
        for exchange in dispatch:
            for region in shared:
                span = min(exchange.end_ms, region.end_ms) - max(exchange.start_ms, region.start_ms)
                if span > 0:
                    pairs += 1
                    overlap_ms += span
        return (
            f"dispatch regions {len(dispatch)}, shared-MLP regions {len(shared)}; "
            f"{pairs} overlapping pairs totalling {overlap_ms:.2f} ms of overlap"
        )
    finally:
        backend.close()
        os.environ.pop(TIMELINE_ENV, None)


def report(label: str, result: BenchResult, parallel: int = 2) -> None:
    """The four metrics; TGS is per-GPU throughput (TPS / parallel degree)."""
    print(
        f"{label:16s} TTFT {result.ttft_ms:6.1f} ms | TPOT {result.tpot_ms:6.2f} ms "
        f"| TPS {result.tps:7.1f} tok/s | TGS/GPU {result.tps / parallel:7.1f}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, nargs="+", default=[32, 64])
    ap.add_argument(
        "--kv-blocks",
        type=int,
        default=None,
        help="cap the KV pool in tokens instead of profiling it (shared-GPU runs)",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    summary: dict[str, dict] = {}
    for batch in args.batches:
        print(f"\n=== {MODEL}  TP=2 EP=2  batch={batch}  gen=64 ===", flush=True)
        rows: dict[str, BenchResult] = {}
        texts: dict[str, list[str]] = {}
        for sbo in (False, True):
            label = "sbo_on" if sbo else "sbo_off"
            rows[label], texts[label] = arm(batch, sbo, kv_blocks=args.kv_blocks)
            report(label, rows[label])

        off, on = rows["sbo_off"], rows["sbo_on"]
        delta = off.tpot_ms - on.tpot_ms
        print(f"-> SBO cuts TPOT by {delta:.2f} ms ({delta / off.tpot_ms:.1%})", flush=True)
        report_agreement(texts["sbo_off"], [("sbo_on", texts["sbo_on"])])

        summary[str(batch)] = {
            "tpot_ms": {k: v.tpot_ms for k, v in rows.items()},
            "ttft_ms": {k: v.ttft_ms for k, v in rows.items()},
            "tps": {k: v.tps for k, v in rows.items()},
            "tgs_per_gpu": {k: round(v.tps / 2, 1) for k, v in rows.items()},
            "tpot_improvement_pct": round(delta / off.tpot_ms * 100, 2),
        }

    print("\n=== timeline: shared MLP vs dispatch exchange overlap ===", flush=True)
    evidence = timeline_evidence(args.batches[0], kv_blocks=args.kv_blocks)
    print(evidence, flush=True)

    if args.json:
        write_json_log(
            args.json,
            vars(args),
            {
                "batches": summary,
                "timeline": evidence,
                "note": (
                    "SBO moves the shared MLP onto an alternate stream so it "
                    "computes while the routed path's dispatch exchange is on "
                    "the wire; the timeline counts the intersecting pairs. "
                    "Both arms here run eager, so their TPOTs sit on the Python "
                    "launch floor and the exchange SBO can hide is a small "
                    "share of a step — the overlap is real but the gain does "
                    "not show in TPOT. EP now keeps its graphs, so "
                    "bench_overlap_sbo_graph.py re-runs the same arms "
                    "launch-floor-free where the overlap pays. Greedy "
                    "divergences between the arms are the bf16 reduction-order "
                    "noise the EP arms already show."
                ),
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
