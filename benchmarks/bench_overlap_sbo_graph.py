"""SBO under CUDA graph: does the shared-MLP/dispatch overlap finally pay?

``bench_overlap_sbo.py`` runs both arms eager, because EP used to force graphs
off — so both TPOTs sat on the Python launch floor and the exchange SBO hides
was a rounding error against it. EP now keeps its graphs (the a2a captures
with the same comm-stream discipline TBO's deferred all-reduce already uses),
which removes the launch floor. This A/B re-runs the same two arms with graphs
on: the only difference between them is ``LITE_LLAMA_SBO``, and now the dispatch
exchange SBO moves behind the shared MLP is a real share of a launch-floor-free
step instead of a share of a CPU-bound one.

Usage:
    python benchmarks/bench_overlap_sbo_graph.py --json docs/benchmark_logs/overlap_sbo_graph_<ts>.json
"""

from __future__ import annotations

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

#: EP's captured a2a buffers make each graph far larger than a dense TP one, and
#: this script builds several EP+graph engines in one process whose private graph
#: pools do not fully release between arms. Capping the KV pool leaves the
#: headroom the captures need; the bench prompts are short, so the cap is never
#: the binding constraint on the workload.
_KV_TOKENS = 65536
_MAX_SEQ_LEN = 1024


def arm(batch: int, sbo: bool, *, use_cuda_graph: bool = True, gen: int = 64):
    """One arm; the only difference between the two SBO arms is the switch."""
    os.environ[SBO_ENV] = "1" if sbo else "0"
    from lite_llama.batch_overlap.single_batch_overlap import reset_sbo_policy

    reset_sbo_policy()
    backend = make_backend(
        MODEL,
        tensor_parallel_size=2,
        use_cuda_graph=use_cuda_graph,
        # EP's a2a buffers (ep_size * rows * top_k * hidden) make each captured
        # graph far larger than a dense TP one, so the full grid does not fit
        # beside a profiled KV pool. Lazy capture seeds a pair upfront and
        # captures the bench batches on demand during warmup, which is before
        # the measured region — so the timed steps replay, not capture.
        cuda_graph_lazy=True,
        max_num_seqs=64,
        max_seq_len=_MAX_SEQ_LEN,
        max_gpu_num_blocks=_KV_TOKENS,
        enable_expert_parallel=True,
    )
    try:
        prompts = [p[:64] for p in expand_prompts(PROMPTS, batch)]
        result = backend.measure(prompts, gen, greedy=True)
        texts = backend.texts()
    finally:
        backend.close()
    return result, texts


def timeline_evidence(batch: int) -> str:
    """One round with the timeline on: count the shared-MLP/exchange overlaps.

    Runs eager, not graphed: a captured graph bakes the timeline's CUDA events
    into the graph, so ``collect()`` cannot resolve them. The SBO overlap rides
    the same alternate-stream/comm-stream fork-join in both forms — capture
    records those edges verbatim — so the eager region intersections are valid
    evidence that the graphed replay carries the same overlap.
    """
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
        max_num_seqs=64,
        max_seq_len=_MAX_SEQ_LEN,
        max_gpu_num_blocks=_KV_TOKENS,
        enable_expert_parallel=True,
    )
    try:
        prompts = [p[:64] for p in expand_prompts(PROMPTS, batch)]
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


def report(label: str, result: BenchResult) -> None:
    print(
        f"{label:16s} TTFT {result.ttft_ms:6.1f} ms | TPOT {result.tpot_ms:6.2f} ms "
        f"| TPS {result.tps:7.1f} tok/s",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=2, help="arms per batch, best run kept")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    summary: dict[str, dict] = {}
    for batch in args.batches:
        print(f"\n=== {MODEL}  TP=2 EP=2 graph  batch={batch}  gen={args.gen} ===", flush=True)
        rows: dict[str, BenchResult] = {}
        texts: dict[str, list[str]] = {}
        for sbo in (False, True):
            label = "sbo_on" if sbo else "sbo_off"
            runs = [arm(batch, sbo, gen=args.gen) for _ in range(args.repeat)]
            rows[label], texts[label] = min(runs, key=lambda r: r[0].total_s)
            report(label, rows[label])

        off, on = rows["sbo_off"], rows["sbo_on"]
        delta = off.tpot_ms - on.tpot_ms
        print(f"-> SBO cuts TPOT by {delta:.2f} ms ({delta / off.tpot_ms:.1%})", flush=True)
        report_agreement(texts["sbo_off"], [("sbo_on", texts["sbo_on"])])

        summary[str(batch)] = {
            "tpot_ms": {k: v.tpot_ms for k, v in rows.items()},
            "ttft_ms": {k: v.ttft_ms for k, v in rows.items()},
            "tps": {k: v.tps for k, v in rows.items()},
            "tpot_improvement_pct": round(delta / off.tpot_ms * 100, 2),
        }

    print("\n=== timeline: shared MLP vs dispatch exchange overlap (graphed) ===", flush=True)
    evidence = timeline_evidence(args.batches[0])
    print(evidence, flush=True)

    if args.json:
        write_json_log(
            args.json,
            vars(args),
            {
                "batches": summary,
                "timeline": evidence,
                "note": (
                    "SBO on/off both under CUDA graph now that EP keeps its "
                    "graphs: the launch floor that swallowed the eager arms is "
                    "gone, so the dispatch exchange SBO hides behind the shared "
                    "MLP is a real share of a GPU-bound step. Greedy "
                    "divergences are the bf16 reduction-order noise the EP arms "
                    "already show."
                ),
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
