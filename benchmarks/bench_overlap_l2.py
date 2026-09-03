"""L2 two-batch overlap: decode ping-pong on/off, TP=2.

``measure`` A/B-runs the same workload with the two-batch overlap enabled and
disabled — the only difference between arms is ``LITE_LLAMA_TBO``. Both arms
run eager (the eager policy stands down under graphs), so a third
``graph_reference`` arm reports the graphed deployment shape beside them:
the eager pair answers "does the interleave pay while every kernel is
launched from Python", and the reference shows what replay-from-graph TPOT
looks like on the same load. The gap between the two is the CPU launch floor
the eager arms sit on — not a property of the overlap itself. The fourth
``tbo_graph`` arm is where the interleave finally gets a fair trial: graphs
on, TBO on, the op stream captured inside the graph, so replay carries the
compute/comm overlap at launch-floor-free speed. Prompts stay short and
generation long, making TPOT (not TTFT) the headline number.

Usage:
    python benchmarks/bench_overlap_l2.py --model-dir <ckpt> --timeline
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

TBO_ENV = "LITE_LLAMA_TBO"
TIMELINE_ENV = "LITE_LLAMA_OVERLAP_TIMELINE"


def measure(
    model_dir: str,
    prompts: list[str],
    max_gen_len: int,
    tbo: bool,
) -> tuple[BenchResult, list[str]]:
    """One eager arm: the only difference between the two arms is the TBO switch."""
    os.environ[TBO_ENV] = "1" if tbo else "0"
    from lite_llama.batch_overlap.two_batch_overlap import reset_tbo_policy

    reset_tbo_policy()  # the rank-0 process outlives the first arm; followers do not
    backend = make_backend(
        model_dir,
        tensor_parallel_size=2,
        use_cuda_graph=False,
        max_seq_len=2048,
        max_num_seqs=64,
    )
    try:
        return backend.measure(prompts, max_gen_len, greedy=True), backend.texts()
    finally:
        backend.close()


def graph_reference(model_dir: str, prompts: list[str], max_gen_len: int) -> BenchResult:
    """The graphed deployment shape: TP2 decode replaying a captured graph.

    Not an arm — a reference. Since the TP graph capture landed, eager decode
    on this shape measures the Python launch floor, and the eager TBO arms
    inherit it; this run shows the TPOT the same load gets when decode runs
    from a captured graph instead (~300 kernel launches a step collapse to
    one replay). Read the eager on/off pair against this line.
    """
    os.environ[TBO_ENV] = "0"
    from lite_llama.batch_overlap.two_batch_overlap import reset_tbo_policy

    reset_tbo_policy()
    backend = make_backend(
        model_dir,
        tensor_parallel_size=2,
        use_cuda_graph=True,
        max_seq_len=2048,
        max_num_seqs=64,
    )
    try:
        return backend.measure(prompts, max_gen_len, greedy=True)
    finally:
        backend.close()


def tbo_graph(
    model_dir: str, prompts: list[str], max_gen_len: int
) -> tuple[BenchResult, list[str]]:
    """The arm the L2 switch was building toward: the interleave captured.

    Graphs on, TBO switch on: every eligible captured batch size records
    the ping-pong op stream — halves' compute on the capture stream,
    deferred all-reduces on the comm stream, event fences between — so
    decode replays the overlap instead of scheduling it from Python. The
    launch floor that swallowed the eager TBO arm is gone; what remains of
    the on/off gap against ``graph_reference`` is the overlap's own value.
    """
    os.environ[TBO_ENV] = "1"
    from lite_llama.batch_overlap.two_batch_overlap import reset_tbo_policy

    reset_tbo_policy()
    backend = make_backend(
        model_dir,
        tensor_parallel_size=2,
        use_cuda_graph=True,
        max_seq_len=2048,
        max_num_seqs=64,
    )
    try:
        return backend.measure(prompts, max_gen_len, greedy=True), backend.texts()
    finally:
        backend.close()


def timeline_evidence(model_dir: str, prompts: list[str]) -> str:
    """Run a short round with the timeline on; summarise the ping-pong overlap."""
    os.environ[TBO_ENV] = "1"
    os.environ[TIMELINE_ENV] = "1"
    from lite_llama.batch_overlap.comm_overlap import CommStreamPool
    from lite_llama.batch_overlap.two_batch_overlap import reset_tbo_policy

    reset_tbo_policy()
    CommStreamPool.reset()
    backend = make_backend(
        model_dir,
        tensor_parallel_size=2,
        use_cuda_graph=False,
        max_seq_len=2048,
        max_num_seqs=32,
    )
    try:
        backend.measure(prompts[:8], 8, greedy=True)
        records = CommStreamPool.for_device("cuda").timeline.collect()
        comm = [r for r in records if r.stream == "comm"]
        b_half = [r for r in records if r.name.endswith(".b")]
        pairs = 0
        overlap_ms = 0.0
        for reduce in comm:
            for seg in b_half:
                span = min(reduce.end_ms, seg.end_ms) - max(reduce.start_ms, seg.start_ms)
                if span > 0:
                    pairs += 1
                    overlap_ms += span
        return (
            f"comm regions {len(comm)}, half-B segments {len(b_half)}; "
            f"{pairs} overlapping pairs totalling {overlap_ms:.2f} ms of overlap"
        )
    finally:
        backend.close()
        os.environ.pop(TIMELINE_ENV, None)


def short_prompts(batch: int) -> list[str]:
    """Short prompts, long generation: decode dominates the timing."""
    base = expand_prompts(PROMPTS, batch)
    return [prompt[:64] for prompt in base]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=CKPT)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--max-gen-len", type=int, default=64, help="long: decode dominates")
    ap.add_argument("--repeat", type=int, default=2, help="arms per batch, best run kept")
    ap.add_argument("--timeline", action="store_true", help="one extra round of overlap evidence")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    require_gpus(2)
    summary: dict[str, dict] = {}
    all_texts: dict[str, dict[str, list[str]]] = {}
    for batch in args.batches:
        prompts = short_prompts(batch)
        results: dict[str, BenchResult] = {}
        texts: dict[str, list[str]] = {}
        for tbo in (False, True):
            label = "tbo_on" if tbo else "tbo_off"
            runs = [
                measure(args.model_dir, prompts, args.max_gen_len, tbo) for _ in range(args.repeat)
            ]
            results[label], texts[label] = min(runs, key=lambda r: r[0].total_s)

        print(f"\n{args.model_dir}  TP=2  batch={batch}  gen={args.max_gen_len}")
        for label, result in results.items():
            print(result.row(label))

        off, on = results["tbo_off"], results["tbo_on"]
        delta = off.tpot_ms - on.tpot_ms
        print(f"-> TBO cuts TPOT by {delta:.2f} ms ({delta / off.tpot_ms:.1%})")
        report_agreement(texts["tbo_off"], [("tbo_on", texts["tbo_on"])])

        graph = graph_reference(args.model_dir, prompts, args.max_gen_len)
        print(graph.row("graph_reference"))

        graphed, graphed_texts = tbo_graph(args.model_dir, prompts, args.max_gen_len)
        print(graphed.row("tbo_graph"))
        graph_delta = graph.tpot_ms - graphed.tpot_ms
        print(
            f"-> captured TBO cuts graphed TPOT by {graph_delta:.2f} ms "
            f"({graph_delta / graph.tpot_ms:.1%} vs graph_reference)"
        )
        report_agreement(texts["tbo_off"], [("tbo_graph", graphed_texts)])

        summary[str(batch)] = {
            "tpot_ms": {k: v.tpot_ms for k, v in results.items()},
            "ttft_ms": {k: v.ttft_ms for k, v in results.items()},
            "total_s": {k: v.total_s for k, v in results.items()},
            "tpot_improvement_pct": round(delta / off.tpot_ms * 100, 2),
            "graph_reference_tpot_ms": graph.tpot_ms,
            "tbo_graph_tpot_ms": graphed.tpot_ms,
            "tbo_graph_improvement_pct": round(graph_delta / graph.tpot_ms * 100, 2),
        }
        all_texts[str(batch)] = texts

    evidence = ""
    if args.timeline:
        print("\n=== timeline: all-reduce vs half-B compute overlap ===")
        evidence = timeline_evidence(args.model_dir, short_prompts(8))
        print(evidence)

    if args.json:
        write_json_log(
            args.json,
            vars(args),
            {
                "batches": summary,
                "timeline": evidence,
                "note": (
                    "measured after TBO gained graph capture"
                    " (batch_overlap/two_batch_overlap.py + executor/):"
                    " eager arms still sit on the Python launch floor, so"
                    " eager TBO pays its scheduling against a floor the"
                    " graph never sees. graph_reference is the graphed"
                    " deployment shape (TBO off); tbo_graph is the same"
                    " shape with the interleave captured — replay carries"
                    " the compute/comm overlap at launch-floor-free speed,"
                    " and tbo_graph vs graph_reference is the overlap's"
                    " marginal value. Greedy divergences vs eager arms are"
                    " bf16 reduction-order noise on low-confidence rows."
                ),
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
