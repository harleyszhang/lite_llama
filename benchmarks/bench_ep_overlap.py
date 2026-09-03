"""EP overlap four-arm A/B: V2-Lite TP2 with EP on/off x TBO on/off.

Answers one question: does TBO pay when the communication is an EP
all-to-all (the payload SGLang's TBO hides) instead of a tiny TP
all-reduce? Each arm re-launches the engine so the only difference is
the two switches. A ``graph_reference`` arm closes each batch: the
same TP2 shape replaying a captured graph (ep off, tbo off). The four
eager arms sit on the Python launch floor; the reference shows the
TPOT the same load gets from a graph — read the quadruple against it.
"""

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
    write_json_log,
)

MODEL = "my_weight/DeepSeek-V2-Lite"


def arm(batch: int, tbo: bool, ep: bool, use_cuda_graph: bool = False):
    os.environ["LITE_LLAMA_TBO"] = "1" if tbo else "0"
    from lite_llama.batch_overlap.two_batch_overlap import reset_tbo_policy

    reset_tbo_policy()
    backend = make_backend(
        MODEL,
        tensor_parallel_size=2,
        use_cuda_graph=use_cuda_graph,
        # graph capture walks the num-seqs ladder; the reference only needs
        # to cover the bench batches, so it skips the long ladder to 512
        max_num_seqs=64 if use_cuda_graph else 512,
        max_seq_len=2048,
        enable_expert_parallel=ep,
    )
    try:
        prompts = [p[:64] for p in expand_prompts(PROMPTS, batch)]
        result = backend.measure(prompts, 64, greedy=True)
        texts = backend.texts()
    finally:
        backend.close()
    return result, texts


def report(label: str, result: BenchResult, agree: str) -> None:
    print(
        f"{label:16s} TTFT {result.ttft_ms:6.1f} ms | TPOT {result.tpot_ms:6.2f} ms "
        f"| TPS {result.tps:7.1f} tok/s | agree {agree}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    summary: dict[str, dict] = {}
    for batch in (16, 64):
        print(f"\n=== {MODEL}  TP=2  batch={batch}  gen=64 ===", flush=True)
        baseline = None
        rows: dict[str, BenchResult] = {}
        for ep in (False, True):
            for tbo in (False, True):
                result, texts = arm(batch, tbo, ep)
                if baseline is None:
                    baseline = texts
                    agree = "--"
                else:
                    agree = (
                        f"{sum(a == b for a, b in zip(texts, baseline, strict=False))}/{len(texts)}"
                    )
                label = f"ep={'on' if ep else 'off'} tbo={'on' if tbo else 'off'}"
                rows[label] = result
                report(label, result, agree)
        graph, graph_texts = arm(batch, tbo=False, ep=False, use_cuda_graph=True)
        rows["graph_reference"] = graph
        agree = (
            f"{sum(a == b for a, b in zip(graph_texts, baseline, strict=False))}/{len(graph_texts)}"
        )
        report("graph_reference", graph, agree)
        summary[str(batch)] = {
            "tpot_ms": {label: r.tpot_ms for label, r in rows.items()},
            "ttft_ms": {label: r.ttft_ms for label, r in rows.items()},
        }

    if args.json:
        write_json_log(
            args.json,
            vars(args),
            {
                "batches": summary,
                "note": (
                    "four eager arms + a graphed TP2 reference per batch; the eager"
                    " quadruple sits on the Python launch floor the reference"
                    " escapes (TP graph capture, 3e4d3deb), so eager TBO — and"
                    " eager EP a2a — pays its scheduling cost against a floor the"
                    " graph never sees; the profitable TBO shape is"
                    " graph-captured, which the policy does not support yet"
                ),
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
