"""EP + TBO under CUDA graph, large batch: the SGLang/DeepSeek winning shape.

Usage:
    python benchmarks/bench_overlap_ep_tbo_graph.py --batches 128 256
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
TBO_ENV = "LITE_LLAMA_TBO"
TIMELINE_ENV = "LITE_LLAMA_OVERLAP_TIMELINE"

_KV_TOKENS = 131072
_MAX_SEQ_LEN = 1024


def arm(batch: int, tbo: bool, *, gen: int = 64) -> tuple[BenchResult, list[str]]:
    """One arm; the only difference between the two TBO arms is the switch."""
    os.environ[TBO_ENV] = "1" if tbo else "0"
    os.environ["LITE_LLAMA_TBO_MIN_ROWS"] = "8"
    from lite_llama.batch_overlap.two_batch_overlap import reset_tbo_policy

    reset_tbo_policy()
    backend = make_backend(
        MODEL,
        tensor_parallel_size=2,
        use_cuda_graph=True,
        cuda_graph_lazy=True,
        max_num_seqs=max(256, batch),
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


def report(label: str, result: BenchResult) -> None:
    print(
        f"{label:16s} TTFT {result.ttft_ms:6.1f} ms | TPOT {result.tpot_ms:6.2f} ms "
        f"| TPS {result.tps:7.1f} tok/s | TGS/GPU {result.tps / 2:7.1f}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=2, help="arms per batch, best run kept")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    summary: dict[str, dict] = {}
    for batch in args.batches:
        print(f"\n=== {MODEL}  TP=2 EP=2 graph  batch={batch}  gen={args.gen} ===", flush=True)
        rows: dict[str, BenchResult] = {}
        texts: dict[str, list[str]] = {}
        for tbo in (False, True):
            label = "tbo_on" if tbo else "tbo_off"
            runs = [arm(batch, tbo, gen=args.gen) for _ in range(args.repeat)]
            rows[label], texts[label] = min(runs, key=lambda r: r[0].total_s)
            report(label, rows[label])

        off, on = rows["tbo_off"], rows["tbo_on"]
        delta = off.tpot_ms - on.tpot_ms
        print(f"-> TBO cuts TPOT by {delta:.2f} ms ({delta / off.tpot_ms:.1%})", flush=True)
        report_agreement(texts["tbo_off"], [("tbo_on", texts["tbo_on"])])

        summary[str(batch)] = {
            "tpot_ms": {k: v.tpot_ms for k, v in rows.items()},
            "ttft_ms": {k: v.ttft_ms for k, v in rows.items()},
            "tps": {k: v.tps for k, v in rows.items()},
            "tgs_per_gpu": {k: round(v.tps / 2, 1) for k, v in rows.items()},
            "tpot_improvement_pct": round(delta / off.tpot_ms * 100, 2),
        }

    if args.json:
        write_json_log(
            args.json,
            vars(args),
            {
                "batches": summary,
                "note": (
                    "EP MoE + large batch + TBO, both arms under a captured "
                    "graph (EP now keeps its graphs). This is the shape "
                    "SGLang/DeepSeek two-batch overlap targets: a big "
                    "dispatch/combine all-to-all whose wire time TBO hides "
                    "behind the other micro-batch's compute, with the launch "
                    "floor removed by replay. min_rows forced low so the "
                    "interleave is captured at the bench batches."
                ),
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
