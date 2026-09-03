"""P8: CUDA graphs under data parallelism — TPOT, capture cost, memory.

Four cells, one per ``(dp, graph)`` combination. The build time includes
capture (non-lazy, the text default), so the on/off build delta is the price
of the graphs; per-GPU memory comes from nvidia-smi while the replicas are
alive, so the on/off delta is the graphs' device memory. Decode quality is
the number DP+CUDA Graph lives or dies on: TPOT on vs off at the same dp,
plus DP2-over-DP1 throughput as the no-lock-step evidence — replicas that
were accidentally synchronising could not scale.

Weak scaling: every replica decodes the same batch size, so throughput
should track the replica count.

Usage:
    python benchmarks/bench_dp_graph.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import (
    PROMPTS,
    expand_prompts,
    measure_generate,
    report_agreement,
    require_gpus,
    timestamped_log_path,
    write_json_log,
)
from lite_llama import DataParallelEngine

CKPT = "my_weight/Qwen3-0.6B"


def gpu_used_mb() -> list[int]:
    """Per-GPU used memory in MB, straight from the driver."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(x) for x in out.strip().splitlines()]


def run_arm(dp: int, graph: bool, batch: int, gen_len: int, iters: int):
    """One ``(dp, graph)`` cell: build time, TPOT, per-GPU memory, texts."""
    prompts = expand_prompts(PROMPTS, batch)
    before = gpu_used_mb()
    time.sleep(2.0)  # let the previous arm's workers fully release

    t0 = time.perf_counter()
    with DataParallelEngine(
        model=CKPT,
        data_parallel_size=dp,
        tensor_parallel_size=1,
        load_balancer="round_robin",
        max_num_seqs=batch,
        max_seq_len=1024,
        use_cuda_graph=graph,
    ) as engine:
        build_s = time.perf_counter() - t0
        median, tokens, texts = measure_generate(
            engine.generate, prompts, gen_len=gen_len, iters=iters, tokenizer=engine.tokenizer
        )
        used = gpu_used_mb()

    return {
        "build_s": round(build_s, 2),
        "tpot_ms": round(median * 1000 / gen_len, 3),
        "tps": round(tokens / median, 1),
        "gpu_used_mb": used[:dp],
        "gpu_delta_mb": [after - base for after, base in zip(used[:dp], before[:dp], strict=True)],
        "texts": texts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", default=CKPT)
    parser.add_argument("--batch-per-replica", type=int, default=16)
    parser.add_argument("--max-gen-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--max-dp", type=int, default=2)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    visible = require_gpus(1)
    args.max_dp = min(args.max_dp, visible)

    results = {}
    texts_by_cell = {}
    for dp in range(1, args.max_dp + 1):
        for graph in (False, True):
            label = f"dp{dp}_{'graph' if graph else 'eager'}"
            cell = run_arm(dp, graph, args.batch_per_replica * dp, args.max_gen_len, args.iters)
            results[label] = {k: v for k, v in cell.items() if k != "texts"}
            texts_by_cell[label] = cell["texts"]

            print(f"\n{label}  batch={args.batch_per_replica * dp}  gen={args.max_gen_len}")
            print(f"  build      {cell['build_s']:7.2f} s   (capture included when graph)")
            print(f"  TPOT       {cell['tpot_ms']:7.3f} ms")
            print(f"  throughput {cell['tps']:7.1f} tok/s")
            print(f"  gpu used   {cell['gpu_used_mb']} MB   delta {cell['gpu_delta_mb']} MB")

    print("\n=== deltas ===")
    for dp in range(1, args.max_dp + 1):
        eager, graph = results[f"dp{dp}_eager"], results[f"dp{dp}_graph"]
        tpot = (eager["tpot_ms"] - graph["tpot_ms"]) / eager["tpot_ms"]
        print(
            f"dp={dp}: graph cuts TPOT by {tpot:.1%} "
            f"({eager['tpot_ms']:.3f} -> {graph['tpot_ms']:.3f} ms), "
            f"capture adds {graph['build_s'] - eager['build_s']:.2f}s build, "
            f"+{sum(graph['gpu_delta_mb']) - sum(eager['gpu_delta_mb'])} MB"
        )
    if args.max_dp >= 2:
        eager1, graph2 = results["dp1_eager"], results["dp2_graph"]
        print(
            f"no-lock-step: dp2 graph throughput is {graph2['tps'] / eager1['tps']:.2f}x dp1 eager"
        )

    report_agreement(texts_by_cell["dp1_eager"], list(texts_by_cell.items()))

    if args.json is None:
        args.json = timestamped_log_path(
            Path(__file__).resolve().parent.parent / "docs" / "benchmark_logs", "dp_graph"
        )
    write_json_log(
        args.json,
        {
            "model_dir": args.model_dir,
            "batch_per_replica": args.batch_per_replica,
            "max_gen_len": args.max_gen_len,
            "iters": args.iters,
            "max_dp": args.max_dp,
        },
        results,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
