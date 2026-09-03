"""TBO scaling sweep: does the interleave pay at larger batch / larger model?

The L2 bench only covered batch 8/16/32 on Qwen2.5-1.5B, which is exactly the
shape where TBO is expected to lose — decode's M is so small that halving it
leaves every GEMM on the kernel fixed-overhead floor. This sweep walks batch
and model size up to find where (if anywhere) the inequality flips: TBO's cost
is a doubled kernel count, its benefit is the all-reduce it hides, so the
benefit only wins once M is large enough that halved GEMMs stay efficient and
the reduction is a real share of the step.

Every arm reports TTFT / TPOT / TPS / TGS (per-GPU throughput), offline
inference (all prompts submitted at once, no serving queue).

Usage:
    python benchmarks/bench_tbo_scaling.py \
        --models my_weight/Qwen2.5-1.5B-Instruct my_weight/Meta-Llama-3.1-8B-Instruct \
        --batches 32 128 256 --json docs/benchmark_logs/tbo_scaling_<ts>.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.lib import (
    PROMPTS,
    BenchResult,
    environment,
    expand_prompts,
    make_backend,
)

TBO_ENV = "LITE_LLAMA_TBO"


def model_facts(model_dir: str) -> dict:
    """The architecture numbers that decide whether halving M stays efficient."""
    from lite_llama.models.config import ModelConfig

    cfg = ModelConfig.from_pretrained(model_dir, 2048)
    return {
        "model_type": cfg.model_type,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
    }


def arm(model_dir, batch, gen_len, prompt_len, tbo, graph, kv_blocks, tp):
    """One measurement; the only difference between arms is the two switches."""
    os.environ[TBO_ENV] = "1" if tbo else "0"
    from lite_llama.batch_overlap.two_batch_overlap import reset_tbo_policy

    reset_tbo_policy()
    backend = make_backend(
        model_dir,
        tensor_parallel_size=tp,
        use_cuda_graph=graph,
        max_seq_len=2048,
        max_num_seqs=max(256, batch),
        max_gpu_num_blocks=kv_blocks,
    )
    try:
        prompts = [p[:prompt_len] for p in expand_prompts(PROMPTS, batch)]
        return backend.measure(prompts, gen_len, greedy=True)
    finally:
        backend.close()


def metrics(result: BenchResult, tp: int) -> dict:
    """The four metrics, TGS being per-GPU throughput."""
    return {
        "ttft_ms": round(result.ttft_ms, 2),
        "tpot_ms": round(result.tpot_ms, 3),
        "tps": round(result.tps, 1),
        "tgs_per_gpu": round(result.tps / tp, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["my_weight/Qwen2.5-1.5B-Instruct"])
    ap.add_argument("--batches", type=int, nargs="+", default=[32, 128, 256])
    ap.add_argument("--gen-len", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--kv-blocks", type=int, default=65536)
    ap.add_argument("--arms", nargs="+", default=["eager_off", "eager_on", "graph_off", "graph_on"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    env = environment()
    print("=== environment ===")
    for k, v in env.items():
        print(f"  {k}: {v}")

    summary: dict[str, dict] = {"environment": env, "workload": {
        "batch_sizes": args.batches,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "tensor_parallel_size": args.tp,
        "kv_blocks": args.kv_blocks,
        "greedy": True,
    }, "framework_flags": {
        "LITE_LLAMA_TBO": "per arm",
        "use_cuda_graph": "per arm",
        "LITE_LLAMA_OVERLAP": "1 (default, L1 pinned-copy)",
        "LITE_LLAMA_COMM_OVERLAP": "0 (L3 off)",
        "LITE_LLAMA_SBO": "0 (SBO off)",
    }, "models": {}, "results": {}}

    arm_specs = {
        "eager_off": (False, False),
        "eager_on": (True, False),
        "graph_off": (False, True),
        "graph_on": (True, True),
    }

    for model_dir in args.models:
        name = Path(model_dir).name
        summary["models"][name] = model_facts(model_dir)
        print(f"\n=== model {name}: {summary['models'][name]} ===")
        for batch in args.batches:
            print(f"\n--- batch={batch} prompt={args.prompt_len} gen={args.gen_len} tp={args.tp} ---")
            rows = {}
            for label in args.arms:
                tbo, graph = arm_specs[label]
                try:
                    result = arm(model_dir, batch, args.gen_len, args.prompt_len,
                                 tbo, graph, args.kv_blocks, args.tp)
                except Exception as exc:
                    print(f"  {label:12s} FAILED: {type(exc).__name__}: {exc}")
                    rows[label] = {"error": f"{type(exc).__name__}: {exc}"}
                    continue
                rows[label] = metrics(result, args.tp)
                print(f"  {label:12s} TTFT {result.ttft_ms:7.1f} ms | TPOT {result.tpot_ms:7.3f} ms "
                      f"| TPS {result.tps:8.1f} | TGS/GPU {result.tps / args.tp:8.1f}")
            base = rows.get("graph_off") or rows.get("eager_off")
            for label in ("graph_on", "eager_on"):
                cur, ref = rows.get(label), base
                if cur and ref and "error" not in cur and "error" not in ref:
                    ref_key = "graph_off" if label == "graph_on" else "eager_off"
                    r = rows.get(ref_key)
                    if r and "error" not in r:
                        d = (r["tpot_ms"] - cur["tpot_ms"]) / r["tpot_ms"] * 100
                        print(f"  -> {label} vs {ref_key}: TPOT {d:+.1f}%")
            summary["results"][f"{name}_b{batch}"] = rows

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "results": summary}, fh, indent=2, ensure_ascii=False)
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
