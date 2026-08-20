"""端到端指标基线:lite_llama eager / CUDA Graph 的 TTFT / TPOT / TPS 分解。

指标口径见 benchmarks/common.py(对齐 vLLM/TensorRT-LLM);
与 HF transformers 的同口径对照见 benchmarks/bench_hf_baseline.py。

用法:
    .venv/bin/python benchmarks/bench_e2e.py --greedy --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys

from common import PROMPTS, LiteBackend, expand_prompts, print_table

CKPT = "my_weight/Qwen2.5-0.5B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gen-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--greedy", action="store_true", help="temperature=0, deterministic")
    ap.add_argument("--mode", choices=["eager", "graph", "both"], default="both")
    ap.add_argument("--model-dir", type=str, default=CKPT)
    args = ap.parse_args()

    modes = [("eager", False), ("graph", True)]
    if args.mode != "both":
        modes = [m for m in modes if m[0] == args.mode]

    prompts = expand_prompts(PROMPTS, args.batch)
    results = {}
    for label, graph in modes:
        backend = LiteBackend(
            args.model_dir,
            use_cuda_graph=graph,
            max_seq_len=2048,
            max_gpu_num_blocks=40960,
        )
        results[label] = backend.measure(prompts, args.max_gen_len, args.greedy)
        backend.close()
    print_table(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "config": vars(args),
                    "results": {k: v.as_dict() for k, v in results.items()},
                },
                f,
                indent=2,
            )
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
