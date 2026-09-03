"""End-to-end metric baseline: TTFT / TPOT / TPS, eager vs CUDA Graph.

``run_lite`` and ``run_hf`` drive the same prompts while ``verify``
checks the graph path still answers exactly like eager — the numbers
and the guarantee in one run.

Usage:
    python benchmarks/bench_e2e.py --model-dir <ckpt> --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import (
    GREEDY_PARAMS,
    PROMPTS,
    HFBackend,
    LiteBackend,
    VLLMBackend,
    expand_prompts,
    make_backend,
    print_table,
    write_json_log,
)

CKPT = "my_weight/Qwen2.5-0.5B"


def run_hf(args, prompts: list[str]) -> None:
    """HF transformers 对照:同一批 prompt、同一指标口径。"""
    backend = HFBackend(args.model_dir, attn=args.attn)
    result = backend.measure(prompts, args.max_gen_len, greedy=args.greedy)
    print_table({f"hf-{args.attn}": result})
    print(f"sample[0]: {backend.sample_text()!r}")


def run_vllm(args, prompts: list[str]) -> None:
    """vllm 对照:同一批 prompt、同一指标口径（TTFT 为单独一轮 1-token 测量）。"""
    backend = VLLMBackend(args.model_dir, max_model_len=2048)
    result = backend.measure(prompts, args.max_gen_len, greedy=args.greedy)
    print_table({"vllm": result})
    print(f"sample[0]: {backend.sample_text()!r}")
    if args.json:
        write_json_log(args.json, vars(args), {"vllm": result.as_dict()})


def run_lite(args, prompts: list[str]) -> dict:
    """每个 mode 一行;后端由工厂按 checkpoint 选(多模态自动走视觉口径)。"""
    modes = [("eager", False), ("graph", True)]
    if args.mode != "both":
        modes = [m for m in modes if m[0] == args.mode]

    results = {}
    for label, graph in modes:
        backend = make_backend(
            args.model_dir,
            use_cuda_graph=graph,
            image_path=args.image,
            max_seq_len=2048,
            max_gpu_num_blocks=args.max_gpu_num_blocks,
        )
        results[label] = backend.measure(prompts, args.max_gen_len, args.greedy)
        backend.close()
    return results


def verify_graph_matches_eager(args) -> int:
    """短 prompt + 长生成隔离 decode:graph capture 不能改变贪心输出。"""
    from lite_llama import SamplingParams

    params = SamplingParams(max_gen_len=args.max_gen_len, **GREEDY_PARAMS)
    outputs = {}
    for label, graph in (("eager", False), ("graph", True)):
        backend = LiteBackend(
            args.model_dir,
            use_cuda_graph=graph,
            max_seq_len=2048,
            max_gpu_num_blocks=args.max_gpu_num_blocks,
        )
        outputs[label] = backend.generator.generate(["The capital of France is"], params)[0]
        backend.close()
    if outputs["eager"] == outputs["graph"]:
        print("\nverify: eager == graph greedy output")
        return 0
    print("\nERROR: CUDA Graph output diverged from eager!")
    print("  eager:", repr(outputs["eager"]))
    print("  graph:", repr(outputs["graph"]))
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gen-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--greedy", action="store_true", help="temperature=0, deterministic")
    ap.add_argument("--mode", choices=["eager", "graph", "both"], default="both")
    ap.add_argument("--backend", choices=["lite", "hf", "vllm"], default="lite")
    ap.add_argument("--model-dir", type=str, default=CKPT)
    ap.add_argument(
        "--attn",
        default="sdpa",
        help="hf backend 的 attn_implementation: sdpa | flash_attention_2 | eager",
    )
    ap.add_argument("--verify", action="store_true", help="断言 eager 与 graph 贪心输出一致")
    ap.add_argument(
        "--max-gpu-num-blocks",
        type=int,
        default=40960,
        help="KV pool size in tokens; shrink for checkpoints near the device budget",
    )
    ap.add_argument(
        "--image",
        default="examples/assets/vision_bench.jpg",
        help="Image fed to vision-language checkpoints (ignored for text models)",
    )
    args = ap.parse_args()

    prompts = expand_prompts(PROMPTS, args.batch)

    if args.backend == "hf":
        run_hf(args, prompts)
        return 0

    if args.backend == "vllm":
        run_vllm(args, prompts)
        return 0

    results = run_lite(args, prompts)
    print_table(results)
    rc = verify_graph_matches_eager(args) if args.verify else 0

    if args.json:
        write_json_log(args.json, vars(args), {k: v.as_dict() for k, v in results.items()})
    return rc


if __name__ == "__main__":
    sys.exit(main())
