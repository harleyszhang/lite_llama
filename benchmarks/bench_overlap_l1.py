"""L1 cross-stream overlap: copy-stream input upload on/off, with timeline proof.

``measure`` A/B-runs the same workload with overlap enabled and
disabled; ``timeline_evidence`` records the timeline showing uploads
actually overlapping compute.

Usage:
    python benchmarks/bench_overlap_l1.py --model-dir <ckpt> --timeline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import (
    PROMPTS,
    expand_prompts,
    free_gpu,
    sampling_params,
    write_json_log,
)

CKPT = "my_weight/Qwen2.5-1.5B-Instruct"

#: 重叠开关由 ModelWorker 在建引擎时从环境读取;两个 arm 唯一的差别就是它。
OVERLAP_ENV = "LITE_LLAMA_OVERLAP"
TIMELINE_ENV = "LITE_LLAMA_OVERLAP_TIMELINE"


def measure(
    model_dir: str,
    prompts: list[str],
    max_gen_len: int,
    overlap: bool,
    max_num_batched_tokens: int,
) -> float:
    """跑完整个工作负载,返回墙钟秒数;两个 arm 只差 overlap 开关。"""
    os.environ[OVERLAP_ENV] = "1" if overlap else "0"
    # 开关在建引擎时从环境读取,所以引擎必须在 os.environ 落定之后才导入。
    from lite_llama.engine.continuous_engine import ContinuousBatchingEngine

    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=2048,
        max_num_seqs=16,
        max_num_batched_tokens=max_num_batched_tokens,
        use_cuda_graph=True,
    )
    params = sampling_params(max_gen_len)
    try:
        engine.generate(prompts[:2], sampling_params(8))  # 预热
        torch.cuda.synchronize()
        started = time.perf_counter()
        engine.generate(prompts, params)
        torch.cuda.synchronize()
        return time.perf_counter() - started
    finally:
        engine.shutdown()
        del engine
        free_gpu()


def timeline_evidence(model_dir: str, prompts: list[str], max_num_batched_tokens: int) -> str:
    """开 timeline 跑一小轮,返回 copy/compute region 表(重叠成立的直接证据)。"""
    os.environ[OVERLAP_ENV] = "1"
    os.environ[TIMELINE_ENV] = "1"
    from lite_llama.engine.continuous_engine import ContinuousBatchingEngine

    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
        use_cuda_graph=True,
    )
    try:
        engine.generate(prompts[:4], sampling_params(8))
        return engine.timeline_summary()
    finally:
        engine.shutdown()
        del engine
        os.environ.pop(TIMELINE_ENV, None)
        free_gpu()


def long_prompts(batch: int) -> list[str]:
    """把基础 prompt 拉长到几百 token,让 prefill 分多个 chunk、与 decode 交错。

    短 prompt 一步就 prefill 完,整轮几乎全是纯 decode 步,没有可重叠的 pass;
    长短不一的长 prompt(逐条错开 chunk 边界)才能让混合步占住工作负载的中段。
    """
    base = expand_prompts(PROMPTS, batch)
    return [" ".join([prompt] * (18 + 6 * (i % 5))) for i, prompt in enumerate(base)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=CKPT)
    ap.add_argument("--batch", type=int, default=16, help="请求数;长短不齐的长 prompt 产生混合步")
    ap.add_argument("--max-gen-len", type=int, default=64)
    ap.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=512,
        help="单步 token 预算;小于单条长 prompt 才会切 chunk,制造 prefill/decode 混合步",
    )
    ap.add_argument("--repeat", type=int, default=3, help="每个 arm 重复次数,报最好的一次")
    ap.add_argument("--timeline", action="store_true", help="额外跑一轮 timeline 证据")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    prompts = long_prompts(args.batch)
    results: dict[str, float] = {}
    for overlap in (False, True):
        label = "overlap_on" if overlap else "overlap_off"
        runs = [
            measure(args.model_dir, prompts, args.max_gen_len, overlap, args.max_num_batched_tokens)
            for _ in range(args.repeat)
        ]
        results[label] = min(runs)
        print(f"{label:12s} best of {args.repeat}: {results[label]:7.3f}s")

    delta = results["overlap_off"] - results["overlap_on"]
    print(f"-> overlap saves {delta * 1000:.0f} ms ({delta / results['overlap_off']:.1%})")

    evidence = ""
    if args.timeline:
        print("\n=== timeline: copy 流与 compute 流的 region ===")
        evidence = timeline_evidence(args.model_dir, prompts, args.max_num_batched_tokens)
        print(evidence)

    if args.json:
        write_json_log(args.json, vars(args), {"wall_s": results, "timeline": evidence})
    return 0


if __name__ == "__main__":
    sys.exit(main())
