"""L1 跨 stream 重叠:copy-stream 输入上传的 on/off A/B 与 timeline 佐证。

连续批处理的一步最多带三个 pass(prefill/extend/decode),pass i+1 的输入上传
与 pass i 的 forward 没有数据依赖——decode 喂的是上一步收获的 token。默认路径里
这些上传是 compute stream 上的页式 ``torch.tensor(..., device=...)``,每个都是一次
host 停顿;开重叠后它们从 pinned staging 经 copy stream 起飞,与上一个 pass 的计算
并行。

这个脚本回答两个问题:

1. 墙钟差多少?同一工作负载跑两遍,``LITE_LLAMA_OVERLAP`` 一开一关。输入上传
   本身很小,收益集中在"pass 之间 host 不再停顿",所以量的是端到端 wall time,
   不是上传带宽。
2. 机制真的发生了吗?``--timeline`` 单独跑一小轮,打印 copy/compute 两条 stream
   上的 region 表——upload.* 落在 copy 流、forward.* 落在 compute 流,且区间相交,
   这才是"重叠"的证据,而不是"开关没报错"。

用法:
    python benchmarks/bench_overlap_l1.py --model-dir /data/shared/llm_weights/Qwen3-0.6B
    python benchmarks/bench_overlap_l1.py --timeline --json out.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch
from common import PROMPTS, expand_prompts

CKPT = "my_weight/Qwen2.5-1.5B-Instruct"

#: 重叠开关由 ModelWorker 在建引擎时从环境读取;两个 arm 唯一的差别就是它。
OVERLAP_ENV = "LITE_LLAMA_OVERLAP"
TIMELINE_ENV = "LITE_LLAMA_OVERLAP_TIMELINE"


def free() -> None:
    """两个引擎不能同时留在一张卡上。"""
    gc.collect()
    torch.cuda.empty_cache()


def measure(model_dir: str, prompts: list[str], max_gen_len: int, overlap: bool) -> float:
    """跑完整个工作负载,返回墙钟秒数;两个 arm 只差 overlap 开关。"""
    os.environ[OVERLAP_ENV] = "1" if overlap else "0"
    from lite_llama.engine.continuous_engine import ContinuousBatchingEngine
    from lite_llama.engine.sampler import SamplingParams

    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir, max_seq_len=1024, max_num_seqs=16, use_cuda_graph=True
    )
    params = SamplingParams(temperature=0.0, max_gen_len=max_gen_len, repetition_penalty=1.0)
    try:
        engine.generate(prompts[:2], SamplingParams(temperature=0.0, max_gen_len=8))  # 预热
        torch.cuda.synchronize()
        started = time.perf_counter()
        engine.generate(prompts, params)
        torch.cuda.synchronize()
        return time.perf_counter() - started
    finally:
        engine.shutdown()
        del engine
        free()


def timeline_evidence(model_dir: str, prompts: list[str]) -> str:
    """开 timeline 跑一小轮,返回 copy/compute region 表(重叠成立的直接证据)。"""
    os.environ[OVERLAP_ENV] = "1"
    os.environ[TIMELINE_ENV] = "1"
    from lite_llama.engine.continuous_engine import ContinuousBatchingEngine
    from lite_llama.engine.sampler import SamplingParams

    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir, max_seq_len=1024, max_num_seqs=8, use_cuda_graph=True
    )
    try:
        engine.generate(prompts[:4], SamplingParams(temperature=0.0, max_gen_len=8))
        worker = engine._executor._worker
        return worker.timeline.summary()
    finally:
        engine.shutdown()
        del engine
        os.environ.pop(TIMELINE_ENV, None)
        free()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=CKPT)
    ap.add_argument("--batch", type=int, default=16, help="请求数;混合长度自然产生多 pass 步")
    ap.add_argument("--max-gen-len", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=3, help="每个 arm 重复次数,报最好的一次")
    ap.add_argument("--timeline", action="store_true", help="额外跑一轮 timeline 证据")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    prompts = expand_prompts(PROMPTS, args.batch)
    results: dict[str, float] = {}
    for overlap in (False, True):
        label = "overlap_on" if overlap else "overlap_off"
        runs = [measure(args.model_dir, prompts, args.max_gen_len, overlap) for _ in range(args.repeat)]
        results[label] = min(runs)
        print(f"{label:12s} best of {args.repeat}: {results[label]:7.3f}s")

    delta = results["overlap_off"] - results["overlap_on"]
    print(f"-> overlap saves {delta * 1000:.0f} ms ({delta / results['overlap_off']:.1%})")

    evidence = ""
    if args.timeline:
        print("\n=== timeline: copy 流与 compute 流的 region ===")
        evidence = timeline_evidence(args.model_dir, prompts)
        print(evidence)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {"config": vars(args), "wall_s": results, "timeline": evidence},
                handle,
                indent=2,
            )
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
