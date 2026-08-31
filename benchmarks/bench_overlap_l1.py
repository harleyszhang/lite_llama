"""L1 跨 stream 重叠:copy-stream 输入上传的 on/off A/B 与 timeline 佐证。

连续批处理的一步最多带三个 pass(prefill/extend/decode),同一步内的 pass 槽位不相交,
pass i+1 的输入上传与 pass i 的 forward 没有数据依赖——所以引擎把读回推迟到步末
(一步一次同步),后续 pass 的准备从 pinned staging 经 copy stream 起飞,与上一个 pass
的计算并行。默认路径里这些上传是 compute stream 上的页式 ``torch.tensor(...,
device=...)``,每个都是一次 host 停顿,而且被 per-pass 的读回串行化在两段计算之间。

这个脚本回答两个问题:

1. 墙钟差多少?同一工作负载跑两遍,``LITE_LLAMA_OVERLAP`` 一开一关。负载刻意选
   长 prompt + 小 token 预算,让 prefill 被切成多个 chunk、与 decode 交错出大量
   混合 pass 的步——重叠的收益集中在这些步上。纯 decode 稳态步没有可重叠的对
   象,差值落在噪声里是预期行为。
2. 机制真的发生了吗?``--timeline`` 单独跑一小轮,打印 copy/compute 两条 stream
   上的 region 表——混合步里 upload.decode.* 与 forward.prefill 的区间相交,
   这才是"重叠"的证据,而不是"开关没报错"。

用法:
    python benchmarks/bench_overlap_l1.py --model-dir my_weight/Qwen2.5-1.5B-Instruct
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


def measure(
    model_dir: str,
    prompts: list[str],
    max_gen_len: int,
    overlap: bool,
    max_num_batched_tokens: int,
) -> float:
    """跑完整个工作负载,返回墙钟秒数;两个 arm 只差 overlap 开关。"""
    os.environ[OVERLAP_ENV] = "1" if overlap else "0"
    from lite_llama.engine.continuous_engine import ContinuousBatchingEngine
    from lite_llama.engine.sampler import SamplingParams

    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=2048,
        max_num_seqs=16,
        max_num_batched_tokens=max_num_batched_tokens,
        use_cuda_graph=True,
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


def timeline_evidence(model_dir: str, prompts: list[str], max_num_batched_tokens: int) -> str:
    """开 timeline 跑一小轮,返回 copy/compute region 表(重叠成立的直接证据)。"""
    os.environ[OVERLAP_ENV] = "1"
    os.environ[TIMELINE_ENV] = "1"
    from lite_llama.engine.continuous_engine import ContinuousBatchingEngine
    from lite_llama.engine.sampler import SamplingParams

    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=max_num_batched_tokens,
        use_cuda_graph=True,
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
