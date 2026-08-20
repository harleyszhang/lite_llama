"""精度金标准:记录/比对 greedy 逐 token 输出,保证优化不改变数值语义。

用例定义来自 ``tests/golden/cases.py``——与 pytest 侧共用同一份来源,避免脚本
录制的用例和测试回放的用例漂移。本脚本是 **录制工具**;日常校验由
``tests/golden/test_token_parity.py`` 在 pytest 中自动完成。

用法::

    # 录制当前 checkpoint 的基线(测试会自动按 checkpoint 名查找)
    .venv/bin/python scripts/golden_tokens.py \\
        --save tests/golden/data/Qwen2.5-0.5B.json

    # 手动比对(可选;pytest 已覆盖)
    .venv/bin/python scripts/golden_tokens.py --check tests/golden/data/Qwen2.5-0.5B.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lite_llama import SamplingParams, TextGenerator
from tests.golden.cases import (
    CASES,
    MAX_GPU_NUM_BLOCKS,
    MAX_SEQ_LEN,
    PENALTIES,
    case_key,
)


def collect(model_dir: str, use_cuda_graph: bool) -> dict[str, list[str]]:
    """跑完所有 (用例 x repetition_penalty) 组合,返回 key -> 输出文本列表。"""
    gen = TextGenerator(
        checkpoints_dir=model_dir,
        max_seq_len=MAX_SEQ_LEN,
        max_gpu_num_blocks=MAX_GPU_NUM_BLOCKS,
        use_cuda_graph=use_cuda_graph,
    )
    out: dict[str, list[str]] = {}
    for name, prompts, max_gen_len in CASES:
        for penalty in PENALTIES:
            params = SamplingParams(
                temperature=0.0, max_gen_len=max_gen_len, repetition_penalty=penalty
            )
            out[case_key(name, penalty)] = gen.generate(prompts, params)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="my_weight/Qwen2.5-0.5B")
    ap.add_argument("--save", help="录制基线到该 JSON 路径")
    ap.add_argument("--check", help="与该 JSON 基线比对")
    ap.add_argument("--cuda-graph", action="store_true", help="用 CUDA graph 路径采集")
    args = ap.parse_args()

    if not args.save and not args.check:
        ap.error("需要 --save 或 --check 之一")

    got = collect(args.model_dir, args.cuda_graph)

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, ensure_ascii=False, indent=1) + "\n")
        print(f"saved {len(got)} cases -> {path}")
        return 0

    want = json.loads(Path(args.check).read_text())
    bad = 0
    for name in want:
        if got.get(name) != want[name]:
            bad += 1
            print(f"MISMATCH {name}")
            for i, (a, b) in enumerate(zip(want[name], got.get(name, []), strict=False)):
                if a != b:
                    print(f"   seq{i} want: {a[:100]!r}")
                    print(f"   seq{i} got : {b[:100]!r}")
    print(f"{'FAIL' if bad else 'PASS'}: {len(want) - bad}/{len(want)} cases identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
