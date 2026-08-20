"""精度金标准:记录/比对 greedy 逐 token 输出,保证优化不改变数值语义。

用法:
    .venv/bin/python scripts/golden_tokens.py --save /tmp/golden.json   # 优化前
    .venv/bin/python scripts/golden_tokens.py --check /tmp/golden.json  # 优化后
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

from lite_llama import SamplingParams, TextGenerator  # noqa: E402

CASES = [
    # (name, prompts, max_gen_len) — 覆盖单条/等长 batch/混合长度 batch
    ("single", ["The future of artificial intelligence is"], 48),
    ("batch_uniform", ["How to learn python?", "How to learn c++?"], 32),
    ("batch_mixed", ["Hi", "The history of the Roman Empire spans many centuries, and"], 32),
    ("batch8", [
        "I believe the meaning of life is",
        "VGG is a very important cnn backbone,",
        "Can you introduce the American Civil War.",
        "who is the first president of the United States?",
        "How to learn c++, give me some code example.",
        "How to learn python, give me some code examples.",
        "How to learn llm, please introduce transformer",
        "How to learn cnn, please introduce resnet",
    ], 32),
]


def collect(model_dir: str, use_cuda_graph: bool) -> dict:
    gen = TextGenerator(
        checkpoints_dir=model_dir, max_seq_len=2048,
        max_gpu_num_blocks=40960, use_cuda_graph=use_cuda_graph,
    )
    out = {}
    for name, prompts, n in CASES:
        params = SamplingParams(temperature=0.0, max_gen_len=n, repetition_penalty=1.0)
        out[name] = gen.generate(prompts, params)
        params_rp = SamplingParams(temperature=0.0, max_gen_len=n, repetition_penalty=1.1)
        out[name + "_rp"] = gen.generate(prompts, params_rp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="my_weight/Qwen2.5-0.5B")
    ap.add_argument("--save")
    ap.add_argument("--check")
    ap.add_argument("--cuda-graph", action="store_true")
    args = ap.parse_args()

    got = collect(args.model_dir, args.cuda_graph)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(got, f, ensure_ascii=False, indent=1)
        print(f"saved {len(got)} cases -> {args.save}")
        return 0

    with open(args.check) as f:
        want = json.load(f)
    bad = 0
    for name in want:
        if got.get(name) != want[name]:
            bad += 1
            print(f"MISMATCH {name}")
            for i, (a, b) in enumerate(zip(want[name], got.get(name, []))):
                if a != b:
                    print(f"   seq{i} want: {a[:100]!r}")
                    print(f"   seq{i} got : {b[:100]!r}")
    print(f"{'FAIL' if bad else 'PASS'}: {len(want)-bad}/{len(want)} cases identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
