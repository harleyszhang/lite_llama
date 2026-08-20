"""HF transformers 基线:与 bench_e2e.py 同 prompts、同指标口径的对照组。

口径细节(benchmarks/common.py 有完整定义):
    - 左 padding、裸 completion,不套 chat template(lite_llama 同样不套)。
    - 默认 greedy:min_new_tokens == max_gen_len,禁止提前 EOS,
      batch 恰好跑满 max_gen_len 步,与 lite_llama lockstep 对齐。
    - --sample:do_sample=True(temperature=0.7, top_p=0.8),允许提前 EOS,
      gen_tokens 按非 pad token 实数统计,此时主要看 TPS。

用法:
    .venv/bin/python benchmarks/bench_hf_baseline.py --model-dir my_weight/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import sys

from common import PROMPTS, HFBackend, expand_prompts, print_table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--max-gen-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--attn",
        type=str,
        default="sdpa",
        help="attn_implementation: sdpa | flash_attention_2 | eager",
    )
    ap.add_argument(
        "--sample",
        action="store_true",
        help="do_sample=True(temperature=0.7, top_p=0.8);默认 greedy",
    )
    args = ap.parse_args()

    backend = HFBackend(args.model_dir, attn=args.attn)
    prompts = expand_prompts(PROMPTS, args.batch)
    result = backend.measure(prompts, args.max_gen_len, greedy=not args.sample)
    print_table({f"hf-{args.attn}": result})

    # Sanity: 打印一条解码样例,肉眼确认输出可读。
    print(f"sample[0]: {backend.sample_text()!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
