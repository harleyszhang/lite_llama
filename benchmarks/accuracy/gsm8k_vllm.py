"""GSM8K accuracy on vllm — the comparison side of tests/evals/gsm8k.py.

Same prompts (build_prompts), same stop markers, same scoring (score) as the
rapid_llm path; only the generation engine differs, so the two accuracies are
directly comparable. Greedy decoding, chat template for instruction-tuned
checkpoints (the counterpart of runner.as_user_turn).

Usage:
    python -m benchmarks.accuracy.gsm8k_vllm --model-dir <checkpoint> \
        [--num-questions 200] [--chat-template] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.evals.dataset import load_gsm8k
from tests.evals.gsm8k import STOP, build_prompts, score
from tests.evals.runner import truncate_at_stop


def evaluate_gsm8k_vllm(
    model_dir: str,
    *,
    num_questions: int = 200,
    num_shots: int = 5,
    max_gen_len: int = 256,
    use_chat_template: bool = False,
    max_model_len: int = 2048,
    gpu_util: float = 0.85,
) -> dict:
    from vllm import LLM, SamplingParams

    train, test = load_gsm8k()
    prompts, labels = build_prompts(train, test, num_questions=num_questions, num_shots=num_shots)

    llm = LLM(
        model=model_dir,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
    )
    if use_chat_template:
        tokenizer = llm.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

    params = SamplingParams(temperature=0.0, max_tokens=max_gen_len, stop=list(STOP))
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=True)
    latency_s = time.perf_counter() - t0

    # vllm already cuts at the stop markers; truncate_at_stop is belt-and-
    # braces so both engines' completions go through the identical path.
    completions = [truncate_at_stop(o.outputs[0].text, STOP) for o in outputs]
    generated_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    accuracy, invalid_rate = score(completions, labels)

    return {
        "engine": "vllm",
        "model_dir": model_dir,
        "num_questions": len(labels),
        "num_shots": num_shots,
        "max_gen_len": max_gen_len,
        "chat_template": str(use_chat_template),
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "latency_s": latency_s,
        "generated_tokens": generated_tokens,
        "questions_per_second": len(labels) / latency_s if latency_s else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--num-questions", type=int, default=200)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--max-gen-len", type=int, default=256)
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    result = evaluate_gsm8k_vllm(
        args.model_dir,
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        max_gen_len=args.max_gen_len,
        use_chat_template=args.chat_template,
    )
    print(
        f"vllm GSM8K: accuracy={result['accuracy']:.4f} "
        f"invalid={result['invalid_rate']:.4f} "
        f"({result['num_questions']} questions, {result['latency_s']:.1f}s)"
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
