"""GSM8K: grade-school word problems, scored by exact match on the final number.

``extract_answer`` pulls the final number, ``build_prompts`` renders
the few-shot template, and ``score`` compares answers — the whole
benchmark in three pure functions.

Usage:
    prompts, labels = build_prompts(train, test)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from tests.evals import dataset
from tests.evals.runner import (
    EvalResult,
    as_user_turn,
    build_llm,
    generate_completions,
    kv_cache_tokens,
    resolve_model_dir,
)

#: Sentinel for "no number in this completion". Distinct from any real label, so
#: an unparseable answer counts as wrong without ever matching by accident.
INVALID = -9999999

#: Where a completion is cut. ``Question`` catches the model rolling on into the
#: next few-shot item, which is what a base model always does; the other two are
#: the chat markers instruction-tuned checkpoints emit. Same list as vLLM's.
STOP = ("Question", "Assistant:", "<|separator|>")

_NUMBER = re.compile(r"-?\d+")


def extract_answer(text: str) -> int:
    """Last integer in ``text``, or :data:`INVALID`.

    Chain-of-thought answers state intermediate results before the final one, so
    the *last* number is the prediction. Thousands separators are stripped first
    (``1,024`` is one number, not two).
    """
    numbers = _NUMBER.findall(text.replace(",", ""))
    return int(numbers[-1]) if numbers else INVALID


def build_prompts(
    train: Sequence[dict],
    test: Sequence[dict],
    *,
    num_questions: int,
    num_shots: int,
) -> tuple[list[str], list[int]]:
    """Return few-shot prompts and their gold answers.

    The shots are the first ``num_shots`` train records and the questions the
    first ``num_questions`` test records — a fixed prefix rather than a random
    sample, so two runs of the same config score the same subset and their
    accuracies are directly comparable.

    Raises:
        ValueError: A reference answer carries no number, which means the data
            file is not GSM8K and every score computed from it would be noise.
    """
    if num_shots > len(train):
        raise ValueError(f"need {num_shots} shots but train split has {len(train)}")
    num_questions = min(num_questions, len(test))

    shots = "".join(
        f"Question: {row['question']}\nAnswer: {row['answer']}\n\n" for row in train[:num_shots]
    )

    prompts, labels = [], []
    for row in test[:num_questions]:
        prompts.append(f"{shots}Question: {row['question']}\nAnswer:")
        label = extract_answer(row["answer"])
        if label == INVALID:
            raise ValueError(f"reference answer has no number: {row['answer']!r}")
        labels.append(label)
    return prompts, labels


def score(completions: Sequence[str], labels: Sequence[int]) -> tuple[float, float]:
    """Return ``(accuracy, invalid_rate)`` over the completions.

    An unparseable completion is always wrong, even against an unparseable
    label. :func:`build_prompts` rejects such labels up front, so the guard is
    belt-and-braces — but it keeps the sentinel safe here rather than making its
    safety depend on a caller two modules away.
    """
    if len(completions) != len(labels):
        raise ValueError(f"{len(completions)} completions for {len(labels)} labels")
    if not labels:
        return 0.0, 0.0

    predictions = [extract_answer(text) for text in completions]
    correct = sum(p == y and p != INVALID for p, y in zip(predictions, labels, strict=True))
    invalid = sum(p == INVALID for p in predictions)
    return correct / len(labels), invalid / len(labels)


def evaluate_gsm8k(
    model_dir: str | Path,
    *,
    num_questions: int = 1319,
    num_shots: int = 5,
    max_gen_len: int = 256,
    batch_size: int = 16,
    max_seq_len: int = 2048,
    max_gpu_num_blocks: int | None = None,
    use_chat_template: bool = False,
    use_cuda_graph: bool | None = None,
    device: str = "cuda",
    progress: bool = True,
) -> EvalResult:
    """Load the checkpoint, run GSM8K against it and score the completions.

    Args:
        model_dir: HuggingFace checkpoint directory, absolute or relative to the
            repository root.
        num_questions: Test questions to use, capped at the split's 1319.
        num_shots: Worked examples prepended to every question.
        max_gen_len: Decode budget per question.
        batch_size: Questions decoded together.
        max_seq_len: Context bound. Must fit the few-shot prompt *and*
            ``max_gen_len``; five shots alone are around 900 tokens.
        max_gpu_num_blocks: KV-cache size in tokens. Defaults to exactly what the
            batch can use (see :func:`~tests.evals.runner.kv_cache_tokens`); the
            engine's own profiling is wrong for this workload.
        use_chat_template: Wrap each prompt as a user turn. Set it for
            instruction-tuned checkpoints, leave it off for base ones.
        use_cuda_graph: Passed through to the engine; ``None`` keeps its default.
        device: Torch device string.
        progress: Show a progress bar.
    """
    train, test = dataset.load_gsm8k()
    prompts, labels = build_prompts(train, test, num_questions=num_questions, num_shots=num_shots)

    path = resolve_model_dir(str(model_dir))
    with build_llm(
        path,
        max_seq_len=max_seq_len,
        max_gpu_num_blocks=max_gpu_num_blocks or kv_cache_tokens(batch_size, max_seq_len),
        use_cuda_graph=use_cuda_graph,
        device=device,
    ) as llm:
        if use_chat_template:
            prompts = [as_user_turn(llm, p) for p in prompts]
        run = generate_completions(
            llm,
            prompts,
            max_gen_len=max_gen_len,
            batch_size=batch_size,
            stop=STOP,
            progress=progress,
        )

    accuracy, invalid_rate = score(run.completions, labels)
    return EvalResult(
        model_dir=str(model_dir),
        num_questions=len(labels),
        num_shots=num_shots,
        max_gen_len=max_gen_len,
        batch_size=batch_size,
        accuracy=accuracy,
        invalid_rate=invalid_rate,
        latency_s=run.latency_s,
        generated_tokens=run.generated_tokens,
        extra={"benchmark": "gsm8k", "chat_template": str(use_chat_template)},
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GSM8K accuracy evaluation for lite_llama")
    ap.add_argument("--model-dir", default="my_weight/Qwen2.5-0.5B")
    ap.add_argument("--num-questions", type=int, default=1319)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--max-gen-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument(
        "--max-gpu-num-blocks",
        type=int,
        help="KV-cache size in tokens; defaults to batch_size * max_seq_len",
    )
    ap.add_argument(
        "--chat-template",
        action="store_true",
        help="wrap prompts as a user turn (instruction-tuned checkpoints)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-results", help="append the run to this JSON lines file")
    args = ap.parse_args(argv)

    result = evaluate_gsm8k(
        args.model_dir,
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        max_gen_len=args.max_gen_len,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_gpu_num_blocks=args.max_gpu_num_blocks,
        use_chat_template=args.chat_template,
        device=args.device,
    )

    print(f"\nGSM8K — {args.model_dir}")
    print(result.report())

    if args.save_results:
        path = Path(args.save_results)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
        print(f"appended to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
