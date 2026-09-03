"""Accuracy evaluation on HotpotQA / HellaSwag with lite_llama.

:class:`EvaluatorAccuracy` drives one dataset end to end — load data, generate
completions, score exact match / F1 (or MCQ accuracy) — so a checkpoint swap
yields one comparable accuracy number. GSM8K has its own harness under
``tests/evals``; this is the reading-comprehension side.

Usage:
    python examples/eval_accuracy.py \
        --dataset /path_to/hotpot_dev_distractor_v1.json \
        --model /path_to/Llama-3.2-3B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# ``evaluator`` is a sibling package of this file, not an installed one.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from lite_llama.engine import SamplingParams, TextGenerator

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

#: Filename markers naming a supported dataset; the raw files carry no format field.
_DATASET_MARKERS = ("hotpot", "hellaswag")


def _dataset_class(marker: str):
    """The dataset class for ``marker``, imported on demand.

    ``examples/evaluator/datasets.py`` pulls in ``sentence_transformers``, which is not
    one of lite_llama's own requirements — importing it at module scope would make even
    ``--help`` fail on a stock install.
    """
    from evaluator.datasets import HellaSwag, HotpotQA

    return {"hotpot": HotpotQA, "hellaswag": HellaSwag}[marker]


class EvaluatorAccuracy:
    """Runs a benchmark dataset through lite_llama and scores the completions."""

    def __init__(
        self,
        test_data_path: str,
        custom_checkpoints_dir: str,
        data_batch: int = 10,
        max_seq_len: int = 2048,
        max_gen_len: int = 1900,
    ):
        self.custom_checkpoints_dir = custom_checkpoints_dir
        self.test_data_path = test_data_path
        self.data_batch = data_batch
        # Resolved before the generator is built, so an unrecognised file fails
        # before any weight is loaded.
        self.marker = self._dataset_marker()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = TextGenerator(
            checkpoints_dir=custom_checkpoints_dir,
            max_seq_len=max_seq_len,
            device=self.device,
        )
        # Sampling, not greedy: these datasets are scored on the answer, and a
        # long budget (1900) leaves room for a reasoning chain before it.
        self.params = SamplingParams(temperature=0.7, top_p=0.8, max_gen_len=max_gen_len)

    def _dataset_marker(self) -> str:
        """Which supported dataset ``test_data_path`` holds, judged by its filename."""
        name = Path(self.test_data_path).name.lower()
        marker = next((m for m in _DATASET_MARKERS if m in name), None)
        if marker is None:
            raise ValueError(
                f"cannot tell the dataset from {name!r}; supported markers: "
                f"{list(_DATASET_MARKERS)}"
            )
        return marker

    def process_prompts(self, prompts: list[str]) -> list[str]:
        """Generate one completion per prompt, sliced into cache-friendly batches."""
        predictions: list[str] = []
        for start in range(0, len(prompts), self.data_batch):
            batch = prompts[start : start + self.data_batch]
            predictions.extend(self.generator.generate(batch, self.params))
        return predictions

    def process(self) -> None:
        data_obj = _dataset_class(self.marker)(self.test_data_path, self.data_batch)

        ground_truth, prompts, options = data_obj.parse_data()
        predictions = self.process_prompts(prompts)

        if data_obj.data_type == "mcq":
            data_obj.evaluate(predictions, ground_truth, options)
        else:
            data_obj.evaluate(predictions, ground_truth)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="HotpotQA or HellaSwag JSON file")
    ap.add_argument("--model", required=True, help="Checkpoint directory")
    ap.add_argument("--batch", type=int, default=10, help="Prompts per generate call")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-gen-len", type=int, default=1900)
    args = ap.parse_args()

    EvaluatorAccuracy(
        args.dataset,
        args.model,
        data_batch=args.batch,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
    ).process()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
