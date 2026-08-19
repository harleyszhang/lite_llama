"""Accuracy evaluation on HotpotQA / HellaSwag with lite_llama.

Feeds the benchmark prompts through a :class:`~lite_llama.engine.generator.TextGenerator`
in batches and hands the completions to the dataset evaluators in
``evaluator/eval.py`` (HotpotQA uses sentence-embedding similarity, HellaSwag
scores multiple-choice selections).

Run from the repository root:
    python examples/example_eval_acc.py
"""
from __future__ import annotations

import torch

from .evaluator.eval import HellaSwag, HotpotQA

from lite_llama.engine import SamplingParams, TextGenerator

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")


class EvaluatorAccuracy:
    """Runs a benchmark dataset through lite_llama and scores the completions."""

    def __init__(self, test_data_path: str, custom_checkpoints_dir: str, data_batch: int = 10):
        self.custom_checkpoints_dir = custom_checkpoints_dir
        self.test_data_path = test_data_path
        self.data_batch = data_batch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = TextGenerator(
            checkpoints_dir=custom_checkpoints_dir,
            max_seq_len=2048,
            device=self.device,
        )
        self.params = SamplingParams(temperature=0.7, top_p=0.8, max_gen_len=1900)

    def process_prompts(self, prompts: list[str]) -> list[str]:
        """Generate one completion per prompt, sliced into cache-friendly batches."""
        predictions: list[str] = []
        for start in range(0, len(prompts), self.data_batch):
            batch = prompts[start : start + self.data_batch]
            predictions.extend(self.generator.generate(batch, self.params))
        return predictions

    def process(self) -> None:
        if "hotpot" in self.test_data_path.lower():
            data_obj = HotpotQA(self.test_data_path, self.data_batch)
        elif "hellaswag" in self.test_data_path.lower():
            data_obj = HellaSwag(self.test_data_path, self.data_batch)
        else:
            raise AssertionError(
                f"dataset {self.test_data_path!r} may not be supported"
            )

        ground_truth, prompts, options = data_obj.parse_data()
        predictions = self.process_prompts(prompts)

        if data_obj.data_type == "mcq":
            data_obj.evaluate(predictions, ground_truth, options)
        else:
            data_obj.evaluate(predictions, ground_truth)


if __name__ == "__main__":
    ea = EvaluatorAccuracy(
        "/path_to/hotpot_dev_distractor_v1.json", "/path_to/Llama-3.2-3B-Instruct"
    )
    ea.process()
