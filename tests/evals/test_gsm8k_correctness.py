"""GSM8K accuracy regression, one case per config file.

The check is a floor, not an equality: greedy decoding is deterministic, so a
repeat run on the same checkpoint reproduces the score exactly, but a *kernel*
change that is numerically fine can still flip a handful of borderline
questions. ``tolerance`` absorbs that; anything larger is a real regression.

``invalid_rate`` is asserted separately because the two failure modes need
different fixes. A low accuracy with a low invalid rate means the model got the
arithmetic wrong. A high invalid rate means the harness never saw an answer —
``max_gen_len`` too small, or the prompt format broken — and the accuracy number
carries no information about the model at all.

Usage::

    pytest -s -v tests/evals/test_gsm8k_correctness.py
    pytest -s -v tests/evals/test_gsm8k_correctness.py --config-list-file=models-all.txt
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.conftest import checkpoint_problem
from tests.evals.dataset import DatasetUnavailable
from tests.evals.gsm8k import evaluate_gsm8k
from tests.evals.runner import resolve_model_dir

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow, pytest.mark.eval]


def test_gsm8k_correctness(config_filename: Path):
    config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    # Configs name their own checkpoint, so the suite-wide `weights` gate (which
    # only knows about LITE_LLAMA_TEST_MODEL_DIR) is not enough.
    model_dir = resolve_model_dir(config["model_dir"])
    problem = checkpoint_problem(model_dir)
    if problem:
        pytest.skip(f"{config_filename.stem}: {problem}")

    print(f"\nGSM8K — {config['model_dir']}")
    print(f"  questions {config['num_questions']}, {config['num_fewshot']}-shot")
    print(f"  threshold {config['accuracy_threshold']} ± {config.get('tolerance', 0.05)}")

    try:
        result = evaluate_gsm8k(
            config["model_dir"],
            num_questions=config["num_questions"],
            num_shots=config["num_fewshot"],
            max_gen_len=config.get("max_gen_len", 256),
            batch_size=config.get("batch_size", 16),
            max_seq_len=config.get("max_seq_len", 2048),
            max_gpu_num_blocks=config.get("max_gpu_num_blocks"),
            use_chat_template=config.get("chat_template", False),
        )
    except DatasetUnavailable as exc:
        pytest.skip(str(exc))

    print(result.report())

    expected = config["accuracy_threshold"]
    tolerance = config.get("tolerance", 0.05)
    assert result.accuracy >= expected - tolerance, (
        f"GSM8K accuracy dropped: {result.accuracy:.4f} < "
        f"{expected:.4f} - {tolerance:.4f} = {expected - tolerance:.4f}"
    )

    max_invalid = config.get("max_invalid_rate")
    if max_invalid is not None:
        assert result.invalid_rate <= max_invalid, (
            f"{result.invalid_rate:.4f} of completions had no parseable answer "
            f"(limit {max_invalid:.4f}); the accuracy above is not measuring the model"
        )
