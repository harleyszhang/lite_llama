"""GSM8K accuracy regression, one case per config file.

Each YAML config names a checkpoint and an expected accuracy floor; a
run generates on the real model and asserts the score clears the bar —
the tier that catches quality regressions end to end.

Usage:
    pytest tests/evals/test_gsm8k_correctness.py
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
    # only knows about RAPID_LLM_TEST_MODEL_DIR) is not enough.
    model_dir = resolve_model_dir(config["model_dir"])
    problem = checkpoint_problem(model_dir)
    if problem:
        pytest.skip(f"{config_filename.stem}: {problem}")

    print(f"\nGSM8K — {model_dir}")
    print(f"  questions {config['num_questions']}, {config['num_fewshot']}-shot")
    print(f"  threshold {config['accuracy_threshold']} ± {config.get('tolerance', 0.05)}")

    try:
        # The resolved path, not the config's: the chain above may have found
        # the checkpoint somewhere the config does not name, and evaluating the
        # unresolved string would load the very directory just reported missing.
        result = evaluate_gsm8k(
            str(model_dir),
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
