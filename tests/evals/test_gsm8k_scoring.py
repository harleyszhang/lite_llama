"""CPU-tier tests for the parts of the eval harness that decide the score.

The GPU tier can only tell you a number came out; it cannot tell you the number
means anything. Everything asserted here is what stands between "the model
solved it" and "the parser happened to agree":

* answer extraction — a scorer that read the wrong number would rank a broken
  engine as accurate, and no accuracy test would catch it;
* stop truncation — without it a base model rolls into the next few-shot
  question and the "last number" is that question's answer, not the model's;
* prompt construction — a fixed prefix of the split, so two runs of one config
  score the same questions and their accuracies are comparable;
* the config wiring — a typo in a list file must fail loudly rather than
  evaluate nothing and pass.

No network, no GPU, no checkpoint: the data is synthetic and the engine is never
built.
"""

from __future__ import annotations

import json

import pytest
import yaml

from tests.evals import dataset
from tests.evals.conftest import CONFIG_DIR, read_config_list
from tests.evals.gsm8k import INVALID, STOP, build_prompts, extract_answer, score
from tests.evals.runner import truncate_at_stop

# Two-record stand-ins for the real splits, in GSM8K's on-disk shape.
TRAIN = [
    {"question": "2 apples plus 3?", "answer": "2 + 3 = 5\n#### 5"},
    {"question": "4 pears plus 4?", "answer": "4 + 4 = 8\n#### 8"},
]
TEST = [
    {"question": "6 plus 6?", "answer": "6 + 6 = 12\n#### 12"},
    {"question": "7 plus 7?", "answer": "7 + 7 = 14\n#### 14"},
]


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("#### 18", 18),
        ("He has 5 left, so the answer is 42.", 42),  # last number, not the first
        ("The total is 1,024 dollars", 1024),  # thousands separator
        ("It costs $72", 72),
        ("The temperature fell to -8", -8),
        ("no digits at all", INVALID),
        ("", INVALID),
    ],
)
def test_extract_answer(text: str, expected: int):
    assert extract_answer(text) == expected


def test_extract_answer_takes_the_last_number_of_a_reasoning_chain():
    """Chain-of-thought states intermediates first; only the final one is the answer."""
    chain = "She had 16 eggs, ate 3, sold 9, so she earned 9 * 2 = 18 dollars.\n#### 18"
    assert extract_answer(chain) == 18


# --------------------------------------------------------------------------- #
# Stop-string truncation
# --------------------------------------------------------------------------- #
def test_truncate_drops_the_next_few_shot_item():
    """The bug this prevents: scoring the *next* question's answer as the model's."""
    raw = " 6 + 6 = 12\n#### 12\n\nQuestion: 9 plus 9?\nAnswer: 9 + 9 = 18\n#### 18"
    assert extract_answer(raw) == 18
    assert extract_answer(truncate_at_stop(raw, STOP)) == 12


def test_truncate_cuts_at_the_earliest_marker():
    text = "answer 12 <|separator|> tail Question: more"
    assert truncate_at_stop(text, STOP) == "answer 12 "


def test_truncate_is_identity_without_a_marker():
    assert truncate_at_stop("plain 12", STOP) == "plain 12"
    assert truncate_at_stop("plain 12", ()) == "plain 12"


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def test_build_prompts_shape_and_labels():
    prompts, labels = build_prompts(TRAIN, TEST, num_questions=2, num_shots=2)

    assert labels == [12, 14]
    assert len(prompts) == 2
    # Two worked examples plus the question under test, and nothing after the
    # trailing "Answer:" — the model must complete it, not read it.
    assert prompts[0].count("Question:") == 3
    assert prompts[0].endswith("Question: 6 plus 6?\nAnswer:")
    assert TRAIN[0]["answer"] in prompts[0]


def test_build_prompts_is_a_fixed_prefix():
    """Same config, same questions: accuracies across runs must be comparable."""
    first, _ = build_prompts(TRAIN, TEST, num_questions=1, num_shots=1)
    second, _ = build_prompts(TRAIN, TEST, num_questions=1, num_shots=1)
    assert first == second


def test_build_prompts_caps_at_the_split_size():
    prompts, labels = build_prompts(TRAIN, TEST, num_questions=999, num_shots=1)
    assert len(prompts) == len(labels) == len(TEST)


def test_build_prompts_rejects_more_shots_than_data():
    with pytest.raises(ValueError, match="train split has 2"):
        build_prompts(TRAIN, TEST, num_questions=1, num_shots=5)


def test_build_prompts_rejects_unnumbered_reference():
    """A data file that is not GSM8K must fail, not silently score as all-wrong."""
    with pytest.raises(ValueError, match="no number"):
        build_prompts(
            TRAIN, [{"question": "q", "answer": "no number here"}], num_questions=1, num_shots=1
        )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_score_counts_matches_and_unparseable_separately():
    accuracy, invalid = score(["#### 12", "#### 99", "I don't know"], [12, 14, 16])
    assert accuracy == pytest.approx(1 / 3)
    assert invalid == pytest.approx(1 / 3)


def test_score_never_credits_an_unparseable_completion():
    """INVALID must not collide with a label, or garbage would score as correct."""
    accuracy, invalid = score(["no digits"], [INVALID])
    assert accuracy == 0.0
    assert invalid == 1.0


def test_score_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="1 completions for 2 labels"):
        score(["#### 12"], [12, 14])


def test_score_of_nothing_is_zero():
    assert score([], []) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# Dataset plumbing
# --------------------------------------------------------------------------- #
def test_read_jsonl_skips_comments(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('# header\n{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert dataset.read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_cache_dir_honours_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv("LITE_LLAMA_EVAL_DATA_DIR", str(tmp_path))
    assert dataset.cache_dir() == tmp_path


def test_fetch_returns_the_cached_file_without_a_download(tmp_path, monkeypatch):
    """A pre-seeded cache is what makes the harness usable on an offline box."""
    monkeypatch.setenv("LITE_LLAMA_EVAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LITE_LLAMA_EVAL_BASE_URL", "http://0.0.0.0:1/unreachable")
    seeded = tmp_path / "gsm8k" / "test.jsonl"
    seeded.parent.mkdir(parents=True)
    seeded.write_text(json.dumps({"question": "q", "answer": "#### 1"}) + "\n")

    assert dataset.fetch("test.jsonl", "gsm8k") == seeded


def test_fetch_reports_an_unreachable_source(tmp_path, monkeypatch):
    monkeypatch.setenv("LITE_LLAMA_EVAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LITE_LLAMA_EVAL_BASE_URL", "http://0.0.0.0:1/unreachable")
    with pytest.raises(dataset.DatasetUnavailable, match="cannot obtain"):
        dataset.fetch("test.jsonl", "gsm8k")


# --------------------------------------------------------------------------- #
# Config wiring
# --------------------------------------------------------------------------- #
def test_every_shipped_config_is_complete():
    """A config missing a key would only fail hours into a GPU run."""
    required = {"model_dir", "num_questions", "num_fewshot", "accuracy_threshold"}
    configs = sorted(CONFIG_DIR.glob("*.yaml"))
    assert configs, f"no configs found in {CONFIG_DIR}"

    for config_file in configs:
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        missing = required - set(config)
        assert not missing, f"{config_file.name} is missing {sorted(missing)}"
        assert config["num_fewshot"] >= 0
        assert 0.0 <= config["accuracy_threshold"] <= 1.0


def test_config_lists_resolve_to_existing_files():
    assert [c.name for c in read_config_list("models-small.txt")] == ["Qwen2.5-0.5B.yaml"]
    assert len(read_config_list("models-all.txt")) >= 2


def test_missing_config_list_is_an_error_not_an_empty_run():
    with pytest.raises(FileNotFoundError, match="config list not found"):
        read_config_list("models-typo.txt")
