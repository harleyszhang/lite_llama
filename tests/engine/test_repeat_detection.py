"""CPU-only tests for the repetition circuit breaker and stop-token loading."""

from __future__ import annotations

import json
from pathlib import Path

from lite_llama.cli import _infer_prompter_type, _is_instruct_checkpoint
from lite_llama.engine.stop_criteria import detect_repetition, load_stop_token_ids

# --------------------------------------------------------------------- #
# detect_repetition
# --------------------------------------------------------------------- #


def test_counting_loop_is_detected():
    """Loops whose only variation is a number must still be caught."""
    unit = (
        "I have been a member of the Rotary Foundation's Board of Directors "
        "for {n} years. I have been a member of the Rotary Foundation's "
        "Corporate Board of Directors for {n} years. "
    )
    text = "".join(unit.format(n=n) for n in (13, 11, 10, 9, 8, 7))
    assert detect_repetition(text)


def test_verbatim_loop_is_detected():
    # 46-char unit x 12 = 552 chars, past the 4 * window (512) entry threshold.
    unit = "The quick brown fox jumps over the lazy dog. " * 12
    assert detect_repetition(unit)


def test_normal_prose_is_not_flagged():
    text = (
        "Paris is the capital of France. It is known for the Eiffel Tower, "
        "the Louvre museum, and its cafe culture along the Seine. The city "
        "has a population of about two million people in the city proper, "
        "and more than twelve million in the wider metropolitan area, which "
        "makes it one of the largest urban agglomerations in all of Europe "
        "and the most visited tourist destination in the entire world today."
    )
    assert not detect_repetition(text)


def test_short_text_never_triggers():
    assert not detect_repetition("spam " * 50)


def test_custom_window_and_reps():
    unit = "ab" * 8  # 16 chars
    text = unit * 5
    assert detect_repetition(text, window=16, min_reps=3)


# --------------------------------------------------------------------- #
# load_stop_token_ids
# --------------------------------------------------------------------- #


class _FakeTokenizer:
    def __init__(self, eos_token_id: int | None):
        self.eos_token_id = eos_token_id


def test_stop_ids_merge_tokenizer_and_generation_config(tmp_path: Path):
    gen_cfg = tmp_path / "generation_config.json"
    gen_cfg.write_text(json.dumps({"eos_token_id": [151645, 151643]}))
    ids = load_stop_token_ids(str(tmp_path), _FakeTokenizer(151643))
    # The list adds <|im_end|> (151645) on top of the tokenizer's <|endoftext|>.
    assert ids == {151643, 151645}


def test_stop_ids_fall_back_to_tokenizer_without_generation_config(tmp_path: Path):
    ids = load_stop_token_ids(str(tmp_path), _FakeTokenizer(2))
    assert ids == {2}


def test_stop_ids_survive_broken_generation_config(tmp_path: Path):
    (tmp_path / "generation_config.json").write_text("{not json")
    ids = load_stop_token_ids(str(tmp_path), _FakeTokenizer(2))
    assert ids == {2}


def test_stop_ids_handle_scalar_eos_in_generation_config(tmp_path: Path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 13}))
    ids = load_stop_token_ids(str(tmp_path), _FakeTokenizer(None))
    assert ids == {13}


# --------------------------------------------------------------------- #
# CLI checkpoint classification
# --------------------------------------------------------------------- #


def _make_checkpoint(tmp_path: Path, name: str, model_type: str) -> str:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(d)


def test_qwen3_without_instruct_suffix_is_chat(tmp_path: Path):
    # Qwen3-0.6B is an instruct model despite the bare name.
    d = _make_checkpoint(tmp_path, "Qwen3-0.6B", "qwen3")
    assert _is_instruct_checkpoint(d)
    assert _infer_prompter_type(d) == "qwen2"  # ChatML prompter


def test_qwen3_base_variant_is_base(tmp_path: Path):
    d = _make_checkpoint(tmp_path, "Qwen3-0.6B-Base", "qwen3")
    assert not _is_instruct_checkpoint(d)
    assert _infer_prompter_type(d) == "empty"


def test_qwen25_base_stays_base(tmp_path: Path):
    d = _make_checkpoint(tmp_path, "Qwen2.5-0.5B", "qwen2")
    assert not _is_instruct_checkpoint(d)
    assert _infer_prompter_type(d) == "empty"


def test_qwen25_instruct_is_chat(tmp_path: Path):
    d = _make_checkpoint(tmp_path, "Qwen2.5-0.5B-Instruct", "qwen2")
    assert _is_instruct_checkpoint(d)
    assert _infer_prompter_type(d) == "qwen2"


def test_llama_chat_models_route_by_name(tmp_path: Path):
    d = _make_checkpoint(tmp_path, "llama-2-7b-chat", "llama")
    assert _is_instruct_checkpoint(d)
    assert _infer_prompter_type(d) == "llama"


def test_llama_base_stays_base(tmp_path: Path):
    d = _make_checkpoint(tmp_path, "llama-2-7b", "llama")
    assert not _is_instruct_checkpoint(d)
    assert _infer_prompter_type(d) == "empty"
