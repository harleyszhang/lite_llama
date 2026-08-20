"""Tests for :func:`get_model_name_from_path`.

Small helper, real consequence: ``LLMEngine`` derives the tokenizer's
``use_fast`` flag from it (``"llava" not in name``), so a wrong answer loads the
wrong tokenizer class for LLaVA checkpoints.

The previous test for this function defined its own copy of the implementation
inside the test file and ran nothing, so the shipped helper was untested.
"""

from __future__ import annotations

import pytest

from lite_llama.utils.file_interface import get_model_name_from_path


@pytest.mark.parametrize(
    "path,expected",
    [
        pytest.param("my_weight/Qwen2.5-0.5B", "Qwen2.5-0.5B", id="relative"),
        pytest.param("/abs/path/to/llava-1.5-7b-hf", "llava-1.5-7b-hf", id="absolute"),
        pytest.param("my_weight/Qwen3-0.6B/", "Qwen3-0.6B", id="trailing-slash"),
        pytest.param("Qwen2.5-0.5B", "Qwen2.5-0.5B", id="bare-name"),
    ],
)
def test_returns_the_final_path_component(path, expected):
    assert get_model_name_from_path(path) == expected


def test_checkpoint_directories_keep_their_parent_name():
    """``.../run-name/checkpoint-500`` must not collapse to just "checkpoint-500".

    Training output directories are all named ``checkpoint-N``, so dropping the
    parent would make every run indistinguishable -- and would lose the "llava"
    substring the tokenizer decision depends on.
    """
    assert get_model_name_from_path("out/llava-run/checkpoint-500") == "llava-run_checkpoint-500"


def test_llava_checkpoint_name_still_contains_llava():
    """The property ``LLMEngine`` actually relies on, asserted directly."""
    name = get_model_name_from_path("my_weight/llava-1.5-7b-hf/")
    assert "llava" in name.lower()
