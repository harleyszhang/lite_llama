"""Tests for the ModelRegistry lookup and error handling.

The registry shrank to "``model_type`` -> implementation class + multimodal flag",
so these tests pin the two things that can actually break: the table's contents
(a model dropped from it becomes unservable) and the laziness of the import (a
transformers build missing one architecture must not break the other five).
"""

from __future__ import annotations

import pytest

from lite_llama.models.registry import ModelRegistry, ModelSpec

EXPECTED_TYPES = {"llama", "qwen2", "qwen3", "qwen3_moe", "llava", "qwen3_vl"}


def test_supported_types_covers_every_supported_model():
    assert set(ModelRegistry.supported_types()) == EXPECTED_TYPES


def test_resolve_returns_expected_spec_for_qwen3():
    spec = ModelRegistry.resolve("qwen3")
    assert spec.model_type == "qwen3"
    assert spec.is_multimodal is False


def test_resolve_flags_multimodal_models():
    assert ModelRegistry.resolve("llava").is_multimodal
    assert ModelRegistry.resolve("qwen3_vl").is_multimodal
    assert not ModelRegistry.resolve("qwen3_moe").is_multimodal


def test_resolve_is_case_insensitive():
    assert ModelRegistry.resolve("Qwen3_MoE").model_type == "qwen3_moe"


def test_resolve_unknown_model_lists_alternatives():
    with pytest.raises(ValueError, match="supported: "):
        ModelRegistry.resolve("mystery")


@pytest.mark.parametrize("model_type", sorted(EXPECTED_TYPES))
def test_every_registered_implementation_imports(model_type: str):
    """A typo in an implementation path would otherwise only surface at load time."""
    import torch.nn as nn

    cls = ModelRegistry.resolve(model_type).load_class()
    assert issubclass(cls, nn.Module)


def test_model_spec_load_class_defers_import():
    """A misconfigured implementation string must fail only when actually used."""
    spec = ModelSpec(model_type="broken", implementation="lite_llama.models.nope:Missing")
    with pytest.raises(ModuleNotFoundError):
        spec.load_class()
