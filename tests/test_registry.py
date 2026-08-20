"""Tests for the ModelRegistry lookup and error handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lite_llama.models.model_config import LlamaConfig, Qwen2Config, Qwen3Config
from lite_llama.models.registry import ModelRegistry, ModelSpec


def test_supported_types_covers_every_supported_model():
    assert set(ModelRegistry.supported_types()) == {
        "llama",
        "qwen2",
        "qwen3",
        "qwen3_moe",
        "llava",
        "qwen3_vl",
    }


def test_resolve_returns_expected_spec_for_qwen3():
    spec = ModelRegistry.resolve("qwen3")
    assert spec.model_type == "qwen3"
    assert spec.is_multimodal is False


def test_resolve_flags_multimodal_models():
    assert ModelRegistry.resolve("llava").is_multimodal
    assert ModelRegistry.resolve("qwen3_vl").is_multimodal


def test_resolve_unknown_model_lists_alternatives():
    with pytest.raises(ValueError, match="supported: "):
        ModelRegistry.resolve("mystery")


@pytest.mark.parametrize(
    "model_type,expected_cls",
    [("llama", LlamaConfig), ("qwen2", Qwen2Config), ("qwen3", Qwen3Config)],
)
def test_load_text_config_from_disk(tmp_path: Path, model_type: str, expected_cls):
    checkpoint_dir = tmp_path / model_type
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "hidden_size": 512,
                "num_attention_heads": 8,
                "num_hidden_layers": 4,
                "num_key_value_heads": 4,
                "vocab_size": 1024,
                "max_position_embeddings": 2048,
            }
        )
    )

    config, spec = ModelRegistry.load_config(checkpoint_dir, max_seq_len=1024)
    assert isinstance(config, expected_cls)
    assert config.max_seq_len == 1024
    assert spec.model_type == model_type


def test_missing_config_json_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ModelRegistry.load_config(tmp_path, max_seq_len=1024)


def test_model_spec_load_class_defers_import():
    """A misconfigured implementation string must fail only when actually used."""
    spec = ModelSpec(
        model_type="broken",
        config_loader=lambda p, m: None,
        implementation="lite_llama.models.does_not_exist:Missing",
    )
    with pytest.raises(ModuleNotFoundError):
        spec.load_class()
