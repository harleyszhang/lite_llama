"""CPU-only tests for the config dataclasses."""

from __future__ import annotations

import pytest

from lite_llama.models.model_config import LlamaConfig, Qwen2Config, Qwen3Config


def test_llama_defaults_are_self_consistent():
    cfg = LlamaConfig()
    assert cfg.num_kv_heads == cfg.num_heads
    assert cfg.head_dim == cfg.hidden_size // cfg.num_heads
    assert cfg.intermediate_size == cfg.hidden_size * 4
    assert cfg.q_size == cfg.hidden_size
    assert cfg.kv_size == cfg.hidden_size


def test_from_dict_applies_hf_aliases():
    """HF names should map to the internal short names."""
    cfg = LlamaConfig.from_dict(
        {
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "num_key_value_heads": 4,
            "hidden_size": 256,
        }
    )
    assert cfg.num_heads == 8
    assert cfg.num_layers == 4
    assert cfg.num_kv_heads == 4


def test_from_dict_ignores_unknown_keys():
    cfg = LlamaConfig.from_dict({"hidden_size": 512, "future_field": "ignored"})
    assert cfg.hidden_size == 512


def test_qwen2_disables_sliding_window_by_default():
    cfg = Qwen2Config()
    assert cfg.sliding_window is None


def test_qwen3_supports_head_dim_bigger_than_hidden_over_heads():
    """Qwen3-0.6B ships with hidden=1024, num_heads=16, head_dim=128 -> q_size=2048."""
    cfg = Qwen3Config(hidden_size=1024, num_heads=16, head_dim=128)
    assert cfg.head_dim == 128
    assert cfg.q_size == 2048
    assert cfg.q_size != cfg.hidden_size


def test_validate_rejects_bad_gqa_ratio():
    with pytest.raises(ValueError, match="divisible"):
        Qwen2Config(num_heads=12, num_kv_heads=5)


def test_validate_rejects_context_beyond_position_bound():
    with pytest.raises(ValueError, match="max_seq_len"):
        LlamaConfig(max_seq_len=8192, max_position_embeddings=4096)
