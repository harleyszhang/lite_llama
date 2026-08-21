"""Tests for RoPE and mrope frequency generation.

These only exercise the CPU forward path; the Triton apply kernel is covered by
the GPU tests. What matters here is that the tables have the right shape, that
mrope reduces to plain RoPE when its position ids collapse to a single component,
and that the flat config mapping the layer now takes is exactly what
:attr:`~lite_llama.models.config.ModelConfig.rope_config` produces.
"""

from __future__ import annotations

import json

import pytest
import torch

from lite_llama.models.config import ModelConfig
from lite_llama.models.rotary_embedding import (
    MRotaryEmbedding,
    RotaryEmbedding,
    compute_default_rope,
    compute_llama3_rope,
)

_BASE_CONFIG = {"rope_theta": 10000.0, "head_dim": 64, "num_heads": 4, "hidden_size": 256}


def _rope_config_from_json(tmp_path, **overrides) -> dict:
    """Build a real ``rope_config`` by round-tripping a config.json through AutoConfig."""
    body = {
        "model_type": "qwen3",
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 512,
        "vocab_size": 1024,
        "head_dim": 64,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        **overrides,
    }
    (tmp_path / "config.json").write_text(json.dumps(body))
    return ModelConfig.from_pretrained(tmp_path, max_seq_len=64).rope_config


def test_default_rope_shape_is_half_of_rotary_dim():
    inv_freq, scaling = compute_default_rope(_BASE_CONFIG)
    assert inv_freq.shape == (32,)
    assert scaling == 1.0


def test_llama3_rope_reads_its_factors_from_the_flat_config():
    """The rescaling knobs are top-level keys of ``rope_config``, not a nested dict."""
    config = {
        **_BASE_CONFIG,
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    inv_freq, scaling = compute_llama3_rope(config)
    assert inv_freq.shape == (32,)
    assert scaling == 1.0
    # Long wavelengths must actually have been divided down.
    plain, _ = compute_default_rope(config)
    assert inv_freq[-1] < plain[-1]


def test_unsupported_rope_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported rope_type"):
        RotaryEmbedding({**_BASE_CONFIG, "rope_type": "made_up"})


def test_inv_freq_stays_fp32():
    """fp16 ``inv_freq`` costs ~0.1 rad of phase error by position 500.

    The buffer is non-persistent and computed from the config, so nothing in the
    load path should be casting it along with the weights.
    """
    rope = RotaryEmbedding(_BASE_CONFIG)
    assert rope.inv_freq.dtype == torch.float32


def test_rotary_embedding_returns_expected_shape_on_cpu(tmp_path):
    rope = RotaryEmbedding(_rope_config_from_json(tmp_path))
    x = torch.randn(1, 16, 256)
    position_ids = torch.arange(16).unsqueeze(0)
    cos, sin = rope(x, position_ids)
    assert cos.shape == (1, 16, 64)
    assert sin.shape == (1, 16, 64)


def test_mrope_reduces_to_plain_rope_when_mrope_section_is_absent(tmp_path):
    """Without an mrope_section MRotaryEmbedding must delegate to the plain path."""
    config = _rope_config_from_json(tmp_path)
    plain = RotaryEmbedding(config)
    mrope = MRotaryEmbedding(config)
    x = torch.randn(1, 8, 256)
    pos = torch.arange(8).unsqueeze(0)

    assert mrope.mrope_section is None
    for got, want in zip(mrope(x, pos), plain(x, pos), strict=True):
        torch.testing.assert_close(got, want)


def test_mrope_shape_with_three_component_positions(tmp_path):
    config = _rope_config_from_json(
        tmp_path,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [12, 10, 10],
            "mrope_interleaved": True,
        },
    )
    rope = MRotaryEmbedding(config)
    assert rope.mrope_section == [12, 10, 10]

    x = torch.randn(1, 8, 256)
    pos = torch.arange(8).view(1, 1, 8).expand(3, 1, 8).clone()
    cos, sin = rope(x, pos)
    assert cos.shape == (1, 8, 64)
    assert sin.shape == (1, 8, 64)


def test_mrope_with_identical_components_equals_plain_rope(tmp_path):
    """When t == h == w the interleaved merge must collapse to the 1-D table."""
    config = _rope_config_from_json(
        tmp_path,
        rope_scaling={"rope_type": "default", "mrope_section": [12, 10, 10]},
    )
    x = torch.randn(1, 8, 256)
    flat = torch.arange(8).unsqueeze(0)
    stacked = flat.view(1, 1, 8).expand(3, 1, 8).clone()

    plain = RotaryEmbedding(config)(x, flat)
    mroped = MRotaryEmbedding(config)(x, stacked)
    for got, want in zip(mroped, plain, strict=True):
        torch.testing.assert_close(got, want)


def test_mrope_section_must_have_three_entries(tmp_path):
    config = _rope_config_from_json(
        tmp_path, rope_scaling={"rope_type": "default", "mrope_section": [12, 10]}
    )
    with pytest.raises(ValueError, match="3 entries"):
        MRotaryEmbedding(config)
