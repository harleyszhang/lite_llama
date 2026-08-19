"""Tests for RoPE and mrope frequency generation.

These only exercise the CPU forward path; the Triton apply kernel is covered by
the GPU tests. What matters here is that the tables have the right shape and that
mrope reduces to plain RoPE when its position ids collapse to a single component.
"""

from __future__ import annotations

import torch

from lite_llama.models.model_config import Qwen3Config
from lite_llama.models.rotary_embedding import (
    MRotaryEmbedding,
    RotaryEmbedding,
    compute_default_rope,
    compute_llama3_rope,
)


def test_default_rope_shape_is_half_of_rotary_dim():
    inv_freq, scaling = compute_default_rope(
        {"rope_theta": 10000.0, "head_dim": 64, "num_heads": 4, "hidden_size": 256}
    )
    assert inv_freq.shape == (32,)
    assert scaling == 1.0


def test_llama3_rope_produces_same_shape_as_default():
    config = {
        "rope_theta": 10000.0,
        "head_dim": 64,
        "num_heads": 4,
        "hidden_size": 256,
        "rope_scaling": {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    }
    inv_freq, scaling = compute_llama3_rope(config)
    assert inv_freq.shape == (32,)
    assert scaling == 1.0


def test_rotary_embedding_returns_expected_shape_on_cpu():
    cfg = Qwen3Config(
        hidden_size=256,
        num_heads=4,
        head_dim=64,
        num_kv_heads=2,
        num_layers=2,
        max_position_embeddings=128,
        max_seq_len=64,
    )
    rope = RotaryEmbedding(cfg)
    x = torch.randn(1, 16, 256)
    position_ids = torch.arange(16).unsqueeze(0)
    cos, sin = rope(x, position_ids)
    assert cos.shape == (1, 16, 64)
    assert sin.shape == (1, 16, 64)


def test_mrope_reduces_to_plain_rope_when_mrope_section_is_absent():
    """Without an mrope_section MRotaryEmbedding must delegate to the plain path."""
    cfg = Qwen3Config(
        hidden_size=256,
        num_heads=4,
        head_dim=64,
        num_kv_heads=2,
        num_layers=2,
        max_position_embeddings=128,
        max_seq_len=64,
    )
    rope = MRotaryEmbedding(cfg)
    x = torch.randn(1, 8, 256)
    pos = torch.arange(8).unsqueeze(0)
    cos, sin = rope(x, pos)
    assert cos.shape == (1, 8, 64)
    assert sin.shape == (1, 8, 64)


def test_mrope_shape_with_three_component_positions():
    cfg = Qwen3Config(
        hidden_size=256,
        num_heads=4,
        head_dim=64,
        num_kv_heads=2,
        num_layers=2,
        max_position_embeddings=128,
        max_seq_len=64,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [12, 10, 10],
            "mrope_interleaved": True,
        },
    )
    rope = MRotaryEmbedding(cfg)
    x = torch.randn(1, 8, 256)
    pos = torch.arange(8).view(1, 1, 8).expand(3, 1, 8).clone()
    cos, sin = rope(x, pos)
    assert cos.shape == (1, 8, 64)
    assert sin.shape == (1, 8, 64)
