"""Tests for RoPE and mrope frequency generation.

Default and llama3 rescaling factors read from flat configs, fp32
inv-freq preservation, cached-table equality with per-step maths, and
rejection of unsupported rope types.

Usage:
    pytest tests/models/test_rotary_embedding.py
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from rapid_llm.models.config import ModelConfig
from rapid_llm.modules.rotary_embedding import (
    MRotaryEmbedding,
    RotaryEmbedding,
    compute_default_rope,
    compute_llama3_rope,
    compute_yarn_rope,
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


def test_cached_table_matches_per_step_computation(tmp_path):
    """The precomputed cache must return what recomputing per step returns.

    A cache row is built from the same fp32 outer product the fallback runs,
    so the two paths must agree bit-for-bit; drifting apart would silently
    rotate positions wrongly only for models built with a max_seq_len.
    """
    config = _rope_config_from_json(tmp_path)  # carries max_seq_len=64
    cached = RotaryEmbedding(config)
    assert cached.max_seq_len == 64
    bare = RotaryEmbedding({k: v for k, v in config.items() if k != "max_seq_len"})

    x = torch.randn(1, 8, 256)
    position_ids = torch.arange(40, 48).unsqueeze(0)  # mid-table offsets
    for got, want in zip(cached(x, position_ids), bare(x, position_ids), strict=True):
        torch.testing.assert_close(got, want)


def test_cache_follows_module_onto_the_device_and_dtype(tmp_path):
    """The ``.to()`` must move the caches like any buffer, or the gather would miss."""
    rope = RotaryEmbedding(_rope_config_from_json(tmp_path))
    rope.to(torch.float16)
    assert rope.cos_cache.dtype == torch.float16
    x = torch.randn(1, 4, 256, dtype=torch.float16)
    cos, _ = rope(x, torch.arange(4).unsqueeze(0))
    assert cos.dtype == torch.float16


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


# --------------------------------------------------------------------------- #
# YaRN (DeepSeek-V2/V3): every number is checked against HF's own implementation
# --------------------------------------------------------------------------- #
#: DeepSeek-V2-Lite's rope_scaling, verbatim. Its ``mscale == mscale_all_dim``
#: cancels the attention scaling to exactly 1.0.
_V2_LITE_YARN = {
    "factor": 40.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 0.707,
    "mscale_all_dim": 0.707,
    "original_max_position_embeddings": 4096,
}


def _hf_yarn(config_dir) -> tuple[torch.Tensor, float]:
    """``(inv_freq, attention_scaling)`` as HF's YaRNScaledRotaryEmbedding computes them."""
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import _compute_yarn_parameters

    return _compute_yarn_parameters(AutoConfig.from_pretrained(config_dir), None)


def test_yarn_inv_freq_and_scaling_match_transformers(tmp_path):
    config = _rope_config_from_json(
        tmp_path, max_position_embeddings=163840, rope_scaling={"type": "yarn", **_V2_LITE_YARN}
    )
    got_inv, got_scale = compute_yarn_rope(config)
    want_inv, want_scale = _hf_yarn(tmp_path)
    torch.testing.assert_close(got_inv, want_inv)
    assert got_scale == want_scale == 1.0


def test_yarn_distinct_mscale_matches_transformers(tmp_path):
    """``mscale != mscale_all_dim`` keeps the full ratio formula (the V3 spelling)."""
    yarn = {**_V2_LITE_YARN, "mscale": 0.707, "mscale_all_dim": 1.0}
    config = _rope_config_from_json(
        tmp_path, max_position_embeddings=163840, rope_scaling={"type": "yarn", **yarn}
    )
    got_inv, got_scale = compute_yarn_rope(config)
    want_inv, want_scale = _hf_yarn(tmp_path)
    torch.testing.assert_close(got_inv, want_inv)
    assert got_scale == want_scale != 1.0


def test_yarn_default_betas_and_plain_scaling_match_transformers(tmp_path):
    """No ``beta_*``/``mscale`` keys: the 32/1 defaults and the paper's scaling kick in."""
    config = _rope_config_from_json(
        tmp_path,
        max_position_embeddings=16384,
        rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096},
    )
    got_inv, got_scale = compute_yarn_rope(config)
    want_inv, want_scale = _hf_yarn(tmp_path)
    torch.testing.assert_close(got_inv, want_inv)
    assert got_scale == want_scale == pytest.approx(0.1 * math.log(4.0) + 1.0)


def test_yarn_derives_factor_when_unstated():
    """A checkpoint stating the extended length instead of the ratio (V3 spelling).

    HF's validator makes ``factor`` mandatory on the config-file path, so this
    derivation is checked for self-consistency instead of against HF.
    """
    derived, _ = compute_yarn_rope(
        {
            **_BASE_CONFIG,
            "rope_type": "yarn",
            "max_position_embeddings": 16384,
            "original_max_position_embeddings": 4096,
        }
    )
    explicit, _ = compute_yarn_rope(
        {
            **_BASE_CONFIG,
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 4096,
        }
    )
    torch.testing.assert_close(derived, explicit)


def test_yarn_untruncated_correction_range_matches_transformers(tmp_path):
    yarn = {**_V2_LITE_YARN, "truncate": False}
    config = _rope_config_from_json(
        tmp_path, max_position_embeddings=163840, rope_scaling={"type": "yarn", **yarn}
    )
    got_inv, _ = compute_yarn_rope(config)
    want_inv, _ = _hf_yarn(tmp_path)
    torch.testing.assert_close(got_inv, want_inv)


def test_yarn_cos_sin_cache_matches_hf_full_table(tmp_path):
    """The precomputed table must carry the attention scaling, row for row."""
    config = _rope_config_from_json(
        tmp_path, max_position_embeddings=163840, rope_scaling={"type": "yarn", **_V2_LITE_YARN}
    )
    rope = RotaryEmbedding(config)  # max_seq_len=64 -> the cache path
    assert rope.attention_scaling == 1.0

    want_inv, want_scale = _hf_yarn(tmp_path)
    freqs = torch.outer(torch.arange(64, dtype=torch.float32), want_inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    torch.testing.assert_close(rope.cos_cache, emb.cos() * want_scale)
    torch.testing.assert_close(rope.sin_cache, emb.sin() * want_scale)
