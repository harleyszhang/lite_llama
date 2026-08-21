"""CPU-only tests for :class:`~lite_llama.models.config.ModelConfig`.

The config layer no longer declares its own schema, so what needs pinning changed:
not "does the alias table rename ``num_attention_heads``", but "does the HF config
get read correctly, including the fields transformers has moved between versions".

Real ``config.json`` bodies are used rather than hand-built ``PretrainedConfig``
instances, because the whole point of the refactor is that ``AutoConfig`` does the
parsing — building the config object by hand would test nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lite_llama.models.config import ModelConfig, read_model_type

# A Qwen2 body with ``rope_theta`` at the top level, i.e. the transformers 4.x
# serialisation. transformers 5.x folds it into ``rope_parameters`` on load, which
# is exactly the migration ModelConfig has to absorb.
QWEN2_BODY = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "num_hidden_layers": 24,
    "intermediate_size": 4864,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
}


def write_config(directory: Path, body: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(body))
    return directory


@pytest.fixture
def qwen2_dir(tmp_path: Path) -> Path:
    return write_config(tmp_path / "qwen2", QWEN2_BODY)


# --------------------------------------------------------------------------- #
# Geometry read off the HF config
# --------------------------------------------------------------------------- #
def test_geometry_comes_from_the_hf_config(qwen2_dir: Path):
    cfg = ModelConfig.from_pretrained(qwen2_dir, max_seq_len=1024)
    assert cfg.model_type == "qwen2"
    assert (cfg.num_layers, cfg.num_heads, cfg.num_kv_heads) == (24, 14, 2)
    assert cfg.hidden_size == 896
    assert cfg.intermediate_size == 4864
    assert cfg.vocab_size == 151936
    assert cfg.max_seq_len == 1024


def test_head_dim_defaults_to_hidden_over_heads(qwen2_dir: Path):
    """Qwen2 ships no ``head_dim``; it must fall back to ``hidden_size // num_heads``."""
    cfg = ModelConfig.from_pretrained(qwen2_dir)
    assert cfg.head_dim == 896 // 14 == 64
    assert cfg.q_size == cfg.hidden_size
    assert cfg.kv_size == 2 * 64


def test_explicit_head_dim_wins_over_the_derived_one(tmp_path: Path):
    """Qwen3-0.6B has hidden=1024, num_heads=16, head_dim=128 -> q_size=2048.

    Deriving ``head_dim`` here would give 64 and silently halve every attention
    projection, so the explicit field has to take precedence.
    """
    body = {
        **QWEN2_BODY,
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "qwen3", body))
    assert cfg.head_dim == 128
    assert cfg.q_size == 2048 != cfg.hidden_size
    assert cfg.kv_size == 1024


def test_missing_num_key_value_heads_means_no_gqa(tmp_path: Path):
    body = {k: v for k, v in QWEN2_BODY.items() if k != "num_key_value_heads"}
    body["model_type"] = "llama"
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "llama", body))
    assert cfg.num_kv_heads == cfg.num_heads


# --------------------------------------------------------------------------- #
# RoPE settings across transformers versions
# --------------------------------------------------------------------------- #
def test_rope_theta_survives_the_transformers_5_move(qwen2_dir: Path):
    """``rope_theta`` is a top-level field in 4.x and nested in 5.x; both must work.

    transformers 5.x drops the loose ``rope_theta`` attribute entirely, so reading
    it directly would silently fall back to 10000.0 and detune every position.
    """
    cfg = ModelConfig.from_pretrained(qwen2_dir)
    assert cfg.rope_parameters["rope_theta"] == 1000000.0
    assert cfg.rope_config["rope_theta"] == 1000000.0
    assert cfg.rope_config["rope_type"] == "default"


def test_rope_config_carries_the_geometry_the_kernel_needs(qwen2_dir: Path):
    cfg = ModelConfig.from_pretrained(qwen2_dir)
    rope = cfg.rope_config
    assert rope["head_dim"] == cfg.head_dim
    assert rope["hidden_size"] == cfg.hidden_size
    assert rope["num_heads"] == cfg.num_heads
    assert rope["partial_rotary_factor"] == 1.0


def test_mrope_section_is_preserved(tmp_path: Path):
    """mrope settings must reach the RoPE layer even when only nested."""
    body = {
        **QWEN2_BODY,
        "model_type": "qwen3",
        "head_dim": 128,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "rope_scaling": {
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        },
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "mrope", body))
    assert cfg.rope_config["mrope_section"] == [24, 20, 20]


def test_nested_rope_theta_is_not_lost(tmp_path: Path):
    """The Qwen3-VL regression this refactor fixes, in its smallest form.

    transformers 5.x serialises the RoPE base *only* inside ``rope_parameters``. The
    old config layer read the top-level ``rope_theta`` field, found nothing and fell
    back to its dataclass default of 10000.0 — a 500x error against Qwen3-VL's
    declared 5e6, enough to send some logit vectors in the opposite direction from
    the HuggingFace reference.
    """
    body = {
        "model_type": "qwen3_vl",
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 2560,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 36,
            "intermediate_size": 9728,
            "vocab_size": 151936,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            # No loose ``rope_theta``: this is exactly how transformers 5.x writes it.
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5000000,
                "mrope_section": [24, 20, 20],
                "mrope_interleaved": True,
            },
        },
        "vision_config": {"model_type": "qwen3_vl_vision", "hidden_size": 1152, "num_heads": 16},
        "image_token_id": 151655,
        "video_token_id": 151656,
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "qwen3_vl", body), max_seq_len=2048)
    assert cfg.rope_config["rope_theta"] == 5000000
    assert cfg.rope_config["mrope_section"] == [24, 20, 20]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validate_rejects_bad_gqa_ratio(tmp_path: Path):
    body = {**QWEN2_BODY, "num_attention_heads": 12, "num_key_value_heads": 5, "head_dim": 64}
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig.from_pretrained(write_config(tmp_path / "bad_gqa", body))


def test_validate_rejects_head_dim_the_kernels_cannot_tile(tmp_path: Path):
    body = {**QWEN2_BODY, "head_dim": 60}
    with pytest.raises(ValueError, match="multiple of 8"):
        ModelConfig.from_pretrained(write_config(tmp_path / "bad_head_dim", body))


def test_validate_rejects_context_beyond_position_bound(qwen2_dir: Path):
    with pytest.raises(ValueError, match="max_seq_len"):
        ModelConfig.from_pretrained(qwen2_dir, max_seq_len=65536)


# --------------------------------------------------------------------------- #
# Fall-through to the HF text config
# --------------------------------------------------------------------------- #
def test_unnormalised_fields_fall_through_to_the_text_config(qwen2_dir: Path):
    """Model code reads HF field names directly instead of re-declaring each one."""
    cfg = ModelConfig.from_pretrained(qwen2_dir)
    assert cfg.rms_norm_eps == 1e-6
    assert cfg.tie_word_embeddings is True


def test_unknown_field_names_the_config_it_looked_in(qwen2_dir: Path):
    cfg = ModelConfig.from_pretrained(qwen2_dir)
    with pytest.raises(AttributeError, match="no attribute 'not_a_field'"):
        _ = cfg.not_a_field


def test_vision_language_config_unwraps_its_text_config(tmp_path: Path):
    """A multimodal wrapper must expose the *decoder's* geometry, not the wrapper's."""
    body = {
        "model_type": "llava",
        "text_config": {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "vocab_size": 32064,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
        },
        "vision_config": {
            "model_type": "clip_vision_model",
            "hidden_size": 1024,
            "num_attention_heads": 16,
        },
        "image_token_index": 32000,
        "projector_hidden_act": "gelu",
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default",
    }
    cfg = ModelConfig.from_pretrained(write_config(tmp_path / "llava", body))
    assert cfg.model_type == "llava"  # registry key: the *outer* type
    assert cfg.num_layers == 32
    assert cfg.hidden_size == 4096
    assert cfg.vocab_size == 32064
    # The vision tower is reached through the untouched HF config.
    assert cfg.hf_config.vision_config.hidden_size == 1024


# --------------------------------------------------------------------------- #
# read_model_type
# --------------------------------------------------------------------------- #
def test_read_model_type_returns_the_lowercased_type(qwen2_dir: Path):
    assert read_model_type(qwen2_dir) == "qwen2"


def test_read_model_type_reports_a_missing_config(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_model_type(tmp_path)


def test_read_model_type_reports_malformed_json(tmp_path: Path):
    (tmp_path / "config.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_model_type(tmp_path)


def test_read_model_type_reports_a_config_without_the_field(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="model_type"):
        read_model_type(tmp_path)
