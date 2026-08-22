"""Unit tests for the HF-key -> lite_llama-parameter mapping.

Two tiers, both pure CPU and free of real checkpoints:

1. **Key translation** (:func:`lite_llama.models.weights.translate_text_key`) as a
   pure function: one assertion per key shape, no tensors involved.
2. **Coverage accounting** in :func:`lite_llama.models.weights.load_weights`. This
   is the safety net that makes the rest of the suite trustworthy: a rename rule
   that stops matching leaves a parameter unwritten, and an unwritten parameter
   produces a model that runs and returns nonsense instead of failing.

Round-trip parity against real HuggingFace models lives in
``test_weight_parity.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lite_llama.executor.weight_utils import hf_weight_files
from lite_llama.models import weights

# --------------------------------------------------------------------------- #
# Tier 1: key translation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "key,expected",
    [
        # Untouched: these already carry lite_llama parameter names.
        ("embed_tokens.weight", "embed_tokens.weight"),
        ("layers.3.mlp.gate_proj.weight", "layers.3.mlp.gate_proj.weight"),
        ("layers.3.mlp.up_proj.weight", "layers.3.mlp.up_proj.weight"),
        ("layers.3.mlp.down_proj.weight", "layers.3.mlp.down_proj.weight"),
        # Flattened: the module level is folded into the parameter name.
        ("norm.weight", "norm_weight"),
        ("lm_head.weight", "lm_head_weight"),
        ("layers.3.input_layernorm.weight", "layers.3.input_layernorm_weight"),
        ("layers.3.post_attention_layernorm.weight", "layers.3.post_attention_layernorm_weight"),
        # Projections are submodules (LinearBase), so their parameters use dots.
        ("layers.3.self_attn.q_proj.weight", "layers.3.self_attn.q_proj.weight"),
        ("layers.3.self_attn.q_proj.bias", "layers.3.self_attn.q_proj.bias"),
        ("layers.3.self_attn.o_proj.weight", "layers.3.self_attn.o_proj.weight"),
        ("layers.3.self_attn.q_norm.weight", "layers.3.self_attn.q_norm_weight"),
        ("layers.3.self_attn.k_norm.weight", "layers.3.self_attn.k_norm_weight"),
        # MoE router. Must not be confused with the dense ``mlp.gate_proj``.
        ("layers.3.mlp.gate.weight", "layers.3.mlp.gate_weight"),
        # Fused pairs.
        ("layers.3.self_attn.k_proj.weight", "layers.3.self_attn.kv_proj.weight"),
        ("layers.3.self_attn.v_proj.weight", "layers.3.self_attn.kv_proj.weight"),
        ("layers.3.self_attn.k_proj.bias", "layers.3.self_attn.kv_proj.bias"),
        ("layers.3.self_attn.v_proj.bias", "layers.3.self_attn.kv_proj.bias"),
        # Stacked experts.
        ("layers.3.mlp.experts.7.gate_proj.weight", "layers.3.mlp.experts.gate_up_proj"),
        ("layers.3.mlp.experts.7.up_proj.weight", "layers.3.mlp.experts.gate_up_proj"),
        ("layers.3.mlp.experts.7.down_proj.weight", "layers.3.mlp.experts.down_proj"),
    ],
)
def test_translate_text_key_names_the_right_parameter(key: str, expected: str):
    name, _ = weights.translate_text_key(key)
    assert name == expected


def test_k_and_v_fill_opposite_halves():
    """Both map to one parameter, so the halves are the only thing keeping them apart."""
    param = torch.zeros(8, 3)
    _, k_dest = weights.translate_text_key("layers.0.self_attn.k_proj.weight")
    _, v_dest = weights.translate_text_key("layers.0.self_attn.v_proj.weight")

    k_dest(param).fill_(1)
    v_dest(param).fill_(2)
    assert torch.equal(param[:4], torch.ones(4, 3))
    assert torch.equal(param[4:], torch.full((4, 3), 2.0))


def test_expert_gate_and_up_fill_opposite_halves_of_their_own_slice():
    """gate/up are fused *within* each expert's slice, not across experts."""
    param = torch.zeros(3, 4, 5)  # [experts, 2 * moe_inter, hidden]
    _, gate = weights.translate_text_key("layers.0.mlp.experts.1.gate_proj.weight")
    _, up = weights.translate_text_key("layers.0.mlp.experts.1.up_proj.weight")

    gate(param).fill_(1)
    up(param).fill_(2)
    assert torch.equal(param[1, :2], torch.ones(2, 5))
    assert torch.equal(param[1, 2:], torch.full((2, 5), 2.0))
    # Other experts untouched.
    assert param[0].abs().sum() == 0
    assert param[2].abs().sum() == 0


def test_strip_prefix_reports_a_non_match():
    assert weights.strip_prefix("model.layers.0", "model.") == "layers.0"
    assert weights.strip_prefix("visual.blocks.0", "model.") is None
    # The empty prefix matches everything, which is what the multimodal routers
    # use as their fall-through rule.
    assert weights.strip_prefix("lm_head.weight", "") == "lm_head.weight"


# --------------------------------------------------------------------------- #
# Tier 2: coverage accounting
# --------------------------------------------------------------------------- #


class _TwoParams(nn.Module):
    """Minimal stand-in: a plain parameter, a fused pair, and a tie target."""

    def __init__(self) -> None:
        super().__init__()
        self.plain = nn.Parameter(torch.zeros(2, 3))
        self.fused = nn.Parameter(torch.zeros(4, 3))
        self.mirror = nn.Parameter(torch.zeros(2, 3))


def _translate(key: str) -> weights.Target:
    table: dict[str, weights.Target] = {
        "plain": ("plain", weights.whole),
        "fused_low": ("fused", weights.half(0)),
        "fused_high": ("fused", weights.half(1)),
        "mirror": ("mirror", weights.whole),
        "ignore_me": None,
    }
    return table.get(key, (key, weights.whole))


def _stream(*, drop: tuple[str, ...] = ()) -> list[tuple[str, torch.Tensor]]:
    """A complete checkpoint stream for :class:`_TwoParams`, minus ``drop``."""
    full = {
        "plain": torch.ones(2, 3),
        "fused_low": torch.full((2, 3), 4.0),
        "fused_high": torch.full((2, 3), 5.0),
        "mirror": torch.full((2, 3), 7.0),
        "ignore_me": torch.zeros(1),
    }
    return [(k, v) for k, v in full.items() if k not in drop]


def test_full_coverage_succeeds():
    model = _TwoParams()
    weights.load_weights(model, _stream(), _translate)

    assert torch.equal(model.plain, torch.ones(2, 3))
    assert torch.equal(model.fused[:2], torch.full((2, 3), 4.0))
    assert torch.equal(model.fused[2:], torch.full((2, 3), 5.0))
    assert torch.equal(model.mirror, torch.full((2, 3), 7.0))


def test_missing_parameter_is_reported_by_name():
    with pytest.raises(ValueError, match=r"never written.*plain"):
        weights.load_weights(_TwoParams(), _stream(drop=("plain",)), _translate)


def test_half_written_fused_parameter_is_rejected():
    """Only half of a fused K/V arriving is the failure a key-set check cannot see."""
    with pytest.raises(ValueError, match=r"partially written.*fused"):
        weights.load_weights(_TwoParams(), _stream(drop=("fused_high",)), _translate)


def test_a_parameter_written_twice_is_rejected():
    """Two keys competing for one destination silently loses whichever lost the race."""
    duplicated = [*_stream(), ("plain", torch.zeros(2, 3))]
    with pytest.raises(ValueError, match=r"plain \(12 of 6 elements\)"):
        weights.load_weights(_TwoParams(), duplicated, _translate)


def test_shape_mismatch_names_both_shapes():
    with pytest.raises(ValueError, match=r"shape \(3, 3\) but 'plain' expects \(2, 3\)"):
        weights.load_weights(_TwoParams(), [("plain", torch.zeros(3, 3))], _translate)


def test_key_mapping_to_a_nonexistent_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown parameter 'typo'"):
        weights.load_weights(_TwoParams(), [("typo", torch.zeros(2, 3))], _translate)


def test_tie_fills_a_parameter_the_checkpoint_omitted():
    """Tied checkpoints ship no ``lm_head``, and the coverage check would reject the gap."""
    model = _TwoParams()
    weights.load_weights(model, _stream(drop=("mirror",)), _translate, tied={"mirror": "plain"})
    assert torch.equal(model.mirror, model.plain)


def test_tie_does_not_override_a_shipped_tensor():
    """A tie is a fallback: an untied checkpoint's own lm_head must survive it."""
    model = _TwoParams()
    weights.load_weights(model, _stream(), _translate, tied={"mirror": "plain"})
    assert torch.equal(model.mirror, torch.full((2, 3), 7.0))


# --------------------------------------------------------------------------- #
# Checkpoint file discovery
# --------------------------------------------------------------------------- #


def test_safetensors_wins_over_legacy_bin(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "pytorch_model.bin").touch()
    assert [p.name for p in hf_weight_files(tmp_path)] == ["model.safetensors"]


def test_all_shards_are_returned_in_order(tmp_path):
    for name in ("model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"):
        (tmp_path / name).touch()
    assert [p.name for p in hf_weight_files(tmp_path)] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_a_directory_without_weights_says_what_it_wanted(tmp_path):
    with pytest.raises(FileNotFoundError, match="HuggingFace checkpoint directory"):
        hf_weight_files(tmp_path)
