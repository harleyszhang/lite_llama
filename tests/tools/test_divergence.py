"""Tests for the whole-model divergence checker: naming the first bad layer.

The CPU tier pins the contract that needs no kernel: the report types'
verdicts, the error paths, and the bridge into the single-layer harness. The
numeric tier (GPU) builds a tiny random checkpoint, checks that a clean run
passes every layer, and that a perturbed MLP or attention block is named —
both the layer and the submodule. The argmax row is deliberately not asserted
on random weights: tiny models produce near-degenerate logits whose top-2 gap
sits below the bf16 noise floor, where agreement is a coin flip (the lesson
``test_qwen3_moe.py`` already pins).

Usage:
    pytest tests/tools/test_divergence.py
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from lite_llama.models.config import ModelConfig
from lite_llama.tools.accuracy import (
    DEFAULT_REL_THRESHOLD,
    DivergenceChecker,
    DivergenceReport,
    LayerDiff,
    LogitsDiff,
    SubmoduleDiff,
)
from lite_llama.tools.harness import Diff, SingleLayerHarness

#: Qwen3 dense at test size — the body shape ``test_harness.py`` pins, with
#: four layers so the layer after the perturbed one always exists: the first
#: divergence must be the perturbed layer, not the last.
_BODY = {
    "model_type": "qwen3",
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 256,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


def _checker(tmp_path, *, device: str = "cpu", seed: int = 0) -> DivergenceChecker:
    """Build a tiny random Qwen3 checkpoint and load both sides from disk.

    The HF model is constructed, checkpointed as safetensors, and re-read via
    :meth:`DivergenceChecker.from_checkpoint` — the same through-disk path the
    CLI takes, so the loader's weight mapping is exercised on the way in.
    """
    from transformers import Qwen3ForCausalLM

    (tmp_path / "config.json").write_text(json.dumps(_BODY))
    torch.manual_seed(seed)
    hf_config = ModelConfig.from_pretrained(tmp_path, max_seq_len=128).hf_config
    hf_model = Qwen3ForCausalLM(hf_config).eval()
    save_file(
        {key: value.detach().clone() for key, value in hf_model.state_dict().items()},
        str(tmp_path / "model.safetensors"),
        metadata={"format": "pt"},
    )
    return DivergenceChecker.from_checkpoint(tmp_path, device=device, max_seq_len=128)


def _perturb(module: nn.Module, factor: float = 4.0) -> None:
    """Scale a block's weights so its output leaves the noise band outright.

    Fourfold, not twofold: the perturbed block's share of the residual stream
    is unknown for random weights, and a 4x error on any non-negligible share
    clears the 5e-2 band with room to spare while a 2x one might ride the edge.
    """
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.mul_(factor)


def _diff(max_abs: float, rel: float) -> Diff:
    """A Diff with the given verdict numbers; mean_abs is irrelevant to them."""
    return Diff(max_abs=max_abs, mean_abs=max_abs / 2, rel=rel)


def _report(**overrides) -> DivergenceReport:
    """A hand-built report whose render/to_dict surface the tests read off."""
    body: dict = {
        "model_type": "qwen3",
        "num_layers": 2,
        "seq_len": 8,
        "rel_threshold": DEFAULT_REL_THRESHOLD,
        "layers": (
            LayerDiff(0, _diff(1e-4, 1e-5), diverged=False),
            LayerDiff(1, None, diverged=True, note="shape mismatch: (2, 3) vs (2, 4)"),
        ),
        "first_divergent": 1,
        "culprit": "mlp",
        "submodules": SubmoduleDiff(_diff(2.0, 0.5), _diff(0.0, 0.0)),
        "logits": LogitsDiff(max_abs=0.5, top_lite=3, top_hf=9),
    }
    body.update(overrides)
    return DivergenceReport(**body)


# --------------------------------------------------------------------------- #
# Report types: the verdicts, no kernel needed
# --------------------------------------------------------------------------- #
def test_submodule_culprit_splits_the_three_ways():
    """Whichever block is out of the band first is where the divergence starts."""
    inside, outside = _diff(1e-3, 1e-3), _diff(2.0, 0.5)
    assert SubmoduleDiff(outside, inside).culprit(0.1) == "self_attn"
    assert SubmoduleDiff(inside, outside).culprit(0.1) == "mlp"
    assert SubmoduleDiff(inside, inside).culprit(0.1) == "norm"


def test_submodule_culprit_treats_non_finite_as_divergence():
    """A NaN block must not slip through: every NaN comparison is False."""
    nan = _diff(float("nan"), float("nan"))
    assert SubmoduleDiff(nan, _diff(0.0, 0.0)).culprit(0.1) == "self_attn"
    assert SubmoduleDiff(_diff(0.0, 0.0), nan).culprit(0.1) == "mlp"


def test_logits_agreement_is_argmax_equality():
    assert LogitsDiff(max_abs=0.1, top_lite=7, top_hf=7).agree
    assert not LogitsDiff(max_abs=0.1, top_lite=7, top_hf=8).agree


def test_render_names_the_layer_and_the_culprit():
    text = _report().render()
    assert "first divergent layer: 1" in text
    assert "culprit: mlp" in text
    assert "self_attn" in text and "mlp" in text  # the submodule rows
    assert "disagree" in text  # the logits row flags the argmax mismatch


def test_render_of_a_clean_report_says_so():
    clean = _report(
        layers=(LayerDiff(0, _diff(1e-4, 1e-5), diverged=False),),
        first_divergent=None,
        culprit=None,
        submodules=None,
        logits=LogitsDiff(max_abs=0.5, top_lite=3, top_hf=3),
    )
    text = clean.render()
    assert "no divergence" in text
    assert "first divergent" not in text
    assert "disagree" not in text


def test_render_carries_a_shape_mismatch_note():
    """A shape disagreement is a wiring bug the note names, not a number."""
    assert "shape mismatch" in _report().render()


def test_to_dict_is_json_ready():
    data = _report().to_dict()
    assert data["ok"] is False
    assert data["first_divergent"] == 1
    assert data["culprit"] == "mlp"
    assert data["layers"][0]["diff"] is not None
    assert data["layers"][1]["diff"] is None
    assert data["layers"][1]["note"].startswith("shape mismatch")
    assert data["submodules"]["self_attn"]["rel"] == pytest.approx(0.5)
    assert data["logits"]["agree"] is False
    json.dumps(data)  # the whole payload survives a JSON round trip


# --------------------------------------------------------------------------- #
# The checker's contract around a real checkpoint (no forward on this tier)
# --------------------------------------------------------------------------- #
def test_rejects_a_reference_without_a_decoder_stack(tmp_path):
    """The reference must be built the transformers way; anything else fails loudly."""
    checker = _checker(tmp_path)
    with pytest.raises(ValueError, match="decoder stack"):
        DivergenceChecker(checker.config, checker.lite, nn.Module())


def test_run_refuses_a_batch_above_one(tmp_path):
    """The logits row names one argmax pair; a batch of two is a caller bug."""
    checker = _checker(tmp_path)
    ids = torch.randint(0, _BODY["vocab_size"], (2, 8))
    with pytest.raises(ValueError, match="batch 1"):
        checker.run(ids)


def test_harness_for_needs_a_checkpoint_directory(tmp_path):
    """A checker built by hand has no directory to read the harness's weights from."""
    checker = _checker(tmp_path)
    bare = DivergenceChecker(checker.config, checker.lite, checker.hf, device="cpu")
    with pytest.raises(ValueError, match="checkpoint directory"):
        bare.harness_for(0)


def test_harness_for_hands_the_layer_to_the_single_layer_harness(tmp_path):
    """The bridge: the named layer arrives loaded, ready for per-module timing."""
    checker = _checker(tmp_path)
    harness = checker.harness_for(1)
    assert isinstance(harness, SingleLayerHarness)
    assert harness.layer_index == 1
    assert all(not p.is_meta for p in harness.layer.parameters())


# --------------------------------------------------------------------------- #
# The numeric tier: a real forward on both sides (needs the Triton kernels)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.usefixtures("cuda_available")
def test_a_clean_checkpoint_passes_every_layer(tmp_path):
    """Same weights through both pipelines: every layer stays inside the band."""
    checker = _checker(tmp_path, device="cuda")
    ids = torch.randint(0, _BODY["vocab_size"], (1, 16), device="cuda")

    report = checker.run(ids)

    assert report.ok
    assert report.first_divergent is None
    assert report.culprit is None
    assert report.submodules is None
    assert len(report.layers) == _BODY["num_hidden_layers"]
    assert report.seq_len == 16
    # The argmax pair is reported but not asserted on random weights: tiny
    # models produce near-degenerate logits where agreement is a coin flip.
    assert report.logits is not None
    assert 0 <= report.logits.top_lite < _BODY["vocab_size"]
    assert 0 <= report.logits.top_hf < _BODY["vocab_size"]


@pytest.mark.gpu
@pytest.mark.usefixtures("cuda_available")
def test_a_perturbed_mlp_is_named(tmp_path):
    """Scaling layer 2's MLP weights: layer 2 is named, and the MLP within it."""
    checker = _checker(tmp_path, device="cuda")
    _perturb(checker.lite.layers[2].mlp)
    ids = torch.randint(0, _BODY["vocab_size"], (1, 16), device="cuda")

    report = checker.run(ids)

    assert report.first_divergent == 2
    assert report.culprit == "mlp"
    assert report.submodules is not None
    # The MLP's own output is far out of the band; the attention block of the
    # same layer ran on agreeing inputs and stayed inside it.
    assert report.submodules.mlp.rel > DEFAULT_REL_THRESHOLD
    assert report.submodules.self_attn.rel < DEFAULT_REL_THRESHOLD


@pytest.mark.gpu
@pytest.mark.usefixtures("cuda_available")
def test_a_perturbed_attention_is_named(tmp_path):
    """Scaling layer 1's attention weights: the attention block is the culprit.

    The MLP behind it is fed the perturbed stream too and may leave the band
    as well — but attention is asked first, which is the point of the ordering.
    """
    checker = _checker(tmp_path, device="cuda")
    _perturb(checker.lite.layers[1].self_attn)
    ids = torch.randint(0, _BODY["vocab_size"], (1, 16), device="cuda")

    report = checker.run(ids)

    assert report.first_divergent == 1
    assert report.culprit == "self_attn"
    assert report.submodules is not None
    assert report.submodules.self_attn.rel > DEFAULT_REL_THRESHOLD
