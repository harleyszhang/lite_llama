"""The Triton grouped-topk kernel against the torch reference.

The routing semantics are pinned by ``tests/models/test_grouped_topk.py``
against a naive per-token vLLM transcription; this file pins the kernel
production dispatches to on CUDA: the whole parameter grid re-run through
``grouped_topk`` must reproduce ``grouped_topk_torch`` exactly (selections
identical — seeded randn makes ties measure-zero — weights to
reduction-order tolerance), the hand-built rule examples must survive the
fusion, and the wrapper must fall back to the torch path wherever the
kernel's geometry contract does not hold, so the two paths never disagree
on what runs.

``tests/kernels`` is auto-marked ``gpu`` by conftest.

Usage:
    pytest tests/kernels/test_grouped_topk_kernel.py
"""

from __future__ import annotations

import pytest
import torch

from lite_llama.kernels.ops.moe.grouped_topk import grouped_topk, grouped_topk_torch
from tests.models.test_grouped_topk import _GRID, _assert_same_routing


# --------------------------------------------------------------------------- #
# The kernel against the torch reference, across the semantic grid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "scoring,bias,renormalize,scale,groups,topk_group,top_k",
    _GRID,
)
def test_kernel_matches_torch_reference(
    scoring, bias, renormalize, scale, groups, topk_group, top_k
):
    """The fused kernel must reproduce the reference on every grid combination.

    Group-count and top-k vary at launch time (runtime loop bounds), so the
    sixteen-expert grid compiles one kernel per (scoring, bias, renorm,
    group-block) signature and reuses it — the grid stays cheap to run.
    """
    torch.manual_seed(0)
    logits = torch.randn(32, 16, device="cuda", dtype=torch.float32)
    bias_t = torch.randn(16, device="cuda", dtype=torch.float32) if bias else None
    kwargs = {
        "top_k": top_k,
        "renormalize": renormalize,
        "num_expert_group": groups,
        "topk_group": topk_group,
        "scoring_func": scoring,
        "routed_scaling_factor": scale,
        "e_score_correction_bias": bias_t,
    }
    weights, ids = grouped_topk(logits, **kwargs)
    ref_weights, ref_ids = grouped_topk_torch(logits, **kwargs)
    assert weights.dtype == torch.float32, "the kernel must return fp32 weights"
    assert ids.dtype == torch.int64, "the kernel must return the reference's id dtype"
    _assert_same_routing(
        weights,
        ids,
        ref_weights,
        ref_ids,
        f"kernel-vs-torch scoring={scoring} bias={bias} renorm={renormalize} "
        f"scale={scale} groups={groups} topk_group={topk_group} top_k={top_k}",
    )


@pytest.mark.parametrize("num_experts,groups", [(256, 8), (160, 8), (64, 8), (8, 2)])
def test_kernel_matches_torch_reference_at_real_geometries(num_experts, groups):
    """Real checkpoint geometries, including the non-power-of-two ones.

    V2 routes 160 experts — the kernel pads to 256 lanes, and a padding lane
    scoring 0 must never outscore a real one in the group stage or slip into
    the selection: both would flip routing on a real checkpoint while every
    shape stays valid.
    """
    torch.manual_seed(0)
    logits = torch.randn(64, num_experts, device="cuda", dtype=torch.float32)
    bias_t = torch.randn(num_experts, device="cuda", dtype=torch.float32)
    kwargs = {
        "top_k": min(8, num_experts // groups * min(groups, 4)),
        "renormalize": True,
        "num_expert_group": groups,
        "topk_group": min(groups, 4),
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 2.5,
        "e_score_correction_bias": bias_t,
    }
    weights, ids = grouped_topk(logits, **kwargs)
    ref_weights, ref_ids = grouped_topk_torch(logits, **kwargs)
    _assert_same_routing(weights, ids, ref_weights, ref_ids, f"{num_experts}-experts")


# --------------------------------------------------------------------------- #
# The rules the fusion must not lose, each pinned by an example
# --------------------------------------------------------------------------- #
def test_kernel_bias_selects_experts_but_original_scores_weight_them():
    """A heavy bias must route to its expert while leaving the weight tiny."""
    logits = torch.tensor([[-5.0, -4.0, +5.0, +4.0]], device="cuda")
    bias = torch.tensor([10.0, 0.0, 0.0, 0.0], device="cuda")
    weights, ids = grouped_topk(
        logits,
        top_k=1,
        renormalize=False,
        num_expert_group=2,
        topk_group=1,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
    )
    assert ids.tolist() == [[0]], "the bias must decide which expert runs"
    assert weights[0, 0].item() == pytest.approx(torch.sigmoid(logits[0, 0]).item()), (
        "the kernel must weigh with the *original* score, not the biased one"
    )


def test_kernel_saturated_ties_keep_the_renormalise_and_scale_rules():
    """Saturated sigmoid scores tie at exactly 1.0 — the rules must survive.

    The winners themselves are the tie-break's business (the kernel takes the
    lowest index, torch's CUDA top-k is unspecified); what holds regardless:
    the weights are whoever won's *original* scores, renormalised, then
    scaled. A bias leaking into them would read 1.0 + bias.
    """
    torch.manual_seed(0)
    logits = torch.where(torch.arange(16) % 2 == 0, 30.0, -30.0).repeat(8, 1).cuda()
    bias = torch.randn(16, device="cuda")
    weights, ids = grouped_topk(
        logits,
        top_k=4,
        renormalize=True,
        num_expert_group=4,
        topk_group=2,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        e_score_correction_bias=bias,
    )
    assert torch.sigmoid(logits)[:, 0::2].eq(1.0).all(), "the premise: exact 1.0 ties"
    unrenormed = torch.sigmoid(logits).gather(1, ids)
    expected = unrenormed / unrenormed.sum(dim=-1, keepdim=True) * 2.5
    assert torch.allclose(weights, expected, rtol=1e-6)


def test_kernel_handles_an_empty_token_batch():
    """A zero-token route must return empty tensors, not launch a zero grid."""
    logits = torch.empty(0, 16, device="cuda", dtype=torch.float32)
    weights, ids = grouped_topk(
        logits,
        top_k=2,
        renormalize=True,
        num_expert_group=2,
        topk_group=1,
        scoring_func="sigmoid",
    )
    assert weights.shape == (0, 2) and ids.shape == (0, 2)


# --------------------------------------------------------------------------- #
# The dispatch contract: where the kernel must not run
# --------------------------------------------------------------------------- #
def test_cpu_inputs_take_the_torch_path():
    """CPU tensors route through the reference — the kernel is CUDA-only."""
    torch.manual_seed(0)
    logits = torch.randn(8, 16, dtype=torch.float32)
    bias = torch.randn(16)
    kwargs = {
        "top_k": 4,
        "renormalize": True,
        "num_expert_group": 4,
        "topk_group": 2,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 2.5,
        "e_score_correction_bias": bias,
    }
    out = grouped_topk(logits, **kwargs)
    ref = grouped_topk_torch(logits, **kwargs)
    assert torch.equal(out[0], ref[0])
    assert torch.equal(out[1], ref[1])


def test_one_expert_groups_with_bias_degenerate_to_the_reference():
    """A bias with one-expert groups has no top-2 group score anywhere — the
    kernel declines and the reference degenerates the group score to the
    group's single biased expert (the top-2 sum's limit). The wrapper must
    hand it the call so both paths agree; the upstream references
    (transformers, vLLM) crash on this geometry instead."""
    torch.manual_seed(0)
    logits = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    bias = torch.randn(8, device="cuda")
    kwargs = {
        "top_k": 1,
        "renormalize": False,
        "num_expert_group": 8,
        "topk_group": 1,
        "scoring_func": "sigmoid",
        "e_score_correction_bias": bias,
    }
    out = grouped_topk(logits, **kwargs)
    ref = grouped_topk_torch(logits, **kwargs)
    assert torch.equal(out[0], ref[0])
    assert torch.equal(out[1], ref[1])
    # Semantics: the winning group is the one whose single biased expert
    # scores highest, and that expert is exactly what gets selected.
    scores = torch.sigmoid(logits) + bias
    assert torch.equal(out[1].squeeze(-1), scores.argmax(dim=-1))


def test_validation_matches_the_reference_contract():
    """The wrapper validates before dispatching, so both paths raise alike."""
    with pytest.raises(ValueError, match="scoring_func"):
        grouped_topk(
            torch.randn(1, 8, device="cuda"),
            top_k=2,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
            scoring_func="relu",
        )
    with pytest.raises(ValueError, match="divide"):
        grouped_topk(
            torch.randn(1, 10, device="cuda"),
            top_k=2,
            renormalize=False,
            num_expert_group=3,
            topk_group=1,
            scoring_func="sigmoid",
        )
