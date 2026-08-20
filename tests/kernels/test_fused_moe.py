"""Tests for the fused MoE grouped-GEMM kernels against a pure-torch reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from lite_llama.kernels.fused_moe import fused_moe, moe_align_block_size


def _torch_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-expert loop in fp32: the semantics the grouped GEMM must reproduce."""
    x = hidden_states.float()
    inter = w1.shape[1] // 2
    out = torch.zeros_like(x)
    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1).float()
    token_of_slot = torch.arange(x.shape[0], device=x.device).repeat_interleave(
        topk_ids.shape[1]
    )
    for e in flat_ids.unique():
        sel = flat_ids == e
        rows = token_of_slot[sel]
        gate_up = x[rows] @ w1[e].float().T
        h = F.silu(gate_up[:, :inter]) * gate_up[:, inter:]
        out.index_add_(0, rows, (h @ w2[e].float().T) * flat_weights[sel, None])
    return out


# --------------------------------------------------------------------------- #
# moe_align_block_size
# --------------------------------------------------------------------------- #
def test_align_sorts_by_expert_and_pads():
    topk_ids = torch.tensor([[2, 0], [1, 2]], device="cuda", dtype=torch.int32)
    block_size = 4
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, block_size, 3)
    num_post = int(num_post.item())

    # Expert 0: 1 slot (id 1), expert 1: 1 slot (id 2), expert 2: 2 slots (ids 0, 3);
    # each run padded up to a multiple of 4.
    assert num_post == 3 * block_size
    valid = sorted_ids[:num_post].tolist()
    assert [v for v in valid if v != topk_ids.numel()] == [1, 2, 0, 3]
    # One block per expert here; ids are ordered by expert.
    assert expert_ids[: num_post // block_size].tolist() == [0, 1, 2]
    # Padding slots carry the sentinel (== num_slots) so the kernel masks them.
    assert (sorted_ids[1:4] == topk_ids.numel()).all()


def test_align_no_padding_waste_when_full():
    # 8 slots all routed to one expert, block 4 -> exactly 2 blocks, no sentinel.
    topk_ids = torch.full((4, 2), 5, device="cuda", dtype=torch.int32)
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, 4, 8)
    assert int(num_post.item()) == 8
    assert sorted_ids[:8].max() < 8  # every slot is real
    assert expert_ids[:2].tolist() == [5, 5]


# --------------------------------------------------------------------------- #
# fused_moe vs reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_tokens", [1, 3, 37, 128])
@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (128, 8)])
def test_fused_moe_matches_reference(num_tokens, num_experts, top_k):
    hidden, inter = 256, 128
    dtype = torch.float16
    hidden_states = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / hidden**0.5
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda", dtype=dtype) / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / inter**0.5
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device="cuda", dtype=torch.float32), dim=-1
    ).to(dtype)

    out = fused_moe(hidden_states, w1, w2, topk_weights, topk_ids)
    ref = _torch_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids)

    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_fused_moe_routing_weight_folded():
    """Zeroing a routing weight must zero that slot's contribution."""
    hidden, inter, num_experts, top_k = 128, 64, 4, 2
    dtype = torch.float16
    x = torch.randn(5, hidden, device="cuda", dtype=dtype)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda", dtype=dtype) / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / inter**0.5
    ids = torch.randint(0, num_experts, (5, top_k), device="cuda")

    weights = torch.rand(5, top_k, device="cuda", dtype=dtype)
    weights[:, 1] = 0  # second slot contributes nothing
    out = fused_moe(x, w1, w2, weights, ids)

    ref = _torch_moe_reference(x, w1, w2, weights, ids)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
