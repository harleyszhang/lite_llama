"""Tests for the fused MoE grouped-GEMM kernels against a pure-torch reference.

``moe_align_block_size`` sorting and padding is checked exactly, then
the fused kernel is diffed against ``fused_moe_reference`` across
token, expert and top-k counts, including every quantised expert
format and the fp8 W8A8 mode.

Usage:
    pytest tests/kernels/test_fused_moe.py
"""

from __future__ import annotations

import pytest
import torch

from lite_llama.kernels.ops.moe.fused_moe import (
    fused_moe,
    fused_moe_w8a8_fp8,
    moe_align_block_size,
)
from lite_llama.modules.quantization.utils import (
    quantize_fp8_per_channel,
    quantize_fp8_per_token,
    quantize_int4_groupwise,
    quantize_int8_per_channel,
)
from tests.reference import fused_moe_reference

# Redundant with the automatic `gpu` mark applied to tests/kernels/ by
# tests/conftest.py, but harmless and keeps the file self-describing.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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
    ref = fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids)

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

    ref = fused_moe_reference(x, w1, w2, weights, ids)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("act_dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_fp8_blockwise_matches_reference(act_dtype):
    """fp8-e4m3 expert weights with 128x128 block scales, in either activation
    dtype.

    Regression: the dequantised operand used to be hardcoded fp16, so bf16
    activations (Qwen3-30B-A3B-Instruct-2507-FP8) failed kernel compilation
    with "Both operands must be same dtype. Got bf16 and fp16".
    """
    torch.manual_seed(0)
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    gn = gk = 128
    x = torch.randn(7, hidden, device="cuda", dtype=act_dtype) / hidden**0.5
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda", dtype=torch.float32) * 0.05
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=torch.float32) * 0.05
    qw1 = w1.to(torch.float8_e4m3fn).view(torch.uint8)
    qw2 = w2.to(torch.float8_e4m3fn).view(torch.uint8)
    s1 = (
        torch.rand(
            (num_experts, (2 * inter + gn - 1) // gn, (hidden + gk - 1) // gk), device="cuda"
        )
        + 0.5
    )
    s2 = (
        torch.rand((num_experts, (hidden + gn - 1) // gn, (inter + gk - 1) // gk), device="cuda")
        + 0.5
    )
    ids = torch.randint(0, num_experts, (7, top_k), device="cuda")
    weights = torch.softmax(torch.randn(7, top_k, device="cuda", dtype=torch.float32), dim=-1)

    out = fused_moe(x, qw1, qw2, weights, ids, w1_scale=s1, w2_scale=s2, group_n=gn, group_k=gk)

    # Reference: dequantise (the e4m3 values are exact in fp32) and matmul in fp32.
    deq1 = qw1.view(torch.float8_e4m3fn).float() * s1.repeat_interleave(gn, 1).repeat_interleave(
        gk, 2
    )
    deq2 = qw2.view(torch.float8_e4m3fn).float() * s2.repeat_interleave(gn, 1).repeat_interleave(
        gk, 2
    )
    ref = fused_moe_reference(x, deq1, deq2, weights.to(act_dtype), ids)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------- #
# Quantised expert weights
# --------------------------------------------------------------------------- #
#: int4 group size, matching what AWQ/GPTQ checkpoints ship and what
#: ``AWQConfig.group_k`` defaults to.
_INT4_GROUP = 128


def _fp8_experts(w: torch.Tensor):
    q, s = quantize_fp8_per_channel(w)
    # Widened from the *same bytes* by torch, which separates the kernel's
    # bit-trick dequant from fp8 rounding: both sides see identical values.
    return (q, s, None), q.view(torch.float8_e4m3fn).float() * s


def _int8_experts(w: torch.Tensor):
    q, s = quantize_int8_per_channel(w)
    return (q, s, None), q.float() * s


def _int4_experts(w: torch.Tensor):
    # quantize_int4_groupwise packs with a 2D reshape, so it takes [N, K] only.
    parts = [quantize_int4_groupwise(w[e], _INT4_GROUP) for e in range(w.shape[0])]
    q, s, z = (torch.stack(t) for t in zip(*parts, strict=True))
    e, n, _ = q.shape
    k = w.shape[-1]
    # `& 0xF` after the shift is load-bearing: a top nibble can set the sign bit
    # and torch's int32 `>>` is arithmetic, so it sign-extends.
    shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
    nibbles = ((q.unsqueeze(-1) >> shifts) & 0xF).reshape(e, n, k).float()
    groups = nibbles.reshape(e, n, k // _INT4_GROUP, _INT4_GROUP)
    deq = ((groups - z.unsqueeze(-1)) * s.unsqueeze(-1)).reshape(e, n, k)
    return (q, s, z), deq


_QUANT_FORMATS = {"fp8": _fp8_experts, "int8": _int8_experts, "int4": _int4_experts}


@pytest.mark.parametrize("fmt", sorted(_QUANT_FORMATS))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_quantised_matches_reference(fmt, dtype):
    """Every 8/4-bit expert format, at both activation dtypes.

    Each format runs a different branch of the kernel's inner loop -- an e4m3 bit
    trick, an int8 convert, a nibble unpack with a zero point -- so a format
    verified through a sibling is unverified. The reference multiplies the
    *dequantised* weights, which isolates the kernel's arithmetic from the
    format's own error: both sides see the same numbers.

    Both dtypes, because neither axis used to work. The quantised branches widened
    the weight tile to a hard-coded fp16, so ``tl.dot`` got two operand types on
    any bf16 model -- which is every Qwen3-MoE checkpoint -- and the layer failed
    to compile rather than to compute. int4 was worse: it folded its fp32 scale
    into the operand before the dot, so that branch could not compile at *any*
    activation dtype, and no test reached it.
    """
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    # 33 tokens is not a multiple of any BLOCK_M the launcher picks, so the padded
    # slots and their sentinel mask are exercised rather than avoided.
    tokens = 33
    torch.manual_seed(0)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    (q1, s1, z1), ref1 = _QUANT_FORMATS[fmt](w1)
    (q2, s2, z2), ref2 = _QUANT_FORMATS[fmt](w2)

    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    ids = torch.rand(tokens, num_experts, device="cuda").topk(top_k, dim=-1).indices.to(torch.int32)
    weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=-1).to(dtype)

    # Per-channel scales are one group spanning all of K, and the two GEMMs have
    # different K (hidden for gate_up, inter for down) but share one group_k, which
    # the launcher clamps with min(group_k, K) -- so the larger K covers both.
    group_k = _INT4_GROUP if fmt == "int4" else max(hidden, inter)
    out = fused_moe(
        x,
        q1,
        q2,
        weights,
        ids,
        w1_scale=s1,
        w2_scale=s2,
        w1_zeros=z1,
        w2_zeros=z2,
        group_n=1,
        group_k=group_k,
    )
    ref = fused_moe_reference(x, ref1, ref2, weights, ids)

    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_fused_moe_rejects_mixed_quant_formats():
    """A scale on one weight and not the other is a caller bug, not a fp16 path."""
    hidden, inter, num_experts, top_k = 128, 64, 4, 2
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_fp8_per_channel(w1)
    x = torch.randn(3, hidden, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, num_experts, (3, top_k), device="cuda", dtype=torch.int32)
    weights = torch.rand(3, top_k, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="same quantisation format"):
        fused_moe(x, q1, w2.half(), weights, ids, w1_scale=s1, group_n=1, group_k=hidden)


# --------------------------------------------------------------------------- #
# fp8 W8A8 experts (activation quantised too)
# --------------------------------------------------------------------------- #
#: RMS error relative to the reference's RMS, per activation dtype. This is the
#: gate rather than a max-element ``atol`` because the max grows with the number
#: of output elements (a max over more samples of the same distribution), while
#: the RMS is stable across shapes: measured 1.1e-3 / 4.6e-3 / 5.7e-3 for fp16 at
#: 1 / 33 / 129 tokens and 2.8e-2 / 1.3e-2 / 1.4e-2 for bf16.
#:
#: bf16 is ~2.5x looser because the mode stores three intermediates in the
#: activation dtype -- the silu output, each slot's GEMM2 row, and the sum -- and
#: bf16 keeps 8 mantissa bits where fp16 keeps 11.
_A8_RMS_REL = {torch.float16: 1.5e-2, torch.bfloat16: 4.0e-2}

#: Largest single element error, as a fraction of the reference's peak. Measured
#: 1.4e-2 (fp16) and 2.7e-2 (bf16); this bounds it rather than tracking it.
_A8_MAX_OVER_PEAK = 5.0e-2

#: How far quantising the activation moves the answer, RMS relative, versus the
#: same weights with a full-precision activation. Measured 4.4e-2 -- an order of
#: magnitude above ``_A8_RMS_REL`` because it is a different quantity: that one is
#: the kernel's error against its own inputs, this one is the *cost of the mode*.
#: e4m3 carries 3 mantissa bits, so each element is good to about 3%, and a dot of
#: K random-sign terms keeps that 3% rather than averaging it down (the error grows
#: like sqrt(K), and so does the sum's own magnitude). Two such GEMMs in series
#: give sqrt(2) x 3% ~ 4e-2, which is what shows up.
_A8_VS_A16_RMS_REL = 6.0e-2


def _fp8_round_trip(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Per-token e4m3 quantise-dequantise, in torch, at the kernel's precision.

    Downcast to ``dtype`` first because the kernel quantises what it stored: the
    activation as it arrived, and the silu output after it was written back in the
    activation dtype. Quantising an fp32 intermediate would compare the kernel
    against a pipeline it does not run.

    Uses the torch quantiser (``modules.quantization.utils``), not the Triton one
    the kernel calls;
    ``test_fp8_quantize_per_token_matches_torch_helper`` gates that the two agree.
    """
    q, scale = quantize_fp8_per_token(t.to(dtype))
    return q.view(torch.float8_e4m3fn).float() * scale


@pytest.mark.parametrize("tokens", [1, 33, 129])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_fp8_a8_matches_reference(tokens, dtype):
    """``fused_moe_w8a8_fp8`` against a torch reference that quantises both sides.

    The reference applies the same per-token e4m3 round trip to the input and to
    the silu output, so what is being checked is the kernel's arithmetic, not
    e4m3's dynamic range. The weight-only rows above cannot stand in for this one:
    mode 4 compiles a different inner loop, where neither operand is widened and
    the activation carries its own row scale.

    The token counts pick the launcher's three ``BLOCK_M`` tiers (16 / 32 / 64),
    which matters more here than for any other format: Triton emits Hopper's fp8
    ``wgmma`` only from ``BLOCK_M >= 64`` and widens both e4m3 operands to an fp16
    ``mma.sync`` below it, so 129 tokens is the only case that reaches the fp8
    tensor cores at all. None of 1, 33 or 129 is a multiple of its tile, so the
    padded slots and their sentinel mask are exercised rather than avoided.
    """
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    torch.manual_seed(0)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_fp8_per_channel(w1)
    q2, s2 = quantize_fp8_per_channel(w2)
    # Widened from the same bytes, so the reference's weights are the kernel's.
    d1 = q1.view(torch.float8_e4m3fn).float() * s1
    d2 = q2.view(torch.float8_e4m3fn).float() * s2

    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    ids = torch.rand(tokens, num_experts, device="cuda").topk(top_k, dim=-1).indices
    ids = ids.to(torch.int32)
    weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=-1).to(dtype)

    out = fused_moe_w8a8_fp8(
        x,
        q1,
        q2,
        weights,
        ids,
        w1_scale=s1,
        w2_scale=s2,
        group_n=1,
        group_k=max(hidden, inter),
    )
    ref = fused_moe_reference(
        x, d1, d2, weights, ids, act_quant=lambda t: _fp8_round_trip(t, dtype)
    )

    assert out.dtype == dtype
    err = (out.float() - ref).abs()
    rms_rel = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert rms_rel < _A8_RMS_REL[dtype], f"rms relative error {rms_rel:.3e}"
    peak_rel = (err.max() / ref.abs().max()).item()
    assert peak_rel < _A8_MAX_OVER_PEAK, f"worst element {peak_rel:.3e} of peak"


def test_fused_moe_fp8_a8_differs_from_weight_only():
    """The two fp8 entry points are not aliases of one another.

    They were, before this mode existed: ``W8A8Fp8MoEMethod.apply`` and
    ``Fp8MoEMethod.apply`` ran the same weight-only kernel, so the W8A8 scheme was
    a name with no behaviour behind it. Quantising the activation has to move the
    numbers, or the mode is decoration.
    """
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    torch.manual_seed(0)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_fp8_per_channel(w1)
    q2, s2 = quantize_fp8_per_channel(w2)
    x = torch.randn(64, hidden, device="cuda", dtype=torch.float16)
    ids = torch.rand(64, num_experts, device="cuda").topk(top_k, dim=-1).indices.to(torch.int32)
    weights = torch.softmax(torch.randn(64, top_k, device="cuda"), dim=-1).to(torch.float16)

    kw = {"w1_scale": s1, "w2_scale": s2, "group_n": 1, "group_k": max(hidden, inter)}
    a16 = fused_moe(x, q1, q2, weights, ids, **kw)
    a8 = fused_moe_w8a8_fp8(x, q1, q2, weights, ids, **kw)

    assert not torch.equal(a16, a8)
    # Same operation either way, so the gap must stay at the size of one extra
    # e4m3 rounding per operand and not grow into a different answer.
    rel = (
        (a8.float() - a16.float()).pow(2).mean().sqrt() / a16.float().pow(2).mean().sqrt()
    ).item()
    assert rel < _A8_VS_A16_RMS_REL, f"the two fp8 paths diverge by {rel:.3e}"


def test_fused_moe_fp8_a8_rejects_non_fp8_experts():
    """Mode 4 cannot be inferred from a dtype, so a wrong caller must be told."""
    hidden, inter, num_experts, top_k = 128, 64, 4, 2
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_int8_per_channel(w1)
    q2, s2 = quantize_int8_per_channel(w2)
    x = torch.randn(3, hidden, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, num_experts, (3, top_k), device="cuda", dtype=torch.int32)
    weights = torch.rand(3, top_k, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="uint8 e4m3"):
        fused_moe_w8a8_fp8(
            x, q1, q2, weights, ids, w1_scale=s1, w2_scale=s2, group_n=1, group_k=hidden
        )
