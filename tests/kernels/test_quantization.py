"""Tests for the quantisation kernels: fp8 W8A8, w8a16, w4a16, smoothquant, nvfp4.

Each GEMM is diffed against a dequantised fp reference across shapes
and group sizes; config parsing tests pin scheme strings to formats.

Usage:
    pytest tests/kernels/test_quantization.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lite_llama.kernels.ops.quantization import (
    COLUMN_MAJOR_TMA,
    NVFP4_BLOCK,
    ROW_MAJOR,
    create_scale_output,
    fp8_matmul,
    fp8_quantize_per_token,
    infer_scale_layout,
    nvfp4_matmul,
    per_token_group_quant,
    quantize_nvfp4_blockwise,
    smoothquant_matmul,
    unpack_int8_experts,
    w4a16_matmul,
    w8a16_matmul,
)
from lite_llama.modules.quantization.utils import (
    gptq_adapt_key,
    quantize_fp8_per_token,
    quantize_int4_groupwise,
    quantize_int8_groupwise_asym,
    quantize_int8_per_channel,
)
from tests.reference import nvfp4_dequant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

#: fp8-e4m3 keeps 3 mantissa bits, and W8A8 rounds *both* operands, so the
#: product carries roughly two roundings' worth of error. Same value the fp8
#: rows of ``benchmarks/kernels/bench_quant_gemm.py`` gate on.
_FP8_RTOL, _FP8_ATOL = 5e-2, 5e-2

#: Share of elements on which the fused quantiser may disagree with the torch
#: helper. Only exact ties can disagree (a quotient landing halfway between two
#: e4m3 codes, which the hardware cvt and torch's software cast break in opposite
#: directions); measured at ~3e-5 of elements, so this leaves two orders of
#: margin while still failing a real rounding bug, which would move far more.
_FP8_TIE_FRACTION = 1e-3


# --------------------------------------------------------------------------- #
# Per-token activation quantisation: fused Triton kernel vs the torch helper
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(1, 256), (8, 2560), (512, 9728), (3, 17), (2, 5, 320)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_quantize_per_token_matches_torch_helper(shape, dtype):
    """The fused quantiser must be a drop-in for the torch chain it replaced.

    ``linear_w8a8_fp8`` and the fp8 MoE path switched to the Triton kernel purely
    for launch overhead, so the numerics have to be the *same* numerics, not
    merely close ones: scales exactly equal, and bytes equal except where an
    exact e4m3 tie is broken the other way.
    """
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype) * 3.0

    qx, scale = fp8_quantize_per_token(x)
    ref_qx, ref_scale = quantize_fp8_per_token(x)

    assert qx.shape == x.shape and qx.dtype == torch.uint8
    assert scale.shape == (*x.shape[:-1], 1) and scale.dtype == torch.float32
    # amax/448 is the same arithmetic on both sides, so this one is exact; a
    # tolerance here would hide a reduction that missed part of the row.
    assert torch.equal(scale, ref_scale)

    differing = qx != ref_qx
    assert differing.float().mean().item() <= _FP8_TIE_FRACTION
    # A tie can only move the code by one. Anything larger is a scale or
    # addressing bug wearing a rounding disguise.
    delta = (qx.to(torch.int16) - ref_qx.to(torch.int16)).abs()
    assert int(delta.max().item()) <= 1


def test_fp8_quantize_per_token_handles_zero_rows():
    """An all-zero row keeps scale 1.0 rather than dividing by its own amax."""
    x = torch.zeros(4, 128, device="cuda", dtype=torch.bfloat16)
    x[1] = 1.0

    qx, scale = fp8_quantize_per_token(x)

    assert scale[0].item() == 1.0 and scale[2].item() == 1.0
    assert not qx[0].any() and not qx[2].any()
    torch.testing.assert_close(qx[1].view(torch.float8_e4m3fn).float() * scale[1], x[1].float())


def test_fp8_quantize_per_token_rounds_to_nearest():
    """Round-trip error stays within half an e4m3 step, i.e. no truncation.

    e4m3 has 3 mantissa bits, so consecutive codes are 1/8 apart in relative
    terms and round-to-nearest cannot be off by more than 1/16. Truncation would
    double that and bias every element the same way, which this catches.
    """
    torch.manual_seed(0)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16) * 2.0

    qx, scale = fp8_quantize_per_token(x)
    deq = qx.view(torch.float8_e4m3fn).float() * scale

    ref = x.float()
    # atol covers the e4m3 subnormals near zero, where the step is absolute
    # rather than relative.
    torch.testing.assert_close(deq, ref, rtol=1.0 / 16, atol=scale.max().item() * 2)
    assert deq.abs().sum() > 0  # a kernel that stored zeros would pass rtol


# --------------------------------------------------------------------------- #
# Per-token-group activation quantisation: the A-side of block-wise W8A8
# (sglang's per_token_group_quant), vs a torch reference
# --------------------------------------------------------------------------- #
def _ref_per_token_group_quant(
    x: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
    fuse_silu_and_mul: bool = False,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch spelling of ``per_token_group_quant`` for the byte-level tests."""
    k = x.shape[-1]
    h = k // 2 if fuse_silu_and_mul else k
    val = x.reshape(-1, k).float()
    if fuse_silu_and_mul:
        gate, up = val[:, :h], val[:, h:]
        val = gate * torch.sigmoid(gate) * up
    t, g = val.shape[0], h // group_size
    grouped = val.reshape(t, g, group_size)
    qmax = 448.0 if out_dtype is torch.uint8 else 127.0
    scale = grouped.abs().amax(dim=-1).clamp_min(eps) / qmax
    q = (grouped / scale[:, :, None]).clamp(-qmax, qmax)
    if out_dtype is torch.uint8:
        q = q.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        # torch.round is round-half-even, the same rule as the kernel's rint.
        q = q.round().to(torch.int8)
    return q.reshape(*x.shape[:-1], h), scale.reshape(*x.shape[:-1], g)


#: (shape, group_size) pairs exercising single-token rows, several groups per
#: program, a non-divisible group count (2560/128 = 20, GROUPS_PER_PROG=8) and
#: batched leading dims.
_GROUP_CASES = [
    ((1, 128), 32),
    ((1, 128), 128),
    ((8, 512), 32),
    ((8, 512), 256),
    ((512, 7168), 128),
    ((2, 5, 2560), 128),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("shape", "group_size"), _GROUP_CASES)
def test_per_token_group_quant_int8_matches_reference(shape, group_size, dtype):
    """Scales are exact; int8 bytes may differ only on round-boundary ties.

    Both sides load the same bf16 value into fp32 and divide by the same
    ``max(amax, eps)/127`` scale, so the scales are exact. The bytes can still
    disagree where a quotient lands on a ``.5`` boundary: a 1 ULP difference
    between the kernel's and torch's fp32 division flips the half-to-even
    rint, moving the byte by exactly one — the same tie story as the fp8
    quantiser, measured at ~4e-4 of elements on a 512x7168 row.
    """
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype) * 3.0

    q, s = per_token_group_quant(x, group_size, out_dtype=torch.int8)
    ref_q, ref_s = _ref_per_token_group_quant(x, group_size, torch.int8)

    assert q.shape == x.shape and q.dtype == torch.int8
    assert s.shape == (*x.shape[:-1], x.shape[-1] // group_size)
    assert torch.equal(s, ref_s)
    differing = q != ref_q
    assert differing.float().mean().item() <= _FP8_TIE_FRACTION
    # A tie can only move the code by one. Anything larger is a scale or
    # addressing bug wearing a rounding disguise.
    delta = (q.to(torch.int16) - ref_q.to(torch.int16)).abs()
    assert int(delta.max().item()) <= 1


@pytest.mark.parametrize(("shape", "group_size"), _GROUP_CASES)
def test_per_token_group_quant_fp8_matches_reference(shape, group_size):
    """Scales are exact; fp8 bytes may differ only on e4m3 ties.

    Same contract as the per-token fp8 quantiser: the hardware cvt breaks
    exact ties the other way from torch's software cast, so a bounded tie
    fraction (with single-code moves) is the correct notion of agreement.
    """
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 3.0

    q, s = per_token_group_quant(x, group_size, out_dtype=torch.uint8)
    ref_q, ref_s = _ref_per_token_group_quant(x, group_size, torch.uint8)

    assert q.shape == x.shape and q.dtype == torch.uint8
    assert torch.equal(s, ref_s)
    differing = q != ref_q
    assert differing.float().mean().item() <= _FP8_TIE_FRACTION
    delta = (q.to(torch.int16) - ref_q.to(torch.int16)).abs()
    assert int(delta.max().item()) <= 1


@pytest.mark.parametrize("out_dtype", [torch.int8, torch.uint8])
def test_per_token_group_quant_round_trips_within_one_step(out_dtype):
    """Dequantised groups stay within one quantisation step of the input."""
    torch.manual_seed(0)
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16) * 3.0

    q, s = per_token_group_quant(x, 128, out_dtype=out_dtype)
    deq = q.view(torch.float8_e4m3fn).float() if out_dtype is torch.uint8 else q.float()
    widened = s.repeat_interleave(128, dim=-1)
    if out_dtype is torch.uint8:
        # e4m3 keeps 3 mantissa bits: relative error bounded by half a step.
        torch.testing.assert_close(deq * widened, x.float(), rtol=1.0 / 16, atol=s.max().item() * 2)
    else:
        # int8 quantises evenly: every element lands within half a step.
        torch.testing.assert_close(deq * widened, x.float(), rtol=0.0, atol=0.51)
    assert deq.abs().sum() > 0  # a kernel that stored zeros would pass rtol


def test_per_token_group_quant_column_major_scales():
    """The transposed scale view carries the same numbers, stride (1, T)."""
    torch.manual_seed(0)
    # T=37: no divisor of 4 anywhere, so a layout bug cannot cancel out.
    x = torch.randn(37, 2560, device="cuda", dtype=torch.bfloat16)

    q_rm, s_rm = per_token_group_quant(x, 128)
    q_cm, s_cm = per_token_group_quant(x, 128, column_major_scales=True)

    assert s_cm.shape == s_rm.shape == (37, 20)
    assert s_cm.stride() == (1, 37)
    assert torch.equal(s_cm, s_rm)
    assert torch.equal(q_cm, q_rm)


def test_per_token_group_quant_tma_aligned_scales():
    """``layout=COLUMN_MAJOR_TMA`` pads the token stride to a 4-word multiple."""
    torch.manual_seed(0)
    # T=37 pads to 40; T=38 is already a multiple of 4 and needs no pad.
    x = torch.randn(37, 2560, device="cuda", dtype=torch.bfloat16)

    q_tma, s_tma = per_token_group_quant(x, 128, layout=COLUMN_MAJOR_TMA)
    q_rm, s_rm = per_token_group_quant(x, 128)

    assert s_tma.shape == (37, 20) and s_tma.stride() == (1, 40)
    assert torch.equal(s_tma, s_rm) and torch.equal(q_tma, q_rm)
    assert infer_scale_layout(s_tma).tma_aligned

    for rows in (1, 3, 4, 37, 38):
        xr = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
        _, sr = per_token_group_quant(xr, 128, layout=COLUMN_MAJOR_TMA)
        assert sr.stride() == (1, (rows + 3) // 4 * 4), rows


def test_per_token_group_quant_caller_owned_buffers():
    """Caller-owned buffers are filled in place; their strides set the layout."""
    torch.manual_seed(0)
    x = torch.randn(37, 2560, device="cuda", dtype=torch.bfloat16)
    q_ref, s_ref = per_token_group_quant(x, 128)

    # A column-major slab allocated elsewhere (another framework, a captured
    # graph pool): its strides speak for it, no layout flag needed.
    out_q = torch.empty(37, 2560, device="cuda", dtype=torch.int8)
    out_s = create_scale_output(x.shape, x.device, 128, COLUMN_MAJOR_TMA)
    q, s = per_token_group_quant(x, 128, output_q=out_q, output_s=out_s)
    assert q.data_ptr() == out_q.data_ptr() and s.data_ptr() == out_s.data_ptr()
    assert s.stride() == (1, 40)
    assert torch.equal(q, q_ref) and torch.equal(s, s_ref)

    # A row-major scale buffer alone routes the whole call through its layout.
    row_s = create_scale_output(x.shape, x.device, 128)
    _, s2 = per_token_group_quant(x, 128, output_s=row_s)
    assert s2.data_ptr() == row_s.data_ptr() and s2.stride() == (20, 1)
    assert torch.equal(s2, s_ref)


def test_per_token_group_quant_layout_contradictions_raise():
    """An ambiguous layout request fails loudly instead of guessing."""
    torch.manual_seed(0)
    x = torch.randn(37, 2560, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="contradicts"):
        per_token_group_quant(x, 128, layout=ROW_MAJOR, column_major_scales=True)
    # A caller-owned buffer contradicts the request spelled next to it.
    with pytest.raises(ValueError, match="row-major"):
        per_token_group_quant(
            x, 128, output_s=torch.empty(37, 20, device="cuda"), column_major_scales=True
        )
    with pytest.raises(ValueError, match="output_s strides describe"):
        per_token_group_quant(
            x, 128, output_s=torch.empty(37, 20, device="cuda"), layout=COLUMN_MAJOR_TMA
        )


@pytest.mark.parametrize("out_dtype", [torch.int8, torch.uint8])
def test_per_token_group_quant_fuses_silu_and_mul(out_dtype):
    """The fused gate/up quantiser matches the eager silu-mul then quantise.

    Compared through the dequantised values rather than the bytes: the kernel's
    ``tl.sigmoid`` and torch's differ by ULPs, which can flip a round boundary —
    but a flipped boundary moves the value by one step, not beyond tolerance.
    """
    torch.manual_seed(0)
    t, h = 64, 2048
    gate_up = torch.randn(t, 2 * h, device="cuda", dtype=torch.bfloat16)

    q, s = per_token_group_quant(gate_up, 128, out_dtype=out_dtype, fuse_silu_and_mul=True)

    assert q.shape == (t, h) and q.dtype == out_dtype
    assert s.shape == (t, h // 128) and s.dtype == torch.float32
    gate, up = gate_up[:, :h].float(), gate_up[:, h:].float()
    ref = gate * torch.sigmoid(gate) * up
    deq = q.view(torch.float8_e4m3fn).float() if out_dtype is torch.uint8 else q.float()
    widened = s.repeat_interleave(128, dim=-1)
    if out_dtype is torch.uint8:
        torch.testing.assert_close(deq * widened, ref, rtol=1.0 / 16, atol=s.max().item() * 2)
    else:
        torch.testing.assert_close(deq * widened, ref, rtol=0.0, atol=0.51)


def test_per_token_group_quant_zero_group_keeps_finite_scale():
    """An all-zero group divides by eps/QMAX, not zero, and stays all-zero."""
    x = torch.zeros(4, 512, device="cuda", dtype=torch.bfloat16)
    x[1, 256:] = 2.0

    q, s = per_token_group_quant(x, 256)

    # fp32 scales vs fp64 literals: compare through approx, not equality.
    assert s[0, 0].item() == pytest.approx(1e-10 / 127.0)
    assert not q[0].any()
    assert s[1, 1].item() == pytest.approx(2.0 / 127.0)
    assert q[1, 256:].abs().max().item() == 127


def test_per_token_group_quant_rejects_misaligned_rows():
    """Shape/layout violations fail loudly instead of quantising garbage."""
    x = torch.zeros(4, 100, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="not a multiple"):
        per_token_group_quant(x, 128)
    with pytest.raises(ValueError, match="power of two"):
        per_token_group_quant(torch.zeros(4, 128, device="cuda"), 96)
    with pytest.raises(ValueError, match="2D inputs only"):
        per_token_group_quant(torch.zeros(2, 4, 128, device="cuda"), 128, column_major_scales=True)
    with pytest.raises(ValueError, match="even row width"):
        per_token_group_quant(torch.zeros(4, 255, device="cuda"), 128, fuse_silu_and_mul=True)
    with pytest.raises(ValueError, match="int8 or uint8"):
        per_token_group_quant(torch.zeros(4, 128, device="cuda"), 128, out_dtype=torch.float16)
    with pytest.raises(ValueError, match="float tensor"):
        per_token_group_quant(torch.zeros(4, 128, device="cuda", dtype=torch.int32), 128)


# --------------------------------------------------------------------------- #
# fp8 W8A8: fp8-e4m3 weight + dynamic per-token fp8-e4m3 activation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "M,N,K", [(1, 512, 256), (8, 2048, 2048), (128, 768, 1024), (512, 768, 1024)]
)
@pytest.mark.parametrize("blockwise", [False, True])
def test_fp8_w8a8_matches_reference(M, N, K, blockwise):
    """Both operands in fp8, per-output-channel or 128x128 block weight scales.

    ``fp8_matmul`` was previously only exercised as the *reference* side of
    ``test_linear_dispatch.py``, which left the kernel itself ungated.

    M=512 is not a fourth shape for its own sake: the launcher picks ``BLOCK_M``
    from M, and Triton only emits Hopper's fp8 ``wgmma`` from ``BLOCK_M >= 64``,
    widening both e4m3 operands to an fp16 ``mma.sync`` below that. The three
    smaller shapes all land on ``BLOCK_M <= 32``, so without this one the fp8
    tensor cores -- the entire point of the scheme -- were never entered by any
    test. That instruction accumulates at reduced precision (see
    ``moe.fused_moe._FP8_A8_PROMOTE_EVERY``), and the tolerance below still holds
    with room to spare: it needs 5e-4 of atol where 5e-2 is granted.
    """
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.05

    group_n, group_k = (128, 128) if blockwise else (1, K)
    qw = w.to(torch.float8_e4m3fn).view(torch.uint8)
    w_scale = (
        torch.rand((N + group_n - 1) // group_n, (K + group_k - 1) // group_k, device="cuda") + 0.5
    )

    qx, x_scale = quantize_fp8_per_token(x)
    out = fp8_matmul(qx, x_scale, qw, w_scale, group_n=group_n, group_k=group_k)

    # Reference: widen both operands with torch and matmul in fp32, so the only
    # thing left to disagree about is the kernel's scale addressing.
    x_deq = qx.view(torch.float8_e4m3fn).float() * x_scale
    w_deq = qw.view(torch.float8_e4m3fn).float()
    s = w_scale.repeat_interleave(group_n, 0).repeat_interleave(group_k, 1)[:N, :K]
    ref = x_deq @ (w_deq * s).T

    torch.testing.assert_close(out.float(), ref, rtol=_FP8_RTOL, atol=_FP8_ATOL)


def test_fp8_w8a8_applies_bias():
    """Bias is added in fp32 before the output cast, not folded into a scale."""
    torch.manual_seed(0)
    M, N, K = 8, 256, 512
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn).view(torch.uint8)
    w_scale = torch.full((N, 1), 0.02, device="cuda")
    bias = torch.randn(N, device="cuda", dtype=torch.float32)

    qx, x_scale = quantize_fp8_per_token(x)
    no_bias = fp8_matmul(qx, x_scale, w, w_scale, group_n=1, group_k=K)
    with_bias = fp8_matmul(qx, x_scale, w, w_scale, group_n=1, group_k=K, bias=bias)

    torch.testing.assert_close(
        (with_bias - no_bias).float(),
        bias.expand(M, N),
        rtol=_FP8_RTOL,
        atol=_FP8_ATOL,
    )


def test_fp8_w8a8_rejects_non_uint8_operands():
    """An accidental fp16 operand must fail loudly, not be reinterpreted."""
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    qw = torch.zeros(64, 128, device="cuda", dtype=torch.uint8)
    scale = torch.ones(64, 1, device="cuda")
    with pytest.raises(ValueError, match="uint8 e4m3 bit patterns"):
        fp8_matmul(x, torch.ones(4, 1, device="cuda"), qw, scale, group_n=1, group_k=128)


# --------------------------------------------------------------------------- #
# w8a16: fp8 blockwise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("M,N,K", [(1, 512, 256), (8, 2048, 2048), (128, 768, 1024)])
def test_w8a16_fp8_blockwise_matches_reference(M, N, K):
    """fp8-e4m3 with 128×128 block scales."""
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.05
    qw = w.to(torch.float8_e4m3fn).view(torch.uint8)
    gn = gk = 128
    scales = torch.rand((N + gn - 1) // gn, (K + gk - 1) // gk, device="cuda") + 0.5

    out = w8a16_matmul(x, qw, scales, group_n=gn, group_k=gk)

    # Reference: dequantise and matmul in fp32.
    w_deq = qw.view(torch.float8_e4m3fn).float()
    s = scales.repeat_interleave(gn, 0).repeat_interleave(gk, 1)[:N, :K]
    ref = x.float() @ (w_deq * s).T
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------- #
# w8a16: int8 per-channel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("M,N,K", [(1, 512, 256), (8, 2048, 2048), (33, 130, 384)])
def test_w8a16_int8_per_channel_matches_reference(M, N, K):
    """Symmetric int8, one scale per output channel."""
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.05
    qw, scale = quantize_int8_per_channel(w)

    out = w8a16_matmul(x, qw, scale, group_n=1, group_k=K)

    ref = x.float() @ (qw.float() * scale).T
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------- #
# w4a16: AWQ/GPTQ int4
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("M,N,K", [(1, 256, 512), (8, 512, 1024)])
@pytest.mark.parametrize("group_size", [32, 128])
def test_w4a16_int4_groupwise_matches_reference(M, N, K, group_size):
    """Group-wise int4 with scales and zeros."""
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.05

    # Quantise to int4
    qw, scales, zeros = quantize_int4_groupwise(w, group_size)

    out = w4a16_matmul(x, qw, scales, zeros, group_size=group_size)

    # Reference: unpack and dequantise
    k_packed = K // 8
    w_unpacked = torch.zeros(N, K, device="cuda", dtype=torch.float32)
    for i in range(k_packed):
        word = qw[:, i].to(torch.int64)
        for j in range(8):
            nibble = (word >> (4 * j)) & 0xF
            w_unpacked[:, i * 8 + j] = nibble.float()

    # Apply scales and zeros
    num_groups = K // group_size
    for g in range(num_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        w_unpacked[:, k_start:k_end] = (
            w_unpacked[:, k_start:k_end] - zeros[:, g : g + 1]
        ) * scales[:, g : g + 1]

    ref = x.float() @ w_unpacked.T
    # int4 has larger quantisation error
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)


# --------------------------------------------------------------------------- #
# nvfp4: e2m1 weights with 16-element e4m3 block scales, weight-only
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "M,N,K", [(1, 512, 256), (8, 2048, 2048), (128, 768, 1024), (33, 512, 512)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_matches_reference(M, N, K, dtype):
    """The kernel's in-register decode must match an independent torch decode.

    The comparison is against ``F.linear`` on the *reconstructed* weight, not on
    the original float weight: NVFP4 loses about 9.5% relative on a Gaussian
    tensor, which is a property of 4 bits and not something a kernel test can
    or should gate on. What is under test is that
    :func:`~tests.reference.nvfp4_dequant` — table lookup plus a
    ``float8_e4m3fn`` view — and the kernel's bit-assembly plus shift trick
    agree on which numbers those bytes name.

    So the tolerance covers output-dtype rounding only. bf16 at ``|y| ~ 5``
    has a ULP of 0.03, which is why ``rtol`` rather than ``atol`` has to carry
    the larger shapes.
    """
    torch.manual_seed(0)
    w = torch.randn(N, K, device="cuda")
    packed, block_scale, global_scale = quantize_nvfp4_blockwise(w)
    assert packed.shape == (N, K // 2)
    assert block_scale.shape == (N, K // NVFP4_BLOCK)

    x = torch.randn(M, K, device="cuda", dtype=dtype) / K**0.5
    out = nvfp4_matmul(x, packed, block_scale, global_scale)

    ref = F.linear(x.float(), nvfp4_dequant(packed, block_scale, global_scale))
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def test_nvfp4_applies_bias():
    torch.manual_seed(0)
    m, n, k = 8, 512, 1024
    w = torch.randn(n, k, device="cuda")
    packed, block_scale, global_scale = quantize_nvfp4_blockwise(w)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    out = nvfp4_matmul(x, packed, block_scale, global_scale, bias=bias)
    wq = nvfp4_dequant(packed, block_scale, global_scale)
    ref = F.linear(x.float(), wq, bias.float())
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def test_nvfp4_round_trips_the_representable_grid_exactly():
    """Every e2m1 magnitude survives quantise -> dequantise unchanged.

    A value table off by one entry, or a sign bit read from the wrong position,
    still produces plausible-looking noise on random weights. On the eight
    magnitudes the format can name exactly it produces an exact mismatch, which
    is why this case exists separately from the GEMM comparison.
    """
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    # Both signs, tiled to fill whole blocks; one row per block so the per-block
    # amax is 6 and the block scale comes out at exactly 1/global_scale.
    w = torch.cat([grid, -grid]).repeat(4, 4)  # [4, 64]
    packed, block_scale, global_scale = quantize_nvfp4_blockwise(w)
    torch.testing.assert_close(nvfp4_dequant(packed, block_scale, global_scale), w)


def test_nvfp4_rejects_shapes_the_format_cannot_express():
    w = torch.randn(64, 128, device="cuda")
    packed, block_scale, global_scale = quantize_nvfp4_blockwise(w)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="multiple of 16"):
        # 24 packed bytes = 48 logical columns, three whole blocks and no more:
        # a K that is not a multiple of 16 has nowhere to put the tail scale.
        nvfp4_matmul(x[:, :40], packed[:, :20], block_scale[:, :2], global_scale)
    with pytest.raises(ValueError, match="block_scale must be"):
        nvfp4_matmul(x, packed, block_scale[:, :-1], global_scale)
    with pytest.raises(ValueError, match="fp16 or bf16"):
        nvfp4_matmul(x.float(), packed, block_scale, global_scale)
    with pytest.raises(ValueError, match="one element"):
        nvfp4_matmul(x, packed, block_scale, global_scale.repeat(2))


# --------------------------------------------------------------------------- #
# smoothquant: W8A8 dynamic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("M,N,K", [(1, 256, 512), (8, 512, 1024), (64, 2048, 2048)])
def test_smoothquant_matches_reference(M, N, K):
    """Dynamic per-token activation quantisation + per-channel weight quantisation."""
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.05
    scale = w.abs().amax(-1) / 127.0
    qw = (w / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)

    out = smoothquant_matmul(x, qw, scale)

    ref = x.float() @ (qw.float() * scale.unsqueeze(-1)).T
    # W8A8 has both weight and activation quantisation error
    torch.testing.assert_close(out.float(), ref, rtol=1e-1, atol=1e-1)


# --------------------------------------------------------------------------- #
# QuantizationConfig
# --------------------------------------------------------------------------- #
def test_quant_config_fp8():
    from lite_llama.modules.quantization.fp8 import Fp8Config

    qc = Fp8Config(group_n=128, group_k=128)
    assert qc.is_fp8
    assert qc.storage_dtype == torch.uint8
    assert qc.scale_shape(256, 512) == (2, 4)


def test_quant_config_int8():
    from lite_llama.modules.quantization.blockwise_int8 import BlockInt8Config

    qc = BlockInt8Config.per_channel()
    assert qc.get_name() == "blockwise_int8"
    assert qc.storage_dtype == torch.int8
    assert qc.scale_shape(256, 512) == (256, 1)


def test_quant_config_int4():
    from lite_llama.modules.quantization.awq import AWQConfig

    qc = AWQConfig(group_size=128)
    assert qc.is_int4
    assert qc.storage_dtype == torch.int32
    assert qc.scale_shape(256, 512) == (256, 4)


def test_quant_config_smoothquant():
    from lite_llama.modules.quantization.w8a8_int8 import W8A8Int8Config

    qc = W8A8Int8Config()
    assert qc.get_name() == "w8a8_int8"
    assert qc.is_dynamic
    assert qc.storage_dtype == torch.int8


# --------------------------------------------------------------------------- #
# w8a16: int8 asymmetric (GPTQ bits=8)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("M,N,K", [(1, 512, 256), (8, 2048, 2048), (33, 130, 384)])
def test_w8a16_int8_asymmetric_matches_reference(M, N, K):
    """GPTQ ``bits=8``: group scales + zero points, one byte per element.

    The zeros take the kernel through the ``HAS_ZEROS`` branch its symmetric
    sibling never enters: the group point is subtracted in fp32 (the difference
    of two int8 values is an exact integer) before the scale multiply.
    """
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.05
    qw, scales, zeros = quantize_int8_groupwise_asym(w, group_size=128)
    # The layout process_weights_after_loading leaves: one int8 byte per value.
    q_int8 = unpack_int8_experts(qw)

    out = w8a16_matmul(x, q_int8, scales, group_n=1, group_k=128, zeros=zeros)

    groups = q_int8.reshape(N, K // 128, 128)
    deq = ((groups - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)).reshape(N, K)
    ref = x.float() @ deq.T
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


def test_w8a16_rejects_zeros_on_fp8_weights():
    """e4m3 has no zero point; the asymmetric args must not be silently ignored."""
    x = torch.randn(4, 128, device="cuda", dtype=torch.float16)
    qw = torch.zeros(64, 128, device="cuda", dtype=torch.uint8)
    scales = torch.ones(64, 1, device="cuda")
    zeros = torch.zeros(64, 1, device="cuda")
    with pytest.raises(ValueError, match="fp8 is symmetric"):
        w8a16_matmul(x, qw, scales, group_n=1, group_k=128, zeros=zeros)


def test_w8a16_rejects_zeros_shape_mismatch():
    """Zeros ride the scales' grid; a mismatched shape is a caller bug."""
    x = torch.randn(4, 128, device="cuda", dtype=torch.float16)
    qw = torch.zeros(64, 128, device="cuda", dtype=torch.int8)
    scales = torch.ones(64, 1, device="cuda")
    zeros = torch.zeros(64, 2, device="cuda")
    with pytest.raises(ValueError, match="share the scales' shape"):
        w8a16_matmul(x, qw, scales, group_n=1, group_k=128, zeros=zeros)


# --------------------------------------------------------------------------- #
# int8 expert unpack: the bits=8 load-time preprocessing kernel
# --------------------------------------------------------------------------- #
def test_unpack_int8_experts_matches_torch_unpack():
    """``[.., K//4]`` int32 words -> ``[.., K]`` int8, bit-exact.

    Random words across the full int32 range exercise the sign path: the top
    byte of a negative word sign-extends under an arithmetic ``>>``, so the
    kernel's ``& 0xFF`` mask (and torch's here) is load-bearing.
    """
    torch.manual_seed(0)
    words = torch.randint(-(2**31), 2**31 - 1, (8, 64, 96), device="cuda", dtype=torch.int64).to(
        torch.int32
    )

    out = unpack_int8_experts(words)
    assert out.shape == (8, 64, 96 * 4)
    assert out.dtype == torch.int8

    shifts = torch.arange(0, 32, 8, device="cuda", dtype=torch.int32)
    ref = ((words.unsqueeze(-1) >> shifts) & 0xFF).flatten(-2)
    ref = ref.to(torch.uint8).view(torch.int8)
    assert torch.equal(out, ref)


# --------------------------------------------------------------------------- #
# Quantisation utilities
# --------------------------------------------------------------------------- #
def test_quantize_int8_per_channel():
    w = torch.randn(64, 128, device="cuda")
    qw, scale = quantize_int8_per_channel(w)
    assert qw.dtype == torch.int8
    assert scale.shape == (64, 1)
    # Check that the max abs value maps to ~127
    reconstructed = qw.float() * scale
    torch.testing.assert_close(reconstructed, w, rtol=2e-2, atol=2e-2)


def test_quantize_int4_groupwise():
    w = torch.randn(64, 256, device="cuda")
    qw, scales, zeros = quantize_int4_groupwise(w, group_size=128)
    assert qw.dtype == torch.int32
    assert qw.shape == (64, 32)  # 256 / 8 = 32
    assert scales.shape == (64, 2)  # 256 / 128 = 2
    assert zeros.shape == (64, 2)


def test_quantize_int8_groupwise_asym_roundtrip():
    """GPTQ ``bits=8`` quantiser: pack, unpack, dequant within half a step.

    Two shapes of failure are gated here that the GEMM tests would blur. The
    pack must be bit-faithful — a negative byte's sign bits riding into the
    byte above would pass the GEMM's rtol but fail this equality. And an
    all-negative group drives the uint8-domain zero point past 255; the fp32
    zeros container must carry it unclamped, where a clamp would collapse the
    whole group onto one code level.
    """
    torch.manual_seed(0)
    for w in (
        torch.randn(64, 256, device="cuda") * 0.05,
        # Every group entirely negative: -min/scale exceeds 255 by construction.
        torch.linspace(-1.0, -0.5, 256, device="cuda").unsqueeze(0).repeat(8, 1).contiguous(),
    ):
        qw, scales, zeros = quantize_int8_groupwise_asym(w, group_size=128)
        assert qw.dtype == torch.int32
        assert qw.shape == (w.shape[0], 256 // 4)  # four bytes per int32 word
        assert scales.shape == zeros.shape == (w.shape[0], 2)

        un = unpack_int8_experts(qw).reshape(-1, 2, 128)
        deq = ((un - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)).reshape(w.shape)
        err = (deq - w).abs().max().item()
        assert err <= scales.max().item() / 2 + 1e-6, f"roundtrip error {err} > half-step"


def test_gptq_int8_checkpoint_adapters():
    """``bits=8`` key adapter: qzeros domain-shift, qweight transpose, g_idx drop.

    AutoGPTQ packs int8 zero points four bytes per int32 word and stores
    ``z_true - 1`` (the same bias its int4 packing uses), so the adapter
    unpacks bytes, undoes the ``+1``, and shifts the uint8 domain AutoGPTQ
    quantises in into the int8 domain our kernels subtract in — the exact
    chain vLLM's ``loaded_weight.T + 1`` performs in a uint8 container.
    """
    torch.manual_seed(0)
    N, K, G = 128, 256, 128
    prefix = "model.layers.0.mlp.gate_proj"

    # qzeros: [G, N//4] int32 words holding biased zero points (z_true - 1).
    # z_true ∈ [1, 255] because the GPTQ bias stores z_true - 1 ∈ [0, 254];
    # z_true = 0 would make z_cp = -1, which the byte cannot represent.
    z_true = torch.randint(1, 256, (G, N), dtype=torch.int32)
    z_cp = z_true - 1
    shifts = torch.arange(4, dtype=torch.int32) * 8
    packed_z = ((z_cp.reshape(G, N // 4, 4) & 0xFF) << shifts).sum(-1)
    key, out = gptq_adapt_key(f"{prefix}.qzeros", packed_z, bits=8)
    assert key == f"{prefix}.weight_zeros"
    assert out.shape == (N, G) and out.dtype == torch.float32
    assert torch.equal(out, (z_true - 128).t().float())

    # qweight: [K//4, N] -> transposed canonical packing, words untouched.
    qw = torch.randint(-(2**31), 2**31 - 1, (K // 4, N), dtype=torch.int64).to(torch.int32)
    key, out = gptq_adapt_key(f"{prefix}.qweight", qw, bits=8)
    assert key == f"{prefix}.weight"
    assert torch.equal(out, qw.t().contiguous())

    # scales: [G, N] -> [N, G] fp32, the same relayout as int4.
    scales = torch.rand(G, N)
    key, out = gptq_adapt_key(f"{prefix}.scales", scales, bits=8)
    assert key == f"{prefix}.weight_scale"
    assert torch.equal(out, scales.t().float().contiguous())

    # g_idx: desc_act checkpoints only; dropped, the groups ride in K order.
    assert gptq_adapt_key(f"{prefix}.g_idx", torch.arange(K // G), bits=8) is None
