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
    NVFP4_BLOCK,
    fp8_matmul,
    fp8_quantize_per_token,
    nvfp4_matmul,
    quantize_nvfp4_blockwise,
    smoothquant_matmul,
    w4a16_matmul,
    w8a16_matmul,
)
from lite_llama.modules.quantization.utils import (
    quantize_fp8_per_token,
    quantize_int4_groupwise,
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
