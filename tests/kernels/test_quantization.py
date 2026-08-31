"""Tests for the quantisation kernels: w8a16, w4a16, smoothquant.

Each kernel is tested against a pure-torch reference that dequantises the weight
explicitly and runs the matmul in fp32. The tolerance is loose enough to absorb
the rounding noise of 8-bit (or 4-bit) storage but tight enough to catch a
swapped scale or a mis-addressed tile.
"""

from __future__ import annotations

import pytest
import torch

from lite_llama.kernels.ops.quantization import smoothquant_matmul, w4a16_matmul, w8a16_matmul
from lite_llama.modules.quantization.utils import (
    quantize_int4_groupwise,
    quantize_int8_per_channel,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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
