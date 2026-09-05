"""Quantized CPU MoE execution, including unequal projection widths."""

import pytest
import torch

from rapid_llm import kernels
from rapid_llm.modules.quantization.utils import quantize_fp8_per_channel, quantize_int8_per_channel


@pytest.mark.parametrize("mode", ["fp8", "int8"])
def test_w8a8_experts(mode):
    torch.manual_seed(23)
    quantize = quantize_fp8_per_channel if mode == "fp8" else quantize_int8_per_channel
    w1, s1 = quantize(torch.randn(2, 512, 128) * 0.01)
    w2, s2 = quantize(torch.randn(2, 128, 256) * 0.01)
    x = torch.randn(3, 128).to(torch.bfloat16)
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = torch.tensor([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]])
    operation = getattr(kernels, f"fused_moe_w8a8_{mode}")
    result = operation(x, w1, w2, weights, ids, w1_scale=s1, w2_scale=s2, group_n=1, group_k=128)
    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert torch.isfinite(result).all()
    assert result.abs().sum() > 0
    # Routing order must not change which expert's weight is applied.
    reordered = operation(
        x, w1, w2, weights.flip(1), ids.flip(1), w1_scale=s1, w2_scale=s2, group_n=1, group_k=128
    )
    torch.testing.assert_close(result, reordered)


def test_mxfp4_experts():
    from rapid_llm.modules.quantization.mxfp4 import dequant_mxfp4, repack_mxfp4_pairs

    torch.manual_seed(24)
    w1 = torch.randint(0, 256, (2, 128, 32), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (2, 64, 32), dtype=torch.uint8)
    s1, s2 = torch.full((2, 128, 2), 0.01), torch.full((2, 64, 2), 0.01)
    x = torch.randn(3, 64).to(torch.bfloat16)
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = torch.full((3, 2), 0.5)
    expected = kernels.fused_moe(x, dequant_mxfp4(w1, s1), dequant_mxfp4(w2, s2), weights, ids)
    actual = kernels.fused_moe(
        x,
        repack_mxfp4_pairs(w1),
        repack_mxfp4_pairs(w2),
        weights,
        ids,
        w1_scale=s1,
        w2_scale=s2,
        group_n=1,
        group_k=32,
        mxfp4=True,
    )
    torch.testing.assert_close(actual, expected)
