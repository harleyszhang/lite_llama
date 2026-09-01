"""GPU tests: the ``linear`` scheme entry points forward to the right kernel.

Each adapter (w8a16 fp8/int8, w4a16, w8a8) is diffed against its Triton
kernel called directly — the adapter layer adds dispatch, not maths.

Usage:
    pytest tests/kernels/test_linear_dispatch.py
"""

from __future__ import annotations

import pytest
import torch

from lite_llama.kernels.ops.gemm.linear import (
    linear_w4a16,
    linear_w8a8_fp8,
    linear_w8a8_int8,
    linear_w8a16,
)
from lite_llama.kernels.ops.quantization import (
    fp8_matmul,
    smoothquant_matmul,
    w4a16_matmul,
    w8a16_matmul,
)
from lite_llama.modules.quantization.base_config import run_quant_linear
from lite_llama.modules.quantization.utils import (
    quantize_fp8_per_token,
    quantize_int4_groupwise,
    quantize_int8_per_channel,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

M, N, K = 8, 256, 512


def _x(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(M, K, device="cuda", dtype=dtype) * 0.5


def test_linear_w8a16_adapter_matches_kernel_fp8() -> None:
    x = _x()
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.rand(2, 4, device="cuda") + 0.5

    out = linear_w8a16(x, w, weight_scale=scales, group_n=128, group_k=128)
    ref = w8a16_matmul(x, w, scales, group_n=128, group_k=128)
    assert torch.equal(out, ref)


def test_linear_w8a16_adapter_matches_kernel_int8() -> None:
    x = _x()
    qw, scale = quantize_int8_per_channel(torch.randn(N, K, device="cuda") * 0.05)

    out = linear_w8a16(x, qw, weight_scale=scale, group_n=1, group_k=K)
    ref = w8a16_matmul(x, qw, scale, group_n=1, group_k=K)
    assert torch.equal(out, ref)


def test_linear_w4a16_adapter_matches_kernel() -> None:
    x = _x()
    qw, scales, zeros = quantize_int4_groupwise(torch.randn(N, K, device="cuda") * 0.05, 128)

    out = linear_w4a16(x, qw, weight_scale=scales, weight_zeros=zeros, group_k=128)
    ref = w4a16_matmul(x, qw, scales, zeros, group_size=128)
    assert torch.equal(out, ref)


def test_linear_w8a8_int8_adapter_matches_kernel() -> None:
    x = _x()
    qw, scale = quantize_int8_per_channel(torch.randn(N, K, device="cuda") * 0.05)

    out = linear_w8a8_int8(x, qw, weight_scale=scale)
    ref = smoothquant_matmul(x, qw, scale)
    assert torch.equal(out, ref)


def test_linear_w8a8_fp8_adapter_matches_manual_chain() -> None:
    x = _x()
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.rand(2, 4, device="cuda") + 0.5

    out = linear_w8a8_fp8(x, w, weight_scale=scales, group_n=128, group_k=128)

    qx, x_scale = quantize_fp8_per_token(x)  # what the old apply did by hand
    ref = fp8_matmul(qx, x_scale, w, scales, group_n=128, group_k=128, out_dtype=x.dtype)
    assert torch.equal(out, ref)


def test_run_quant_linear_selects_and_runs_the_native_row() -> None:
    """The quant methods' only call site: dispatch -> load -> execute."""
    x = _x()
    qw, scale = quantize_int8_per_channel(torch.randn(N, K, device="cuda") * 0.05)

    out = run_quant_linear("blockwise_int8", x, qw, weight_scale=scale, group_n=1, group_k=K)
    ref = w8a16_matmul(x, qw, scale, group_n=1, group_k=K)
    assert torch.equal(out, ref)
