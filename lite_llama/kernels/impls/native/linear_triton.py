"""Triton rows for the ``linear`` op — the four quantised GEMMs behind one signature.

Each scheme gets its own entry point because the numeric format is a dispatch
key dimension, not a runtime branch: dispatch already knows the scheme when
it picks the row, so the loaded implementation can assume its format and map
straight onto the existing Triton kernels in
:mod:`lite_llama.kernels.quantization` (which stay untouched — these are
adapters, not rewrites). ``w8a8_fp8`` additionally quantises activations
per token inside the impl; that is an implementation detail the caller never
sees, exactly like the per-token int8 quantisation inside the SmoothQuant
kernel.

Usage:
    y = linear_w8a16(x, qweight, weight_scale=scales, group_n=1, group_k=138)
"""

from __future__ import annotations

import torch

from lite_llama.kernels.quantization import (
    fp8_matmul,
    smoothquant_matmul,
    w4a16_matmul,
    w8a16_matmul,
)
from lite_llama.modules.quantization.utils import quantize_fp8_per_token


def linear_w8a16(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    weight_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """8-bit weight-only GEMM (fp8-e4m3 ``uint8`` or symmetric ``int8``)."""
    if weight_scale is None:
        raise ValueError("linear_w8a16 needs weight_scale (fp8 or blockwise-int8 row)")
    return w8a16_matmul(x, weight, weight_scale, group_n=group_n or 1, group_k=group_k, bias=bias)


def linear_w4a16(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    weight_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """4-bit weight-only GEMM (AWQ/GPTQ packed ``int32``, asymmetric zero points)."""
    if weight_scale is None or weight_zeros is None:
        raise ValueError("linear_w4a16 needs weight_scale and weight_zeros (awq/gptq row)")
    # group_size is the k-axis block of the int4 groups; group_n has no analogue.
    return w4a16_matmul(x, weight, weight_scale, weight_zeros, group_size=group_k or 128, bias=bias)


def linear_w8a8_int8(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    weight_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """SmoothQuant W8A8: int8 weights, per-token int8 activations in-kernel."""
    if weight_scale is None:
        raise ValueError("linear_w8a8_int8 needs weight_scale (smoothquant per-channel)")
    return smoothquant_matmul(x, weight, weight_scale, bias=bias)


def linear_w8a8_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    weight_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """True W8A8 fp8: per-token activation quantisation feeding the fp8 GEMM."""
    if weight_scale is None:
        raise ValueError("linear_w8a8_fp8 needs weight_scale (block-wise fp8)")
    qx, x_scale = quantize_fp8_per_token(x)
    return fp8_matmul(
        qx,
        x_scale,
        weight,
        weight_scale,
        group_n=group_n or 1,
        group_k=group_k,
        bias=bias,
        out_dtype=x.dtype,
    )
