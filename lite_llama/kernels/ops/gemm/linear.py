"""Entry points of the ``linear`` logical op: one per quantisation scheme.

The projection GEMM is the same operation in every model, but the numeric
format decides which kernel runs it. That format is a *dispatch key*, not a
runtime branch: :func:`~lite_llama.kernels.dispatcher` already knows the
scheme when it picks a row, so each function below may assume its own format
and map straight onto the Triton kernels in
:mod:`lite_llama.kernels.ops.quantization` — no ``if scheme ==`` chain anywhere on
the hot path. All of them share the
:class:`~lite_llama.kernels.ops.interfaces.LinearOp` signature so a caller
never has to know which one it got; scales it does not have are simply
``None``.

Usage:
    y = linear_torch(x, weight, bias=bias)                          # unquantised
    y = linear_w8a16(x, qweight, weight_scale=scales, group_k=128)  # fp8/int8 weight
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    weight_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """Unquantised floor: ``F.linear``, i.e. cuBLAS.

    This is the row every scheme can fall back to and the one the golden
    baselines are measured against — cuBLAS is the correct floor here, a Triton
    tile would be the optimisation. A quantised scheme reaching this function
    fails loudly instead of silently running a plain GEMM on packed bytes.
    """
    if weight_scale is not None or weight_zeros is not None:
        raise ValueError(
            "native/linear_torch serves the unquantised floor; a quantised "
            "scheme reached it, which is a dispatch or registration bug"
        )
    return F.linear(x, weight, bias)


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
    from ..quantization import w8a16_matmul

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
    from ..quantization import w4a16_matmul

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
    from ..quantization import smoothquant_matmul

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
    """True W8A8 fp8: per-token activation quantisation feeding the fp8 GEMM.

    Quantising the activations is part of running this format, so it happens
    here rather than in the caller — the same way the SmoothQuant kernel
    quantises its own activations internally.
    """
    # Imported lazily: the helper lives with the quantisation *methods*, and
    # this module is itself loaded lazily by dispatch, so nothing pays for it
    # unless an fp8 checkpoint is actually served.
    from ....modules.quantization.utils import quantize_fp8_per_token
    from ..quantization import fp8_matmul

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
