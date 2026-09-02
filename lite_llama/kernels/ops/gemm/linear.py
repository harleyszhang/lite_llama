"""Entry points of the ``linear`` logical op: one per quantisation scheme.

Each function shares the ``(x, weight)`` signature — the weight object
knows its own quant format — so ``run_quant_linear`` can select on scheme
and call the row without a conditional chain at the call site.

Usage:
    y = linear_torch(x, weight)
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
    weight_global_scale: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """Unquantised floor: ``F.linear``, i.e. cuBLAS.

    This is the row every scheme can fall back to and the one the golden
    baselines are measured against — cuBLAS is the correct floor here, a Triton
    tile would be the optimisation. A quantised scheme reaching this function
    fails loudly instead of silently running a plain GEMM on packed bytes.
    """
    if weight_scale is not None or weight_zeros is not None or weight_global_scale is not None:
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
    weight_global_scale: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """8-bit weight-only GEMM: fp8-e4m3 or symmetric ``int8`` bytes, or the
    asymmetric int8 of a GPTQ ``bits=8`` checkpoint (``weight_zeros`` present,
    routed here under the ``gptq_int8`` scheme once the load-time unpack has
    expanded the checkpoint's packed words to one byte per element).
    """
    from ..quantization import w8a16_matmul

    if weight_scale is None:
        raise ValueError("linear_w8a16 needs weight_scale (fp8 or blockwise-int8 row)")
    return w8a16_matmul(
        x,
        weight,
        weight_scale,
        zeros=weight_zeros,
        group_n=group_n or 1,
        group_k=group_k,
        bias=bias,
    )


def linear_w4a16(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    weight_zeros: torch.Tensor | None = None,
    weight_global_scale: torch.Tensor | None = None,
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
    weight_global_scale: torch.Tensor | None = None,
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
    weight_global_scale: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """True W8A8 fp8: per-token activation quantisation feeding the fp8 GEMM.

    Quantising the activations is part of running this format, so it happens
    here rather than in the caller — the same way the SmoothQuant kernel
    quantises its own activations internally.

    The quantiser is the fused Triton one rather than the torch helper: the torch
    chain costs a shape-independent 45-55 us on an H100, which at decode sizes is
    two to three times the GEMM it feeds and made this whole scheme lose to bf16
    cuBLAS for reasons unrelated to fp8. ``benchmarks/kernels/bench_quant_gemm.py``
    carries the ablation row that isolated it.
    """
    # Imported lazily: this module is itself loaded lazily by dispatch, so
    # nothing pays for the quantisation kernels unless a checkpoint is served.
    from ..quantization import fp8_matmul, fp8_quantize_per_token

    if weight_scale is None:
        raise ValueError("linear_w8a8_fp8 needs weight_scale (block-wise fp8)")
    qx, x_scale = fp8_quantize_per_token(x)
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


def linear_nvfp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    weight_zeros: torch.Tensor | None = None,
    weight_global_scale: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """NVFP4 weight-only GEMM: e2m1 nibbles, e4m3 block scales, fp32 global scale.

    The only row that needs ``weight_global_scale``, and the reason the parameter
    is in :class:`~lite_llama.kernels.ops.interfaces.LinearOp` at all: NVFP4's
    block scales are themselves quantised, so reconstructing a weight takes two
    multiplies and the second one has nowhere else to travel.

    ``group_n``/``group_k`` are ignored — the 16-element block is fixed by the
    format, not configured by the checkpoint, so there is nothing for them to say.
    """
    from ..quantization import nvfp4_matmul

    if weight_scale is None or weight_global_scale is None:
        raise ValueError("linear_nvfp4 needs weight_scale (block) and weight_global_scale (tensor)")
    return nvfp4_matmul(x, weight, weight_scale, weight_global_scale, bias=bias)
