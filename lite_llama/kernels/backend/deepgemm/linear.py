"""DeepGEMM dense fp8 GEMM — the ``deepgemm/fp8_gemm_nt`` row's wrapper.

``fp8_gemm_nt`` repacks the weight into DeepGEMM's NT layout via
:func:`_nt_weight` and calls the library, standing behind the native
linear signature so dispatch can pick it transparently.

Usage:
    y = fp8_gemm_nt(x, weight)
"""

from __future__ import annotations

import torch

from .quant import nt_block_fp8_from_checkpoint, per_token_group_quant_fp8

# data_ptr -> (w_fp8, w_scales, source_weight). Holding the source tensor is
# what makes the data_ptr key sound: while the reference lives, the caching
# allocator cannot hand that address to another tensor, so a hit is always
# this weight's own transpose. A shape check could not promise that — a
# freed-and-reused allocation of the same shape used to slip through. A
# reloaded layer builds a fresh tensor, misses the identity check and repacks;
# the old entry just waits to be overwritten.
_NT_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _nt_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    group_n: int,
    group_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached NT fp8 form of one linear weight."""
    key = weight.data_ptr()
    hit = _NT_CACHE.get(key)
    if hit is not None and hit[2] is weight:
        return hit[0], hit[1]
    w_fp8, w_scales = nt_block_fp8_from_checkpoint(weight, weight_scale, group_n, group_k)
    _NT_CACHE[key] = (w_fp8, w_scales, weight)
    return w_fp8, w_scales


def fp8_gemm_nt(
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
    """``dequant(x) @ dequant(weight).T (+ bias)`` on the Hopper fp8 path.

    Args:
        x: ``[tokens, in_features]`` activations (``in_features`` a multiple
            of 128, as DeepGEMM requires).
        weight: ``[out_features, in_features]`` uint8 e4m3 bit patterns.
        bias: Optional ``[out_features]`` additive bias.
        weight_scale: Native block scales; ``None`` only for bf16 weights.
        weight_zeros: Ignored — fp8 is symmetric.
        weight_global_scale: Ignored — fp8 block scales are stored in fp32, so
            there is no second level to undo. Present because
            :class:`~lite_llama.kernels.ops.interfaces.LinearOp` is the superset
            signature every row implements.
        group_n: Rows per weight-scale block (``0`` = per-tensor).
        group_k: Columns per weight-scale block (``0`` = per-tensor).

    Returns:
        ``[tokens, out_features]`` in ``x.dtype``.
    """
    import deep_gemm  # the JIT kernels live with the library; import at call time

    w_fp8, w_scales = _nt_weight(weight, weight_scale, group_n, group_k)
    qx, x_scales = per_token_group_quant_fp8(x)
    out = deep_gemm.fp8_gemm_nt(qx, x_scales, w_fp8, w_scales)  # bf16 [m, n]
    if out.dtype != x.dtype:
        out = out.to(x.dtype)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out
