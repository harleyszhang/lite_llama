"""DeepGEMM dense fp8 GEMM — the ``deepgemm/fp8_gemm_nt`` row's wrapper.

The contract is :class:`~lite_llama.kernels.ops.interfaces.LinearOp`'s superset
signature; the weight arrives the way native fp8 checkpoints store it
(``[N, K]`` uint8 e4m3 bit patterns plus ``weight_scale`` in the native block
convention), and the wrapper owns turning that into DeepGEMM's NT operand:

* the transpose-and-quantise result is cached per weight tensor
  (``data_ptr``-keyed, shape-checked) — inference weights never change, so the
  NT form is built exactly once, which is what the row's ``weight:nt`` /
  ``scale:block_128`` layout tags promise;
* activations are quantised per-token-group-128 on the fly, the same
  "quantisation is part of the format" contract ``linear_w8a8_fp8`` follows
  (native per-token scales are too coarse for DeepGEMM's ``[m, k // 128]``).

The row is ``verified=False``: this wrapper is written against the upstream
``deep_gemm.fp8_gemm_nt`` API but has never run on Hopper, so the golden gate
keeps it out of default dispatch until a max-abs-diff lands.
"""

from __future__ import annotations

import torch

from .quant import nt_block_fp8_from_checkpoint, per_token_group_quant_fp8

# data_ptr -> (w_fp8, w_scales, weight_shape). Entries are tiny (two small
# tensors per linear layer); the shape check catches a freed-and-reused
# allocation before it can serve a stale transpose.
_NT_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Size]] = {}


def _nt_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    group_n: int,
    group_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached NT fp8 form of one linear weight."""
    key = weight.data_ptr()
    hit = _NT_CACHE.get(key)
    if hit is not None and hit[2] == weight.shape:
        return hit[0], hit[1]
    w_fp8, w_scales = nt_block_fp8_from_checkpoint(weight, weight_scale, group_n, group_k)
    _NT_CACHE[key] = (w_fp8, w_scales, weight.shape)
    return w_fp8, w_scales


def fp8_gemm_nt(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    weight_zeros: torch.Tensor | None = None,
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
