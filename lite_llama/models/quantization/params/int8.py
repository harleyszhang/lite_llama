"""INT8 symmetric weight quantisation (W8A16 / SmoothQuant W8A8).

Called by the quant methods when the user passes ``--quantization
int8|int8-blockwise|smoothquant`` and the checkpoint ships fp16 weights.
"""

from __future__ import annotations

import torch


def quantize_int8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``[N, K]`` fp16 weights to symmetric per-channel int8.

    The scale of row ``n`` is ``max|W[n]| / 127``, so the largest magnitude in
    each output channel maps onto the end of the int8 range.

    Args:
        weight: ``[N, K]`` (or ``[E, N, K]`` for stacked experts) float weights.

    Returns:
        ``(qweight, scales)`` with ``scales`` shaped ``[..., N, 1]``.
    """
    scale = weight.abs().amax(dim=-1, keepdim=True).float() / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qweight = (weight.float() / scale).round().clamp_(-127, 127).to(torch.int8)
    return qweight, scale


def quantize_int8_groupwise(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``[..., K]`` fp16 weights to symmetric int8, one scale per group.

    Same storage as per-channel int8 but a finer scale grid, which recovers
    most of the accuracy per-channel loses on outlier channels.

    Args:
        weight: ``[N, K]`` (or ``[E, N, K]``) float weights. K must be a
            multiple of ``group_size``.
        group_size: Input channels covered by one scale.

    Returns:
        ``(qweight, scales)`` with ``scales`` shaped ``[..., N, K//group_size]``.
    """
    k = weight.shape[-1]
    if k % group_size != 0:
        raise ValueError(f"in_features {k} must be a multiple of group_size {group_size}")
    w = weight.float().unflatten(-1, (k // group_size, group_size))
    scale = w.abs().amax(dim=-1) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qweight = (w / scale.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8)
    return qweight.flatten(-2), scale
