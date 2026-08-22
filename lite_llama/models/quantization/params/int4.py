"""INT4 group-wise weight quantisation (AWQ / GPTQ W4A16).

Called by the w4a16 quant method when the user passes ``--quantization int4``
and the checkpoint ships fp16 weights.
"""

from __future__ import annotations

import torch


def quantize_int4_groupwise(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise ``[N, K]`` fp16 weights to group-wise int4 (AWQ/GPTQ format).

    Each group of ``group_size`` input channels gets its own fp32 scale and zero
    point. The packed output stores 8 int4 values per int32 word.

    Args:
        weight: ``[N, K]`` float weights. K must be a multiple of ``group_size``.
        group_size: Number of input channels per quantisation group.

    Returns:
        ``(qweight, scales, zeros)`` where ``qweight`` is ``[N, K//8]`` int32,
        ``scales`` is ``[N, K//group_size]`` fp32, and ``zeros`` is the same
        shape as ``scales``.
    """
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"in_features {k} must be a multiple of group_size {group_size}")

    w = weight.float().reshape(n, k // group_size, group_size)
    w_min = w.amin(dim=-1)
    w_max = w.amax(dim=-1)

    # Symmetric quantisation: use max(|min|, |max|) as the range.
    qmax = 7.0  # int4 range: [-8, 7]
    scale = (w_max - w_min).clamp(min=1e-5) / (2 * qmax)
    # Centre the zero point so that zero maps to the middle of the int4 range.
    zero = (-w_min / scale).round().clamp(0, 15)
    q = (w / scale.unsqueeze(-1) + zero.unsqueeze(-1)).round().clamp(0, 15).to(torch.int32)

    # Pack 8 int4 values per int32 word along the K dimension.
    q = q.reshape(n, -1, 8)
    shifts = torch.arange(8, device=q.device, dtype=torch.int32) * 4
    packed = (q << shifts[None, None, :]).sum(dim=-1)  # [N, K//8]

    return packed.to(torch.int32), scale.float(), zero.float()
