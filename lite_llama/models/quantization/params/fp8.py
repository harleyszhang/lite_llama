"""FP8-e4m3 weight quantisation (W8A16, runtime scheme).

Called by the w8a16 quant method when the user passes ``--quantization fp8``
and the checkpoint ships fp16 weights. Weights are scaled into the e4m3 range
per output channel; activations stay fp16, so no calibration data is needed.
"""

from __future__ import annotations

import torch

#: Largest finite magnitude of the e4m3 format.
FP8_E4M3_MAX = 448.0


def quantize_fp8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``[N, K]`` fp16 weights to fp8-e4m3 with per-output-channel scales.

    Args:
        weight: ``[N, K]`` (or ``[E, N, K]`` for stacked experts) float weights.

    Returns:
        ``(qweight, scales)`` where ``qweight`` is ``uint8`` holding the e4m3
        bit pattern (the container the w8a16 kernel expects) and ``scales`` is
        ``[..., N, 1]`` fp32.
    """
    scale = weight.abs().amax(dim=-1, keepdim=True).float() / FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (weight.float() / scale).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q.view(torch.uint8), scale
