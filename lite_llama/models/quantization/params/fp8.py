"""FP8-e4m3 quantisation: per-channel weights and per-token activations (W8A8).

Called by the fp8 quant method: weights when the user passes ``--quantization
fp8`` and the checkpoint ships fp16 weights, activations on every forward of a
true W8A8 fp8 linear. Weights are scaled into the e4m3 range per output channel;
activations are quantised per token at runtime, so no calibration data is needed.
"""

from __future__ import annotations

import torch

#: Largest finite magnitude of the e4m3 format.
FP8_E4M3_MAX = 448.0


def _quantize_fp8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale ``values`` into the e4m3 range and return the ``uint8`` bit pattern."""
    q = (values / scale).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q.view(torch.uint8)


def quantize_fp8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``[N, K]`` fp16 weights to fp8-e4m3 with per-output-channel scales.

    Args:
        weight: ``[N, K]`` (or ``[E, N, K]`` for stacked experts) float weights.

    Returns:
        ``(qweight, scales)`` where ``qweight`` is ``uint8`` holding the e4m3
        bit pattern (the container the fp8 kernel expects) and ``scales`` is
        ``[..., N, 1]`` fp32.
    """
    scale = weight.abs().amax(dim=-1, keepdim=True).float() / FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return _quantize_fp8(weight.float(), scale), scale


def quantize_fp8_per_tensor(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Quantise ``x`` to fp8-e4m3 with one caller-supplied scalar scale.

    The KV-cache scheme (vLLM-style): the scale is a model-level constant
    rather than measured per call, so it ships with the attention module.

    Args:
        x: fp16 values of any shape.
        scale: ``x / scale`` is what actually lands in e4m3; ``1.0`` stores
            raw values, clamped to the e4m3 range.

    Returns:
        ``uint8`` tensor holding the e4m3 bit pattern, shaped like ``x``.
    """
    return _quantize_fp8(x.float(), torch.full((), scale, dtype=torch.float32, device=x.device))


def quantize_fp8_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise activations to fp8-e4m3, one scale per token (row).

    Args:
        x: ``[..., K]`` fp16/bf16 activations.

    Returns:
        ``(qx, scales)`` with ``qx`` the ``uint8`` e4m3 bit pattern shaped like
        ``x`` and ``scales`` ``[..., 1]`` fp32.
    """
    flat = x.reshape(-1, x.shape[-1]).float()
    scale = flat.abs().amax(dim=-1, keepdim=True) / FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qx = _quantize_fp8(flat, scale).reshape(x.shape)
    return qx, scale.reshape(*x.shape[:-1], 1)
