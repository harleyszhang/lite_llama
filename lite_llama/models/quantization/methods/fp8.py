"""True W8A8 fp8: fp8-e4m3 weights + dynamic per-token fp8-e4m3 activations.

Mirrors vLLM's ``fp8.py``: the weight storage is identical to w8a16 fp8
(uint8 e4m3 bytes plus a scale grid — a checkpoint's fine-grained blocks or a
runtime per-channel grid), but ``apply`` quantises the activations per token
and runs the fp8 GEMM instead of keeping them fp16. The weight is never
dequantised to fp16 outside the kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....kernels.quantization import fp8_matmul
from ..config import QuantConfig
from ..parameter import RawParameter
from ..params import quantize_fp8_per_channel, quantize_fp8_per_token
from .base import LinearQuantMethod


class Fp8LinearMethod(LinearQuantMethod):
    """fp8-e4m3 weights + per-token fp8-e4m3 activations (no calibration)."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int) -> None:
        quant = layer.quant
        layer.weight = RawParameter(
            torch.empty(output_size, input_size, dtype=quant.storage_dtype)
        )
        layer.weight_scale_inv = RawParameter(
            torch.empty(*quant.scale_shape(output_size, input_size), dtype=torch.float32)
        )

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        quant = layer.quant
        qx, x_scale = quantize_fp8_per_token(x)
        return fp8_matmul(
            qx,
            x_scale,
            layer.weight,
            layer.weight_scale_inv,
            group_n=quant.group_n,
            group_k=min(quant.group_k, layer.input_size),
            bias=layer.bias,
        )

    def convert_from_fp16(self, layer: nn.Module, quant: QuantConfig) -> None:
        qweight, scale = quantize_fp8_per_channel(layer.weight.data)
        layer.weight = RawParameter(qweight)
        layer.weight_scale_inv = RawParameter(scale)
