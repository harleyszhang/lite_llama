"""W8A8 SmoothQuant method: per-channel int8 weights + dynamic int8 activations."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....kernels.quantization import smoothquant_matmul
from ..config import QuantConfig
from ..parameter import RawParameter
from ..params import quantize_int8_per_channel
from .base import LinearQuantMethod


class SmoothQuantLinearMethod(LinearQuantMethod):
    """SmoothQuant: both operands are int8, so the GEMM runs on int8 tensor cores.

    Weights are pre-quantised per output channel (static); activations are
    quantised per token inside the kernel (dynamic).
    """

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int) -> None:
        quant = layer.quant
        layer.weight = RawParameter(
            torch.empty(output_size, input_size, dtype=quant.storage_dtype)
        )
        layer.weight_scale_inv = RawParameter(
            torch.empty(*quant.scale_shape(output_size, input_size), dtype=torch.float32)
        )

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return smoothquant_matmul(x, layer.weight, layer.weight_scale_inv, bias=layer.bias)

    def convert_from_fp16(self, layer: nn.Module, quant: QuantConfig) -> None:
        qweight, scale = quantize_int8_per_channel(layer.weight.data)
        layer.weight = RawParameter(qweight)
        layer.weight_scale_inv = RawParameter(scale)
