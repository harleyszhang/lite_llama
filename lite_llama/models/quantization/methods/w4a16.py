"""W4A16 method: group-wise int4 (AWQ/GPTQ) weights, fp16 activation."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....kernels.quantization import w4a16_matmul
from ..config import QuantConfig
from ..parameter import RawParameter
from ..params import quantize_int4_groupwise
from .base import LinearQuantMethod


class W4A16LinearMethod(LinearQuantMethod):
    """Packed int4 weight (8 values per int32 word) + group-wise scales and zeros."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int) -> None:
        quant = layer.quant
        packed_k = (input_size + 7) // 8
        layer.weight = RawParameter(torch.empty(output_size, packed_k, dtype=torch.int32))
        layer.weight_scale = RawParameter(
            torch.empty(*quant.scale_shape(output_size, input_size), dtype=torch.float32)
        )
        layer.weight_zeros = RawParameter(
            torch.empty(*quant.scale_shape(output_size, input_size), dtype=torch.float32)
        )

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return w4a16_matmul(
            x,
            layer.weight,
            layer.weight_scale,
            layer.weight_zeros,
            group_size=layer.quant.group_k,
            bias=layer.bias,
        )

    def convert_from_fp16(self, layer: nn.Module, quant: QuantConfig) -> None:
        qweight, scales, zeros = quantize_int4_groupwise(layer.weight.data, quant.group_k)
        layer.weight = RawParameter(qweight)
        layer.weight_scale = RawParameter(scales)
        layer.weight_zeros = RawParameter(zeros)
