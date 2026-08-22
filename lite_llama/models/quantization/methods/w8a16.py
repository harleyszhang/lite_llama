"""W8A16 methods: 8-bit weight (fp8-e4m3 or int8), fp16 activation.

One kernel serves every 8-bit weight-only scheme because they all share the
same layout — a low-bit weight plus a scale grid — and differ only in grid
granularity: fp8 checkpoints in 128x128 blocks, runtime fp8/int8 per output
channel, and int8 block-wise with one scale per ``group_k`` input channels.
SmoothQuant MoE experts also land here: the expert weights are per-channel
int8 while the activations stay fp16 through the grouped GEMM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....kernels import fused_moe
from ....kernels.quantization import w8a16_matmul
from ..config import FP8, QuantConfig
from ..parameter import RawParameter
from ..params import (
    quantize_fp8_per_channel,
    quantize_int8_groupwise,
    quantize_int8_per_channel,
)
from .base import LinearQuantMethod, MoeQuantMethod


def _quantize_weight(weight: torch.Tensor, quant: QuantConfig, in_size: int):
    """Pick the runtime quantiser matching ``quant``'s format and granularity."""
    if quant.format == FP8:
        return quantize_fp8_per_channel(weight)
    if quant.group_k < in_size:
        return quantize_int8_groupwise(weight, quant.group_k)
    return quantize_int8_per_channel(weight)


class W8A16LinearMethod(LinearQuantMethod):
    """8-bit weight + fp16 activation; per-channel or block-wise scale grid."""

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
        return w8a16_matmul(
            x,
            layer.weight,
            layer.weight_scale_inv,
            group_n=quant.group_n,
            group_k=min(quant.group_k, layer.input_size),
            bias=layer.bias,
        )

    def convert_from_fp16(self, layer: nn.Module, quant: QuantConfig) -> None:
        qweight, scale = _quantize_weight(layer.weight.data, quant, layer.input_size)
        layer.weight = RawParameter(qweight)
        layer.weight_scale_inv = RawParameter(scale)


class W8A16MoeMethod(MoeQuantMethod):
    """8-bit stacked experts (fp8 checkpoint or int8 runtime), fp16 activations."""

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        quant = block.quant
        gate_up_n, gate_up_k = 2 * block.moe_intermediate_size, block.hidden_size
        down_n, down_k = block.hidden_size, block.moe_intermediate_size
        return {
            "gate_up_proj": RawParameter(
                torch.empty(block.num_experts, gate_up_n, gate_up_k, dtype=quant.storage_dtype)
            ),
            "gate_up_proj_scale_inv": RawParameter(
                torch.empty(
                    block.num_experts, *quant.scale_shape(gate_up_n, gate_up_k), dtype=torch.float32
                )
            ),
            "down_proj": RawParameter(
                torch.empty(block.num_experts, down_n, down_k, dtype=quant.storage_dtype)
            ),
            "down_proj_scale_inv": RawParameter(
                torch.empty(
                    block.num_experts, *quant.scale_shape(down_n, down_k), dtype=torch.float32
                )
            ),
        }

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        quant = block.quant
        return fused_moe(
            x,
            block.experts["gate_up_proj"],
            block.experts["down_proj"],
            topk_weights,
            topk_ids,
            w1_scale=block.experts["gate_up_proj_scale_inv"],
            w2_scale=block.experts["down_proj_scale_inv"],
            group_n=quant.group_n,
            # Both GEMMs contract a dimension no wider than the hidden size, so
            # clamping once here covers ``down_proj``'s narrower one too.
            group_k=min(quant.group_k, block.hidden_size),
        )

    def convert_from_fp16(self, block: nn.Module, quant: QuantConfig) -> None:
        for name in ("gate_up_proj", "down_proj"):
            qweight, scale = _quantize_weight(block.experts[name].data, quant, block.hidden_size)
            block.experts[name] = RawParameter(qweight)
            block.experts[f"{name}_scale_inv"] = RawParameter(scale)
