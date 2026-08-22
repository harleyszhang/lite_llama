"""Unquantised (fp16) methods: the default when ``quant`` is ``None``."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....kernels import fused_moe
from .base import LinearQuantMethod, MoeQuantMethod


class UnquantizedLinearMethod(LinearQuantMethod):
    """Plain fp16 weight multiplied by ``F.linear``."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int) -> None:
        layer.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float16), requires_grad=False
        )

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, layer.weight, layer.bias)


class UnquantizedMoeMethod(MoeQuantMethod):
    """Plain fp16 stacked experts; the grouped GEMM runs without scales."""

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        return {
            "gate_up_proj": nn.Parameter(
                torch.empty(
                    block.num_experts,
                    2 * block.moe_intermediate_size,
                    block.hidden_size,
                    dtype=torch.float16,
                ),
                requires_grad=False,
            ),
            "down_proj": nn.Parameter(
                torch.empty(
                    block.num_experts,
                    block.hidden_size,
                    block.moe_intermediate_size,
                    dtype=torch.float16,
                ),
                requires_grad=False,
            ),
        }

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        return fused_moe(
            x,
            block.experts["gate_up_proj"],
            block.experts["down_proj"],
            topk_weights,
            topk_ids,
        )
