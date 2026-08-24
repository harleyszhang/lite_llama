"""Unquantised (fp16) config and methods — the default for any un-quantised model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...kernels import fused_moe
from .base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)


class UnquantizedLinearMethod(LinearMethodBase):
    """Plain fp16 weight multiplied by ``F.linear``."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        layer.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float16), requires_grad=False
        )

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
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


class UnquantizedConfig(QuantizationConfig):
    """Pseudo-config for the fp16 (no quantisation) path."""

    group_n: int = 1
    group_k: int = 1 << 30

    def get_name(self) -> str:
        return "fp16"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "UnquantizedConfig":
        return cls()

    def get_quant_method(
        self, layer: nn.Module, prefix: str = ""
    ) -> QuantizeMethodBase | None:
        # Dispatch by layer type
        from ...modules.moe import SparseMoeBlock

        if isinstance(layer, SparseMoeBlock):
            return UnquantizedFusedMoEMethod()
        return UnquantizedLinearMethod()

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.float16
