"""Unquantised config and methods — the default for every model.

:class:`UnquantizedLinearMethod` and :class:`UnquantizedFusedMoEMethod`
do plain matmuls; :class:`UnquantizedConfig` exists so the loader has a
uniform "no quant" path rather than None branches.

Usage:
    from rapid_llm.modules.quantization import UnquantizedConfig
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
    run_quant_linear,
)


class UnquantizedLinearMethod(LinearMethodBase):
    """Plain weight projected via kernel dispatch (``F.linear`` floor row)."""

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kw,
    ) -> None:
        layer.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=params_dtype),
            requires_grad=False,
        )

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return run_quant_linear("unquantized", x, layer.weight, bias=bias)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    """Plain stacked experts; the grouped GEMM runs without scales."""

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        # ``block.dtype`` is the model's dtype; the getattrs keep bare stubs
        # (tests) on the fp16 default and pre-EP blocks on the full expert
        # count (``num_local_experts`` only differs under expert parallelism).
        dtype = getattr(block, "dtype", torch.float16)
        num_experts = getattr(block, "num_local_experts", block.num_experts)
        return {
            "gate_up_proj": nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * block.moe_intermediate_size,
                    block.hidden_size,
                    dtype=dtype,
                ),
                requires_grad=False,
            ),
            "down_proj": nn.Parameter(
                torch.empty(
                    num_experts,
                    block.hidden_size,
                    block.moe_intermediate_size,
                    dtype=dtype,
                ),
                requires_grad=False,
            ),
        }

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        from ...kernels import fused_moe

        return fused_moe(
            x,
            block.experts["gate_up_proj"],
            block.experts["down_proj"],
            topk_weights,
            topk_ids,
        )


class UnquantizedConfig(QuantizationConfig):
    """Pseudo-config for the unquantised (no quantisation) path."""

    group_n: int = 1
    group_k: int = 1 << 30

    def get_name(self) -> str:
        return "unquantized"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> UnquantizedConfig:
        return cls()

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        # ignored is empty, so quantizes() is always True and this is a pure
        # layer-type dispatch; _dispatch keeps every config's override alike.
        return self._dispatch(layer, prefix, UnquantizedLinearMethod, UnquantizedFusedMoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        # bf16 is rapid_llm's undeclared-checkpoint dtype (see ModelConfig.dtype);
        # an actual checkpoint's type is carried by ``layer.dtype``, not here.
        return torch.bfloat16
