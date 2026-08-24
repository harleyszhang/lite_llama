"""W8A8 int8: SmoothQuant — per-channel int8 weights + dynamic per-token int8 activations.

Mirrors sglang's ``w8a8_int8.py``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ...kernels.quantization import smoothquant_matmul
from ...kernels import fused_moe
from .base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from .parameter import RawParameter
from .utils import quantize_int8_per_channel


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

class W8A8Int8Config(QuantizationConfig):
    """SmoothQuant W8A8: per-channel int8 weights + dynamic per-token int8 activations.

    Used by ``--quantization smoothquant``.
    """

    is_dynamic: bool = True

    def __init__(self, ignored: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.group_n = 1
        self.group_k = 1 << 30
        self.ignored = ignored

    def get_name(self) -> str:
        return "w8a8_int8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75  # int8 tensor cores from Turing

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "W8A8Int8Config":
        ignored = tuple(config.get("modules_to_not_convert") or ())
        return cls(ignored=ignored)

    def get_quant_method(
        self, layer: nn.Module, prefix: str = ""
    ) -> QuantizeMethodBase | None:
        if not self.quantizes(prefix):
            from .unquant import UnquantizedLinearMethod, UnquantizedFusedMoEMethod
            from ...modules.moe import SparseMoeBlock
            if isinstance(layer, SparseMoeBlock):
                return UnquantizedFusedMoEMethod()
            return UnquantizedLinearMethod()
        from ...modules.moe import SparseMoeBlock
        if isinstance(layer, SparseMoeBlock):
            return W8A8Int8MoEMethod()
        return W8A8Int8LinearMethod()

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.int8
    
    
class W8A8Int8LinearMethod(LinearMethodBase):
    """SmoothQuant: int8 weights (per-channel) + int8 activations (per-token dynamic)."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        config: W8A8Int8Config = layer.quant  # type: ignore[assignment]
        layer.weight = RawParameter(
            torch.empty(output_size, input_size, dtype=config.storage_dtype)
        )
        layer.weight_scale_inv = RawParameter(
            torch.empty(*config.scale_shape(output_size, input_size), dtype=torch.float32)
        )

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return smoothquant_matmul(
            x, layer.weight, layer.weight_scale_inv,
            bias=bias,
        )

    def quantize_from_fp16(self, layer: nn.Module, config: "QuantizationConfig") -> None:
        qweight, scale = quantize_int8_per_channel(layer.weight.data)
        layer.weight = RawParameter(qweight)
        layer.weight_scale_inv = RawParameter(scale)


class W8A8Int8MoEMethod(FusedMoEMethodBase):
    """SmoothQuant stacked experts: int8 weights + per-token int8 activations through grouped GEMM."""

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        gate_up_n, gate_up_k = 2 * block.moe_intermediate_size, block.hidden_size
        down_n, down_k = block.hidden_size, block.moe_intermediate_size
        return {
            "gate_up_proj": RawParameter(
                torch.empty(block.num_experts, gate_up_n, gate_up_k, dtype=torch.int8)
            ),
            "gate_up_proj_scale_inv": RawParameter(
                torch.empty(block.num_experts, gate_up_n, 1, dtype=torch.float32)
            ),
            "down_proj": RawParameter(
                torch.empty(block.num_experts, down_n, down_k, dtype=torch.int8)
            ),
            "down_proj_scale_inv": RawParameter(
                torch.empty(block.num_experts, down_n, 1, dtype=torch.float32)
            ),
        }

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        return fused_moe(
            x,
            block.experts["gate_up_proj"],
            block.experts["down_proj"],
            topk_weights,
            topk_ids,
            w1_scale=block.experts["gate_up_proj_scale_inv"],
            w2_scale=block.experts["down_proj_scale_inv"],
            group_n=1,
            group_k=block.hidden_size,
        )

    def quantize_from_fp16(self, block: nn.Module, config: "QuantizationConfig") -> None:
        for name in ("gate_up_proj", "down_proj"):
            qweight, scale = quantize_int8_per_channel(block.experts[name].data)
            block.experts[name] = RawParameter(qweight)
            block.experts[f"{name}_scale_inv"] = RawParameter(scale)

