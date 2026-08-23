"""Blockwise int8: weight-only int8 (per-channel or group-wise), fp16 activations.

Mirrors sglang's ``blockwise_int8.py``. Covers ``--quantization int8`` and
``--quantization int8-blockwise``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ...kernels import fused_moe
from ...kernels.quantization import w8a16_matmul
from .base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from .parameter import RawParameter
from .utils import quantize_int8_groupwise, quantize_int8_per_channel


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

class BlockInt8Config(QuantizationConfig):
    """int8 weight-only with per-channel or group-wise scales.

    Used by ``--quantization int8`` (per-channel) and
    ``--quantization int8-blockwise`` (group-wise with default group=128).
    """

    def __init__(self, group_n: int = 1, group_k: int = 1 << 30, ignored: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.group_n = group_n
        self.group_k = group_k
        self.ignored = ignored

    def get_name(self) -> str:
        return "blockwise_int8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # Volta

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BlockInt8Config":
        ignored = tuple(config.get("modules_to_not_convert") or ())
        group_size = int(config.get("group_size", 1 << 30))
        return cls(group_k=group_size, ignored=ignored)

    @classmethod
    def per_channel(cls) -> "BlockInt8Config":
        """Symmetric int8, one scale per output channel, computed at load time."""
        return cls(group_n=1, group_k=1 << 30)

    @classmethod
    def groupwise(cls, group_size: int = 128) -> "BlockInt8Config":
        """Symmetric int8 with one scale per ``group_size`` input channels."""
        return cls(group_n=1, group_k=group_size)

    def get_quant_method(
        self, layer: nn.Module, prefix: str = ""
    ) -> QuantizeMethodBase | None:
        if not self.quantizes(prefix):
            from ...modules.moe import SparseMoeBlock
            from .unquant import UnquantizedFusedMoEMethod, UnquantizedLinearMethod
            if isinstance(layer, SparseMoeBlock):
                return UnquantizedFusedMoEMethod()
            return UnquantizedLinearMethod()

        from ...modules.moe import SparseMoeBlock
        if isinstance(layer, SparseMoeBlock):
            return BlockInt8MoEMethod()
        return BlockInt8LinearMethod()

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.int8


class BlockInt8LinearMethod(LinearMethodBase):
    """int8 weight + fp16 activation; per-channel or group-wise scale grid."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        config: BlockInt8Config = layer.quant  # type: ignore[assignment]
        layer.weight = RawParameter(
            torch.empty(output_size, input_size, dtype=config.storage_dtype)
        )
        layer.weight_scale_inv = RawParameter(
            torch.empty(*config.scale_shape(output_size, input_size), dtype=torch.float32)
        )

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        config: BlockInt8Config = layer.quant  # type: ignore[assignment]
        return w8a16_matmul(
            x,
            layer.weight,
            layer.weight_scale_inv,
            group_n=config.group_n,
            group_k=min(config.group_k, layer.input_size),
            bias=bias if bias is not None else layer.bias,
        )

    def quantize_from_fp16(self, layer: nn.Module, config: "QuantizationConfig") -> None:
        cfg: BlockInt8Config = config  # type: ignore[assignment]
        if cfg.group_k < layer.input_size:
            qweight, scale = quantize_int8_groupwise(layer.weight.data, cfg.group_k)
        else:
            qweight, scale = quantize_int8_per_channel(layer.weight.data)
        layer.weight = RawParameter(qweight)
        layer.weight_scale_inv = RawParameter(scale)


class BlockInt8MoEMethod(FusedMoEMethodBase):
    """int8 stacked experts, fp16 activations."""

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        config: BlockInt8Config = block.quant  # type: ignore[assignment]
        gate_up_n, gate_up_k = 2 * block.moe_intermediate_size, block.hidden_size
        down_n, down_k = block.hidden_size, block.moe_intermediate_size
        return {
            "gate_up_proj": RawParameter(
                torch.empty(block.num_experts, gate_up_n, gate_up_k, dtype=config.storage_dtype)
            ),
            "gate_up_proj_scale_inv": RawParameter(
                torch.empty(
                    block.num_experts, *config.scale_shape(gate_up_n, gate_up_k), dtype=torch.float32
                )
            ),
            "down_proj": RawParameter(
                torch.empty(block.num_experts, down_n, down_k, dtype=config.storage_dtype)
            ),
            "down_proj_scale_inv": RawParameter(
                torch.empty(
                    block.num_experts, *config.scale_shape(down_n, down_k), dtype=torch.float32
                )
            ),
        }

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        config: BlockInt8Config = block.quant  # type: ignore[assignment]
        return fused_moe(
            x,
            block.experts["gate_up_proj"],
            block.experts["down_proj"],
            topk_weights,
            topk_ids,
            w1_scale=block.experts["gate_up_proj_scale_inv"],
            w2_scale=block.experts["down_proj_scale_inv"],
            group_n=config.group_n,
            group_k=min(config.group_k, block.hidden_size),
        )

    def quantize_from_fp16(self, block: nn.Module, config: "QuantizationConfig") -> None:
        cfg: BlockInt8Config = config  # type: ignore[assignment]
        for name in ("gate_up_proj", "down_proj"):
            if cfg.group_k < block.hidden_size:
                qweight, scale = quantize_int8_groupwise(block.experts[name].data, cfg.group_k)
            else:
                qweight, scale = quantize_int8_per_channel(block.experts[name].data)
            block.experts[name] = RawParameter(qweight)
            block.experts[f"{name}_scale_inv"] = RawParameter(scale)
