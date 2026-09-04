"""Blockwise int8: weight-only int8 (per-channel or group-wise), fp16 activations.

:class:`BlockInt8Config` carries the block shape;
:class:`BlockInt8LinearMethod` dequantises weights in the kernel epilogue
so activations never drop precision.

Usage:
    quant = BlockInt8Config(group_n, group_k, ignored)
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
    allocate_expert_weights,
    allocate_linear_weights,
    run_quant_linear,
)
from .base_config import column_major_scale
from .parameter import RawParameter
from .utils import quantize_int8_groupwise, quantize_int8_per_channel

# --------------------------------------------------------------------------- #
# Runtime quantisation helper
# --------------------------------------------------------------------------- #


def _quantize_int8(
    weight: torch.Tensor, group_k: int, limit: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-wise when a scale spans fewer channels than ``limit``, per-channel otherwise."""
    if group_k < limit:
        return quantize_int8_groupwise(weight, group_k)
    return quantize_int8_per_channel(weight)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class BlockInt8Config(QuantizationConfig):
    """int8 weight-only with per-channel or group-wise scales.

    Used by ``--quantization int8`` (per-channel) and
    ``--quantization int8-blockwise`` (group-wise with default group=128).
    """

    def __init__(
        self, group_n: int = 1, group_k: int = 1 << 30, ignored: tuple[str, ...] = ()
    ) -> None:
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
    def from_config(cls, config: dict[str, Any]) -> BlockInt8Config:
        ignored = tuple(config.get("modules_to_not_convert") or ())
        group_size = int(config.get("group_size", 1 << 30))
        return cls(group_k=group_size, ignored=ignored)

    @classmethod
    def per_channel(cls) -> BlockInt8Config:
        """Symmetric int8, one scale per output channel, computed at load time."""
        return cls(group_n=1, group_k=1 << 30)

    @classmethod
    def groupwise(cls, group_size: int = 128) -> BlockInt8Config:
        """Symmetric int8 with one scale per ``group_size`` input channels."""
        return cls(group_n=1, group_k=group_size)

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        return self._dispatch(layer, prefix, BlockInt8LinearMethod, BlockInt8MoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.int8


class BlockInt8LinearMethod(LinearMethodBase):
    """int8 weight + fp16 activation; per-channel or group-wise scale grid."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        allocate_linear_weights(layer, input_size, output_size)

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        config: BlockInt8Config = layer.quant  # type: ignore[assignment]
        return run_quant_linear(
            "blockwise_int8",
            x,
            layer.weight,
            weight_scale=layer.weight_scale_inv,
            group_n=config.group_n,
            group_k=min(config.group_k, layer.input_size),
            bias=bias,
        )

    def quantize_from_fp16(self, layer: nn.Module, config: QuantizationConfig) -> None:
        cfg: BlockInt8Config = config  # type: ignore[assignment]
        qweight, scale = _quantize_int8(layer.weight.data, cfg.group_k, layer.input_size)
        layer.weight = RawParameter(qweight)
        layer.weight_scale_inv = RawParameter(column_major_scale(scale))


class BlockInt8MoEMethod(FusedMoEMethodBase):
    """int8 stacked experts, fp16 activations."""

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        return allocate_expert_weights(block)

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        from ...kernels import fused_moe

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

    def quantize_from_fp16(self, block: nn.Module, config: QuantizationConfig) -> None:
        cfg: BlockInt8Config = config  # type: ignore[assignment]
        for name in ("gate_up_proj", "down_proj"):
            qweight, scale = _quantize_int8(
                block.experts[name].data, cfg.group_k, block.hidden_size
            )
            block.experts[name] = RawParameter(qweight)
            block.experts[f"{name}_scale_inv"] = RawParameter(column_major_scale(scale))
