"""Fp8 weight-only config (fp8-e4m3 weights, fp16 activations).

Covers Qwen3 / DeepSeek fp8 checkpoints with 128×128 block-wise scales.
Also serves the weight-only portion of MoE experts in fp8 checkpoints.
Mirrors sglang's ``fp8.py`` (the weight-only variant, not W8A8).
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
from .utils import quantize_fp8_per_channel

#: Block size of the fine-grained FP8 format (Qwen/DeepSeek checkpoints).
FP8_BLOCK = 128

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class Fp8Config(QuantizationConfig):
    """fp8-e4m3 weight-only (W8A16) with block-wise or per-channel scales.

    Attributes:
        group_n / group_k: Block dimensions of the scale grid.
        ignored: Checkpoint modules left in fp16.
    """

    def __init__(
        self,
        group_n: int = FP8_BLOCK,
        group_k: int = FP8_BLOCK,
        ignored: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.group_n = group_n
        self.group_k = group_k
        self.ignored = ignored
        self.method = ""

    def get_name(self) -> str:
        return "fp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89  # Ada / Hopper for native fp8; Ampere via software

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Fp8Config:
        fmt = str(config.get("fmt", "e4m3")).lower()
        if fmt != "e4m3":
            raise ValueError(f"unsupported fp8 format {fmt!r}; only e4m3 is implemented")
        block = config.get("weight_block_size") or [FP8_BLOCK, FP8_BLOCK]
        gn, gk = int(block[0]), int(block[1])
        if gk % FP8_BLOCK != 0 or gn % FP8_BLOCK != 0:
            raise ValueError(
                f"weight_block_size {block} is not a multiple of {FP8_BLOCK}; "
                "the w8a16 kernel tiles k in 128-wide steps"
            )
        ignored = tuple(config.get("modules_to_not_convert") or ())
        return cls(gn, gk, ignored)

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        return self._dispatch(layer, prefix, Fp8LinearMethod, Fp8MoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.uint8

    @property
    def is_fp8(self) -> bool:
        return True


class Fp8LinearMethod(LinearMethodBase):
    """fp8-e4m3 weight + fp16 activation; per-channel or block-wise scale grid."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        config: Fp8Config = layer.quant  # type: ignore[assignment]
        layer.weight = RawParameter(
            torch.empty(output_size, input_size, dtype=config.storage_dtype)
        )
        layer.weight_scale_inv = RawParameter(
            torch.empty(*config.scale_shape(output_size, input_size), dtype=torch.float32)
        )

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        config: Fp8Config = layer.quant  # type: ignore[assignment]
        return w8a16_matmul(
            x,
            layer.weight,
            layer.weight_scale_inv,
            group_n=config.group_n,
            group_k=min(config.group_k, layer.input_size),
            bias=bias,
        )

    def quantize_from_fp16(self, layer: nn.Module, config: QuantizationConfig) -> None:
        qweight, scale = quantize_fp8_per_channel(layer.weight.data)
        layer.weight = RawParameter(qweight)
        layer.weight_scale_inv = RawParameter(scale)


class Fp8MoEMethod(FusedMoEMethodBase):
    """fp8 stacked experts (checkpoint), fp16 activations through grouped GEMM."""

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        config: Fp8Config = block.quant  # type: ignore[assignment]
        gate_up_n, gate_up_k = 2 * block.moe_intermediate_size, block.hidden_size
        down_n, down_k = block.hidden_size, block.moe_intermediate_size
        return {
            "gate_up_proj": RawParameter(
                torch.empty(block.num_experts, gate_up_n, gate_up_k, dtype=config.storage_dtype)
            ),
            "gate_up_proj_scale_inv": RawParameter(
                torch.empty(
                    block.num_experts,
                    *config.scale_shape(gate_up_n, gate_up_k),
                    dtype=torch.float32,
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
        config: Fp8Config = block.quant  # type: ignore[assignment]
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
        for name in ("gate_up_proj", "down_proj"):
            qweight, scale = quantize_fp8_per_channel(block.experts[name].data)
            block.experts[name] = RawParameter(qweight)
            block.experts[f"{name}_scale_inv"] = RawParameter(scale)
