"""W8A8 fp8: true W8A8 with fp8-e4m3 weights + dynamic per-token fp8 activations.

The weight storage is identical to the Fp8Config (uint8 e4m3 bytes + scale),
but ``apply`` quantises the activations per token and runs the fp8 GEMM
instead of keeping them fp16. The weight is never dequantised to fp16.
Mirrors sglang's ``w8a8_fp8.py``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ...kernels import fused_moe
from ...kernels.quantization import fp8_matmul
from .base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from .parameter import RawParameter
from .utils import quantize_fp8_per_channel, quantize_fp8_per_token

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class W8A8Fp8Config(QuantizationConfig):
    """True W8A8 with fp8-e4m3: weights per-channel, activations per-token.

    Used by ``--quantization fp8`` (runtime path: converts fp16 checkpoint to
    fp8 per-channel weights on the fly).
    """

    is_dynamic: bool = True

    def __init__(
        self, group_n: int = 1, group_k: int = 1 << 30, ignored: tuple[str, ...] = ()
    ) -> None:
        super().__init__()
        self.group_n = group_n
        self.group_k = group_k
        self.ignored = ignored

    def get_name(self) -> str:
        return "w8a8_fp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> W8A8Fp8Config:
        ignored = tuple(config.get("modules_to_not_convert") or ())
        return cls(ignored=ignored)

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        return self._dispatch(layer, prefix, W8A8Fp8LinearMethod, W8A8Fp8MoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.uint8


class W8A8Fp8LinearMethod(LinearMethodBase):
    """fp8-e4m3 weights + per-token fp8-e4m3 activations (no calibration)."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        config: W8A8Fp8Config = layer.quant  # type: ignore[assignment]
        layer.weight = RawParameter(
            torch.empty(output_size, input_size, dtype=config.storage_dtype)
        )
        layer.weight_scale_inv = RawParameter(
            torch.empty(*config.scale_shape(output_size, input_size), dtype=torch.float32)
        )

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        config: W8A8Fp8Config = layer.quant  # type: ignore[assignment]
        qx, x_scale = quantize_fp8_per_token(x)
        return fp8_matmul(
            qx,
            x_scale,
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


class W8A8Fp8MoEMethod(FusedMoEMethodBase):
    """W8A8 fp8 stacked experts: fp8 weights + per-token fp8 activations through grouped GEMM."""

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        config: W8A8Fp8Config = block.quant  # type: ignore[assignment]
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

    def quantize_from_fp16(self, block: nn.Module, config: QuantizationConfig) -> None:
        for name in ("gate_up_proj", "down_proj"):
            qweight, scale = quantize_fp8_per_channel(block.experts[name].data)
            block.experts[name] = RawParameter(qweight)
            block.experts[f"{name}_scale_inv"] = RawParameter(scale)
