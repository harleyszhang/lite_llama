"""AWQ config and method (mirrors sglang ``awq/awq.py``).

:class:`AWQConfig` carries the checkpoint's group size;
:class:`AWQLinearMethod` / :class:`AWQMoEMethod` create the packed int4
weights and call the w4a16 kernel at run time.

Usage:
    quant = AWQConfig(group_size, ignored)
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
from .base_config import expert_scale_parameter, scale_parameter
from .parameter import RawParameter
from .utils import quantize_int4_groupwise

#: int4 values per int32 storage word.
_PACK_FACTOR = 8

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class AWQConfig(QuantizationConfig):
    """AutoAWQ checkpoint config: group-wise int4 with configurable group size."""

    def __init__(self, group_size: int = 128, ignored: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.group_n = 1
        self.group_k = group_size
        self.ignored = ignored
        self.method = "awq"

    def get_name(self) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # w4a16 unpacks to fp32 and casts to the activation dtype, so bf16
        # activations run through the same kernel.
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AWQConfig:
        bits = int(config.get("bits", 4))
        if bits != 4:
            raise ValueError(f"only 4-bit AWQ is supported, got {bits}")
        group_size = int(config.get("group_size", 128))
        if group_size <= 0 or (group_size & (group_size - 1)) != 0:
            raise ValueError(f"group_size must be a positive power of 2, got {group_size}")
        ignored = tuple(config.get("modules_to_not_convert") or ())
        return cls(group_size, ignored)

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        return self._dispatch(layer, prefix, AWQLinearMethod, AWQMoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.int32

    @property
    def is_int4(self) -> bool:
        return True

    @property
    def is_packed(self) -> bool:
        return True


class AWQLinearMethod(LinearMethodBase):
    """Group-wise int4 from an AutoAWQ checkpoint; runs the w4a16 kernel."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        config: AWQConfig = layer.quant  # type: ignore[assignment]
        packed_k = (input_size + _PACK_FACTOR - 1) // _PACK_FACTOR
        layer.weight = RawParameter(torch.empty(output_size, packed_k, dtype=torch.int32))
        layer.weight_scale = scale_parameter(config.scale_shape(output_size, input_size))
        layer.weight_zeros = scale_parameter(config.scale_shape(output_size, input_size))

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return run_quant_linear(
            "awq",
            x,
            layer.weight,
            weight_scale=layer.weight_scale,
            weight_zeros=layer.weight_zeros,
            group_k=layer.quant.group_k,
            bias=bias,
        )

    def quantize_from_fp16(self, layer: nn.Module, config: QuantizationConfig) -> None:
        cfg: AWQConfig = config  # type: ignore[assignment]
        qweight, scales, zeros = quantize_int4_groupwise(layer.weight.data, cfg.group_k)
        layer.weight = RawParameter(qweight)
        layer.weight_scale = RawParameter(scales)
        layer.weight_zeros = RawParameter(zeros)


class AWQMoEMethod(FusedMoEMethodBase):
    """AWQ int4 MoE: group-wise int4 stacked experts through fused_moe w4a16 path.

    Expert weights load in the checkpoint's ``[E, N, K//8]`` int32 packing
    (8 nibbles per word) with ``[E, N, K//group_k]`` fp32 scales and zeros;
    :meth:`process_weights_after_loading` then swaps each stacked tensor for
    the fused kernel's byte packing (``[E, N, K//2]`` uint8) in one repack.
    """

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        config: AWQConfig = block.quant  # type: ignore[assignment]
        gate_up_n = 2 * block.moe_intermediate_size
        gate_up_k = block.hidden_size
        down_n = block.hidden_size
        down_k = block.moe_intermediate_size
        num_groups_gu = (gate_up_k + config.group_k - 1) // config.group_k
        num_groups_d = (down_k + config.group_k - 1) // config.group_k
        return {
            "gate_up_proj": RawParameter(
                torch.empty(
                    block.num_experts, gate_up_n, gate_up_k // _PACK_FACTOR, dtype=torch.int32
                )
            ),
            "gate_up_proj_scale": expert_scale_parameter(
                block.num_experts, (gate_up_n, num_groups_gu)
            ),
            "gate_up_proj_zeros": expert_scale_parameter(
                block.num_experts, (gate_up_n, num_groups_gu)
            ),
            "down_proj": RawParameter(
                torch.empty(block.num_experts, down_n, down_k // _PACK_FACTOR, dtype=torch.int32)
            ),
            "down_proj_scale": expert_scale_parameter(
                block.num_experts, (down_n, num_groups_d)
            ),
            "down_proj_zeros": expert_scale_parameter(
                block.num_experts, (down_n, num_groups_d)
            ),
        }

    def process_weights_after_loading(self, block: nn.Module) -> None:
        """Repack the int32 word layout into the GEMM kernel's byte layout.

        ``create_weights`` allocates the checkpoint's ``[E, N, K//8]`` int32 so
        the expert loader (and its TP narrow) fills it directly; the kernel
        wants ``[E, N, K//2]`` uint8 instead. One repack here, on the load
        device, exactly the role ``awq_marlin_repack`` plays in vLLM.
        """
        from ...kernels import repack_int4_experts

        for name in ("gate_up_proj", "down_proj"):
            block.experts[name] = RawParameter(repack_int4_experts(block.experts[name].data))

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        from ...kernels import fused_moe

        config: AWQConfig = block.quant  # type: ignore[assignment]
        return fused_moe(
            x,
            block.experts["gate_up_proj"],
            block.experts["down_proj"],
            topk_weights,
            topk_ids,
            w1_scale=block.experts["gate_up_proj_scale"],
            w2_scale=block.experts["down_proj_scale"],
            w1_zeros=block.experts["gate_up_proj_zeros"],
            w2_zeros=block.experts["down_proj_zeros"],
            group_n=1,
            group_k=config.group_k,
        )
