"""GPTQ config and method (mirrors sglang ``gptq/gptq.py``).

:class:`GPTQConfig` carries the checkpoint's bits (4 or 8) and group size;
the linear/MoE methods load the packed word layout AutoGPTQ produces and
repack it in ``process_weights_after_loading`` into whatever their kernel
eats — byte-packed int4 for the fused MoE path, one int8 byte per element
for the dense w8a16 path.

Usage:
    quant = GPTQConfig(group_size, ignored, bits=8)
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
from .parameter import RawParameter
from .utils import quantize_int4_groupwise, quantize_int8_groupwise_asym

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class GPTQConfig(QuantizationConfig):
    """AutoGPTQ checkpoint config: group-wise int4/int8 with configurable group size.

    ``bits=4`` and ``bits=8`` share one container — AutoGPTQ packs 8 nibbles or
    4 bytes per int32 word — and one scale/zero grid; only the pack factor and
    the kernel the methods route to differ.
    """

    def __init__(self, group_size: int = 128, ignored: tuple[str, ...] = (), bits: int = 4) -> None:
        super().__init__()
        self.group_n = 1
        self.group_k = group_size
        self.ignored = ignored
        self.bits = bits
        self.method = "gptq"

    def get_name(self) -> str:
        return "gptq"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # w4a16 unpacks to fp32 and casts to the activation dtype, so bf16
        # activations run through the same kernel.
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GPTQConfig:
        bits = int(config.get("bits", 4))
        if bits not in (4, 8):
            raise ValueError(f"only 4- and 8-bit GPTQ are supported, got {bits}")
        if config.get("desc_act"):
            raise ValueError(
                "GPTQ with desc_act (activation ordering) is not supported; "
                "repack the checkpoint with desc_act=False"
            )
        group_size = int(config.get("group_size", 128))
        if group_size <= 0 or (group_size & (group_size - 1)) != 0:
            raise ValueError(f"group_size must be a positive power of 2, got {group_size}")
        ignored = tuple(config.get("modules_to_not_convert") or ())
        return cls(group_size, ignored, bits=bits)

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        return self._dispatch(layer, prefix, GPTQLinearMethod, GPTQMoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.int32

    @property
    def pack_factor(self) -> int:
        """Quantised values per int32 storage word (8 nibbles or 4 bytes)."""
        return 32 // self.bits

    @property
    def is_int4(self) -> bool:
        return self.bits == 4

    @property
    def is_packed(self) -> bool:
        # bits=8 packs four bytes per int32 word, the same bridge as int4.
        return True


class GPTQLinearMethod(LinearMethodBase):
    """Group-wise int4/int8 from an AutoGPTQ checkpoint; runs the w4a16 kernel.

    Both bit widths load as ``[N, K//pack_factor]`` int32 words. The int4 rows
    feed the w4a16 kernel in exactly that form; the int8 rows instead leave
    :meth:`process_weights_after_loading` as ``[N, K]`` int8 bytes — the layout
    the w8a16 kernel (under the ``gptq_int8`` scheme) and every other int8
    consumer shares.
    """

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        config: GPTQConfig = layer.quant  # type: ignore[assignment]
        packed_k = (input_size + config.pack_factor - 1) // config.pack_factor
        layer.weight = RawParameter(torch.empty(output_size, packed_k, dtype=torch.int32))
        layer.weight_scale = RawParameter(
            torch.empty(*config.scale_shape(output_size, input_size), dtype=torch.float32)
        )
        layer.weight_zeros = RawParameter(
            torch.empty(*config.scale_shape(output_size, input_size), dtype=torch.float32)
        )

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        config: GPTQConfig = layer.quant  # type: ignore[assignment]
        return run_quant_linear(
            "gptq" if config.bits == 4 else "gptq_int8",
            x,
            layer.weight,
            weight_scale=layer.weight_scale,
            weight_zeros=layer.weight_zeros,
            group_k=layer.quant.group_k,
            bias=bias,
        )

    def quantize_from_fp16(self, layer: nn.Module, config: QuantizationConfig) -> None:
        cfg: GPTQConfig = config  # type: ignore[assignment]
        quantize = quantize_int4_groupwise if cfg.bits == 4 else quantize_int8_groupwise_asym
        qweight, scales, zeros = quantize(layer.weight.data, cfg.group_k)
        layer.weight = RawParameter(qweight)
        layer.weight_scale = RawParameter(scales)
        layer.weight_zeros = RawParameter(zeros)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Expand the int8 word packing to the bytes the w8a16 kernel eats.

        int4 rows load in the dense kernel's layout already (packed int32 words)
        and return immediately; int8 rows cross the same bridge the MoE method
        does, just ending at ``[N, K]`` int8 because that is what the dense
        w8a16 kernel takes rather than a stacked-expert variant.
        """
        config: GPTQConfig = layer.quant  # type: ignore[assignment]
        if config.bits == 8:
            from ...kernels import unpack_int8_experts

            layer.weight = RawParameter(unpack_int8_experts(layer.weight.data))


class GPTQMoEMethod(FusedMoEMethodBase):
    """GPTQ int4/int8 MoE: stacked experts through the fused kernel.

    Expert weights load in the checkpoint's ``[E, N, K//pack_factor]`` int32
    packing (8 nibbles or 4 bytes per word) with ``[E, N, K//group_k]`` fp32
    scales and zeros; :meth:`process_weights_after_loading` then swaps each
    stacked tensor for the fused kernel's per-element container — ``[E, N,
    K//2]`` uint8 for int4 (two nibbles per byte), ``[E, N, K]`` int8 for the
    asymmetric bits=8 mode — in one repack.
    """

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        config: GPTQConfig = block.quant  # type: ignore[assignment]
        gate_up_n = 2 * block.moe_intermediate_size
        gate_up_k = block.hidden_size
        down_n = block.hidden_size
        down_k = block.moe_intermediate_size
        pack_factor = config.pack_factor  # values per int32 word
        num_groups_gu = (gate_up_k + config.group_k - 1) // config.group_k
        num_groups_d = (down_k + config.group_k - 1) // config.group_k
        return {
            "gate_up_proj": RawParameter(
                torch.empty(
                    block.num_experts, gate_up_n, gate_up_k // pack_factor, dtype=torch.int32
                )
            ),
            "gate_up_proj_scale": RawParameter(
                torch.empty(block.num_experts, gate_up_n, num_groups_gu, dtype=torch.float32)
            ),
            "gate_up_proj_zeros": RawParameter(
                torch.empty(block.num_experts, gate_up_n, num_groups_gu, dtype=torch.float32)
            ),
            "down_proj": RawParameter(
                torch.empty(block.num_experts, down_n, down_k // pack_factor, dtype=torch.int32)
            ),
            "down_proj_scale": RawParameter(
                torch.empty(block.num_experts, down_n, num_groups_d, dtype=torch.float32)
            ),
            "down_proj_zeros": RawParameter(
                torch.empty(block.num_experts, down_n, num_groups_d, dtype=torch.float32)
            ),
        }

    def process_weights_after_loading(self, block: nn.Module) -> None:
        """Repack the int32 word layout into the fused kernel's per-element one.

        ``create_weights`` allocates the checkpoint's ``[E, N, K//pack_factor]``
        int32 so the expert loader (and its TP narrow) fills it directly; the
        fused MoE kernel instead wants one byte per nibble pair (int4, ``[E, N,
        K//2]`` uint8 — vLLM's layout, whose replicated addressing this cannot
        pay per call) or one byte per value (int8, ``[E, N, K]`` int8, the
        asymmetric mode the kernel's zeros branch dequantises). One repack
        here, on the load device, exactly the role ``awq_marlin_repack`` plays
        in vLLM's same-named hook.
        """
        from ...kernels import repack_int4_experts, unpack_int8_experts

        repack = repack_int4_experts if block.quant.bits == 4 else unpack_int8_experts
        for name in ("gate_up_proj", "down_proj"):
            block.experts[name] = RawParameter(repack(block.experts[name].data))

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        from ...kernels import fused_moe

        config: GPTQConfig = block.quant  # type: ignore[assignment]
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
