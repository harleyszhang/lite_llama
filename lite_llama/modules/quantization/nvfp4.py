"""NVFP4 config and method: 4-bit e2m1 weights with two levels of scale.

The format is NVIDIA ModelOpt's ``modelopt_fp4`` / TensorRT-LLM's NVFP4, and the
two-level scale is what separates it from the int4 configs next door:

* ``weight``       ``[N, K // 2]`` uint8 — two e2m1 nibbles per byte, low nibble
  at the even k index;
* ``weight_scale`` ``[N, K // 16]`` uint8 — one fp8-e4m3 bit pattern per 16
  consecutive k elements;
* ``weight_global_scale`` one fp32 element — brings the block scales themselves
  into e4m3's range.

**Weight-only, and permanently so on this hardware.** sm90 has no fp4 MMA, so
the activations stay 16-bit and the return is bytes rather than FLOPs: 4.5 bits
per weight against 16, which shows up as lower decode latency and a smaller
resident model, not as higher TFLOP/s. See
:mod:`lite_llama.kernels.ops.quantization.nvfp4` for why that is a property of
the device and not of the kernel.

MoE experts are **not** implemented: the fused grouped GEMM would need its own
two-level unpacking path, which is a separate kernel rather than a flag.
:meth:`NVFP4Config.get_quant_method` says so rather than silently handing a MoE
block a linear method it cannot use.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
    run_quant_linear,
)
from .parameter import RawParameter


def _nvfp4_block() -> int:
    """The format's block length, read from the kernel layer on first use.

    Deferred because importing any ``kernels`` submodule registers every spec
    row as a side effect, and ``tests/test_imports.py`` pins ``lite_llama.modules``
    to import without touching that registry.
    """
    from ...kernels.ops.quantization import NVFP4_BLOCK

    return NVFP4_BLOCK

#: Weight elements per byte.
_PACK_FACTOR = 2

#: Smallest k-shard that splits neither a byte nor a block scale, i.e.
#: ``lcm(2, 16)``. See :meth:`NVFP4Config.shard_is_aligned`.
_SHARD_GRANULARITY = 16


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class NVFP4Config(QuantizationConfig):
    """NVFP4 checkpoint/runtime config: e2m1 weights, e4m3 block scales.

    ``group_k`` is 16 and not configurable. Unlike AWQ, where the group size is
    a checkpoint choice, NVFP4's block length is part of the format — a
    different block length is a different format, so there is nothing here for
    ``from_config`` to read out of the checkpoint.
    """

    def __init__(self, ignored: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.group_n = 1
        self.group_k = _nvfp4_block()
        self.ignored = ignored
        self.method = "nvfp4"

    def get_name(self) -> str:
        return "nvfp4"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # Weight-only: the activation never leaves its own dtype, so both
        # 16-bit types run through the one kernel.
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # 89 (Ada) is *not* required by fp4 — nothing here uses an fp8 or fp4
        # tensor-core instruction. It is the floor the e4m3 block-scale decode
        # inherits from the shared w8a16 bit trick, and 80 would very likely
        # work; keeping it at 89 means no untested capability claim ships.
        return 89

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NVFP4Config:
        bits = int(config.get("bits", 4))
        if bits != 4:
            raise ValueError(f"only 4-bit NVFP4 is supported, got {bits}")
        group_size = int(config.get("group_size", _nvfp4_block()))
        if group_size != _nvfp4_block():
            raise ValueError(
                f"NVFP4 block size is fixed at {_nvfp4_block()} by the format, "
                f"checkpoint declares {group_size}"
            )
        ignored = tuple(config.get("modules_to_not_convert") or ())
        return cls(ignored)

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        from ..moe import SparseMoeBlock
        from .unquant import UnquantizedFusedMoEMethod

        if isinstance(layer, SparseMoeBlock) and self.quantizes(prefix):
            raise NotImplementedError(
                "NVFP4 MoE experts are not implemented; the fused grouped GEMM "
                "has no two-level unpacking path. Use --quantization fp8 or int4 "
                f"for MoE models, or add {prefix!r} to modules_to_not_convert."
            )
        # UnquantizedFusedMoEMethod is only reachable for an *ignored* MoE
        # prefix, which the branch above deliberately lets through.
        return self._dispatch(layer, prefix, NVFP4LinearMethod, UnquantizedFusedMoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.uint8

    def shard_is_aligned(self, size: int) -> bool:
        """Whether a TP shard of ``size`` channels keeps whole bytes and blocks.

        Two constraints stack on the k axis, where both the packing and the
        block scales live: a shard must contain whole bytes (2 elements) and
        whole block scales (16). Their lcm is 16, so 16 is the granularity —
        *not* 32. Multiplying the two would reject k=4864, which is exactly the
        ``down_proj`` shard Qwen3-4B gets under TP2.

        The n axis needs no constraint at all: nothing is packed along it.
        """
        return size % _SHARD_GRANULARITY == 0


# --------------------------------------------------------------------------- #
# Linear method
# --------------------------------------------------------------------------- #
class NVFP4LinearMethod(LinearMethodBase):
    """NVFP4 weight-only linear; runs ``native/linear_nvfp4``."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        block = _nvfp4_block()
        if input_size % block != 0:
            raise ValueError(
                f"NVFP4 needs in_features divisible by {block}, got {input_size}"
            )
        config: NVFP4Config = layer.quant  # type: ignore[assignment]
        layer.weight = RawParameter(
            torch.empty(output_size, input_size // _PACK_FACTOR, dtype=torch.uint8)
        )
        # uint8 rather than float8_e4m3fn: the kernel bit-shifts these bytes, and
        # RawParameter exists precisely so the loader does not helpfully cast
        # them to the activation dtype on the way in.
        layer.weight_scale = RawParameter(
            torch.empty(*config.scale_shape(output_size, input_size), dtype=torch.uint8)
        )
        layer.weight_global_scale = RawParameter(torch.empty(1, dtype=torch.float32))

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return run_quant_linear(
            "nvfp4",
            x,
            layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            bias=bias,
        )

    def quantize_from_fp16(self, layer: nn.Module, config: QuantizationConfig) -> None:
        from ...kernels.ops.quantization import quantize_nvfp4_blockwise

        packed, block_scale, global_scale = quantize_nvfp4_blockwise(layer.weight.data)
        layer.weight = RawParameter(packed)
        layer.weight_scale = RawParameter(block_scale)
        layer.weight_global_scale = RawParameter(global_scale)
