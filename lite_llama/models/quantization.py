"""Quantisation: what the weights look like in memory, and how to get there.

One module, multiple schemes. All of them end up in the same shape — 8-bit (or
4-bit) weight plus a grid of scales — which is what lets the w8a16 / w4a16
kernels and the MoE grouped GEMM serve them with one kernel each. Activations
are never quantised in the weight-only schemes: "a16" is the whole point, since
the error budget of an fp16 activation is what keeps the output faithful.

Supported schemes:
    * **fp8** — block-wise fp8-e4m3 with one scale per 128×128 block. Qwen and
      DeepSeek FP8 checkpoints ship this format ready-made.
    * **int8** — symmetric per-output-channel int8, computed at load time from
      an fp16 checkpoint for models that ship no FP8 variant.
    * **awq** / **gptq** — group-wise int4 (W4A16), loaded from pre-quantised
      checkpoints. AWQ uses per-group scales; GPTQ uses a different packing
      order. Both share the same w4a16 kernel at inference time.
    * **smoothquant** — dynamic per-token activation quantisation + per-channel
      weight quantisation (W8A8). The weights are int8 with per-channel scales,
      and the activations are quantised on the fly with per-token scales.

Usage:
    quant = QuantConfig.from_hf(hf_config)          # fp8 checkpoints
    quant = QuantConfig.int8_per_channel()           # --quantization int8
    quant = QuantConfig.from_hf(gptq_config)         # GPTQ/AWQ checkpoints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Format constants
# --------------------------------------------------------------------------- #
FP8 = "fp8"
INT8 = "int8"
INT4 = "int4"       # AWQ / GPTQ
SMOOTHQUANT = "smoothquant"

#: Block size of the fine-grained FP8 format used by Qwen/DeepSeek checkpoints.
FP8_BLOCK = 128

#: Checkpoint suffix of the per-block dequantisation scale.
SCALE_SUFFIX = "weight_scale_inv"

#: Registry: quant_method string in config.json -> QuantConfig factory.
_REGISTRY: dict[str, str] = {
    "fp8": FP8,
    "int8": INT8,
    "gptq": INT4,
    "awq": INT4,
    "smoothquant": SMOOTHQUANT,
}


def register_quant_method(name: str, format_id: str) -> None:
    """Register a new ``quant_method`` name -> format mapping.

    This lets the config parser recognise checkpoint formats that are not in
    the built-in table without forking the module.
    """
    _REGISTRY[name.lower()] = format_id


# --------------------------------------------------------------------------- #
# RawParameter: marker for "do not cast this"
# --------------------------------------------------------------------------- #
class RawParameter(nn.Parameter):
    """A parameter the loader must leave alone instead of casting to fp16.

    :func:`lite_llama.executor.loader.materialise_parameters` gives every
    floating-point parameter fp16 storage, which is right for weights and wrong
    for the two things quantisation adds: the 8-bit weight itself (``uint8`` /
    ``int8``, so it is not floating point anyway) and its fp32 scales, whose
    dynamic range is the reason the fp8 format works at all.
    """

    def __new__(cls, data: torch.Tensor, requires_grad: bool = False) -> RawParameter:
        return super().__new__(cls, data, requires_grad=requires_grad)


# --------------------------------------------------------------------------- #
# QuantConfig
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuantConfig:
    """How the quantised weights of one model are laid out.

    A scale covers a ``group_n x group_k`` block of the ``[out_features,
    in_features]`` weight, which describes all supported schemes:

    * fp8: ``128×128`` blocks (one scale per block).
    * int8 per-channel: ``group_n=1, group_k=K`` (one scale per output row).
    * int4 (AWQ/GPTQ): ``1×group_size`` (one scale per group of input channels).
    * smoothquant: weights are ``group_n=1, group_k=K`` (per-channel).

    Attributes:
        format: One of :data:`FP8`, :data:`INT8`, :data:`INT4`, :data:`SMOOTHQUANT`.
        group_n: Output channels covered by one scale.
        group_k: Input channels covered by one scale.
        ignored: HF module names the checkpoint left unquantised.
        is_dynamic: Whether activations are quantised at runtime (smoothquant).
    """

    format: str
    group_n: int
    group_k: int
    ignored: tuple[str, ...] = ()
    is_dynamic: bool = False

    # ---- construction ----------------------------------------------------- #
    @classmethod
    def from_hf(cls, hf_config: Any) -> QuantConfig | None:
        """Read ``config.json``'s ``quantization_config``, or ``None`` if absent.

        Raises:
            ValueError: If the checkpoint declares a scheme lite_llama cannot
                serve, which is better than silently producing garbage logits.
        """
        raw = getattr(hf_config, "quantization_config", None)
        if not raw:
            return None
        params = raw if isinstance(raw, dict) else raw.to_dict()

        method = str(params.get("quant_method", "")).lower()
        fmt = _REGISTRY.get(method)
        if fmt is None:
            raise ValueError(
                f"unsupported quant_method {method!r}; supported: {sorted(_REGISTRY)}"
            )

        ignored = tuple(params.get("modules_to_not_convert") or ())

        if fmt == FP8:
            return cls._from_fp8(params, ignored)
        if fmt == INT4:
            return cls._from_int4(params, method, ignored)
        if fmt == SMOOTHQUANT:
            return cls._from_smoothquant(params, ignored)
        # INT8 runtime quantisation has no checkpoint config.
        return None

    @classmethod
    def _from_fp8(cls, params: dict, ignored: tuple[str, ...]) -> QuantConfig:
        fmt = str(params.get("fmt", "e4m3")).lower()
        if fmt != "e4m3":
            raise ValueError(f"unsupported fp8 format {fmt!r}; only e4m3 is implemented")
        block = params.get("weight_block_size") or [FP8_BLOCK, FP8_BLOCK]
        gn, gk = int(block[0]), int(block[1])
        if gk % FP8_BLOCK != 0 or gn % FP8_BLOCK != 0:
            raise ValueError(
                f"weight_block_size {block} is not a multiple of {FP8_BLOCK}; "
                "the w8a16 kernel tiles k in 128-wide steps"
            )
        return cls(FP8, gn, gk, ignored)

    @classmethod
    def _from_int4(cls, params: dict, method: str, ignored: tuple[str, ...]) -> QuantConfig:
        # AWQ and GPTQ both use group-wise int4 with a configurable group size.
        group_size = int(params.get("group_size", 128))
        if group_size <= 0 or (group_size & (group_size - 1)) != 0:
            raise ValueError(f"group_size must be a positive power of 2, got {group_size}")
        return cls(INT4, group_n=1, group_k=group_size, ignored=ignored)

    @classmethod
    def _from_smoothquant(cls, params: dict, ignored: tuple[str, ...]) -> QuantConfig:
        return cls(SMOOTHQUANT, group_n=1, group_k=1 << 30, ignored=(), is_dynamic=True)

    @classmethod
    def int8_per_channel(cls) -> QuantConfig:
        """Symmetric int8, one scale per output channel, computed at load time."""
        return cls(INT8, group_n=1, group_k=1 << 30)

    # ---- layout ----------------------------------------------------------- #
    @property
    def storage_dtype(self) -> torch.dtype:
        """Container dtype of the packed weight."""
        if self.format == FP8:
            return torch.uint8
        if self.format == INT4:
            return torch.int32  # AWQ/GPTQ pack 8 int4 values per int32
        return torch.int8  # INT8, SMOOTHQUANT

    @property
    def is_fp8(self) -> bool:
        return self.format == FP8

    @property
    def is_int4(self) -> bool:
        return self.format == INT4

    def scale_shape(self, out_features: int, in_features: int) -> tuple[int, ...]:
        """Scale-grid shape for a ``[out_features, in_features]`` weight."""
        if self.format == INT4:
            # One scale per group of ``group_k`` input channels.
            return (out_features, (in_features + self.group_k - 1) // self.group_k)
        return (
            (out_features + self.group_n - 1) // self.group_n,
            (in_features + self.group_k - 1) // self.group_k,
        )

    def quantizes(self, module_name: str) -> bool:
        """Whether ``module_name`` (an HF-style path) is quantised."""
        return not any(
            module_name == ignored or module_name.startswith(ignored + ".")
            for ignored in self.ignored
        )

    def shard_is_aligned(self, size: int) -> bool:
        """Whether a TP shard of ``size`` channels keeps whole scale blocks."""
        if self.format == INT4:
            return size % self.group_k == 0
        return size % max(self.group_n, self.group_k) == 0 or self.format in (INT8, SMOOTHQUANT)


# --------------------------------------------------------------------------- #
# Quantisation utilities
# --------------------------------------------------------------------------- #
def quantize_int8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``[N, K]`` fp16 weights to symmetric per-channel int8.

    The scale of row ``n`` is ``max|W[n]| / 127``, so the largest magnitude in
    each output channel maps onto the end of the int8 range.

    Args:
        weight: ``[N, K]`` (or ``[E, N, K]`` for stacked experts) float weights.

    Returns:
        ``(qweight, scales)`` with ``scales`` shaped ``[..., N, 1]``.
    """
    scale = weight.abs().amax(dim=-1, keepdim=True).float() / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    qweight = (weight.float() / scale).round().clamp_(-127, 127).to(torch.int8)
    return qweight, scale


def quantize_int4_groupwise(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise ``[N, K]`` fp16 weights to group-wise int4 (AWQ/GPTQ format).

    Each group of ``group_size`` input channels gets its own fp32 scale and zero
    point. The packed output stores 8 int4 values per int32 word.

    Args:
        weight: ``[N, K]`` float weights. K must be a multiple of ``group_size``.
        group_size: Number of input channels per quantisation group.

    Returns:
        ``(qweight, scales, zeros)`` where ``qweight`` is ``[N, K//8]`` int32,
        ``scales`` is ``[N, K//group_size]`` fp32, and ``zeros`` is the same
        shape as ``scales``.
    """
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"in_features {k} must be a multiple of group_size {group_size}")

    w = weight.float().reshape(n, k // group_size, group_size)
    w_min = w.amin(dim=-1)
    w_max = w.amax(dim=-1)

    # Symmetric quantisation: use max(|min|, |max|) as the range.
    qmax = 7.0  # int4 range: [-8, 7]
    scale = (w_max - w_min).clamp(min=1e-5) / (2 * qmax)
    # Centre the zero point so that zero maps to the middle of the int4 range.
    zero = (-w_min / scale).round().clamp(0, 15)
    q = (w / scale.unsqueeze(-1) + zero.unsqueeze(-1)).round().clamp(0, 15).to(torch.int32)

    # Pack 8 int4 values per int32 word along the K dimension.
    q = q.reshape(n, -1, 8)
    shifts = torch.arange(8, device=q.device, dtype=torch.int32) * 4
    packed = (q << shifts[None, None, :]).sum(dim=-1)  # [N, K//8]

    return packed.to(torch.int32), scale.float(), zero.float()
