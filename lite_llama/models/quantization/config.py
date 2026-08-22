"""Quantisation config: what a quantised weight looks like in memory.

Every scheme is reduced to one description — a low-bit weight plus a scale covering a
``group_n x group_k`` block of it — which is why a single w8a16 / w4a16 kernel serves
all of them: fp8-e4m3 in 128x128 blocks (Qwen/DeepSeek checkpoints), per-output-channel
or group-wise int8, group-wise int4 for AWQ/GPTQ, and SmoothQuant's per-channel int8
with activations quantised per token at runtime. Checkpoint schemes are parsed by
:meth:`QuantConfig.from_hf` and rejected loudly when unsupported; runtime schemes come
from ``--quantization`` and are computed after loading. ``_REGISTRY`` keeps the
``quant_method`` -> format mapping extensible without forking the parser.

Usage:
    quant = QuantConfig.from_hf(hf_config)              # fp8 / awq / gptq checkpoints
    quant = QuantConfig.for_runtime_scheme("int8")      # --quantization int8
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

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

#: Runtime quantisation schemes accepted by ``--quantization``.
RUNTIME_SCHEMES: dict[str, str] = {
    "int8": INT8,
    "int8-blockwise": INT8,
    "fp8": FP8,
    "int4": INT4,
    "smoothquant": SMOOTHQUANT,
}


def register_quant_method(name: str, format_id: str) -> None:
    """Register a new ``quant_method`` name -> format mapping.

    This lets the config parser recognise checkpoint formats that are not in
    the built-in table without forking the module.
    """
    _REGISTRY[name.lower()] = format_id


# --------------------------------------------------------------------------- #
# QuantConfig
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuantConfig:
    """How the quantised weights of one model are laid out.

    A scale covers a ``group_n x group_k`` block of the ``[out_features,
    in_features]`` weight, which describes all supported schemes:

    * fp8: ``128×128`` blocks (one scale per block), or per-channel
      (``group_n=1, group_k=K``) for the runtime ``--quantization fp8`` scheme.
    * int8 per-channel: ``group_n=1, group_k=K`` (one scale per output row).
    * int8 block-wise: ``group_n=1, group_k=group_size``.
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

    @classmethod
    def int8_groupwise(cls, group_size: int = 128) -> QuantConfig:
        """Symmetric int8 with one scale per ``group_size`` input channels.

        Finer granularity than per-channel: closer to fp16 accuracy at the same
        8-bit storage, at the price of a ``K / group_size``-wide scale grid.
        """
        return cls(INT8, group_n=1, group_k=group_size)

    @classmethod
    def fp8_per_channel(cls) -> QuantConfig:
        """fp8-e4m3 weights with one scale per output channel, computed at load time.

        Activations stay fp16 (W8A16), so no calibration data is needed.
        """
        return cls(FP8, group_n=1, group_k=1 << 30)

    @classmethod
    def int4_groupwise(cls, group_size: int = 128) -> QuantConfig:
        """Group-wise int4 (AWQ/GPTQ format), computed at load time."""
        return cls(INT4, group_n=1, group_k=group_size)

    @classmethod
    def smoothquant_per_channel(cls) -> QuantConfig:
        """SmoothQuant W8A8: per-channel int8 weights + dynamic per-token activations."""
        return cls(SMOOTHQUANT, group_n=1, group_k=1 << 30, is_dynamic=True)

    @classmethod
    def for_runtime_scheme(cls, name: str) -> QuantConfig:
        """Build a config for ``--quantization <name>``.

        Raises:
            ValueError: On an unrecognised scheme name.
        """
        fmt = RUNTIME_SCHEMES.get(name.lower())
        if fmt is None:
            raise ValueError(
                f"unknown runtime quantisation {name!r}; supported: {sorted(RUNTIME_SCHEMES)}"
            )
        if fmt == INT8:
            return cls.int8_groupwise() if name.lower() == "int8-blockwise" else cls.int8_per_channel()
        if fmt == FP8:
            return cls.fp8_per_channel()
        if fmt == INT4:
            return cls.int4_groupwise()
        if fmt == SMOOTHQUANT:
            return cls.smoothquant_per_channel()
        raise ValueError(f"no runtime factory for format {fmt!r}")

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
        # Per-channel schemes (one scale per output row) have no block to cut.
        if self.group_n <= 1 and self.group_k >= (1 << 30):
            return True
        if self.format == INT4:
            return size % self.group_k == 0
        return size % max(self.group_n, self.group_k) == 0
