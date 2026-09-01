"""Quantisation sub-package: config classes, method classes, layout adapters.

``get_quantization_config`` builds a config from a HF config, and
``for_runtime_scheme`` maps a runtime scheme name (``"w8a8_int8"``) to
its config — the two doors the rest of the codebase uses.

Usage:
    quant = for_runtime_scheme("w8a8_int8")
"""

from __future__ import annotations

from typing import Any

from .awq import AWQConfig
from .base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from .blockwise_int8 import BlockInt8Config
from .fp8 import FP8_BLOCK, Fp8Config
from .gptq import GPTQConfig
from .kv_cache import BaseKVCacheMethod, Fp8KVCacheMethod, get_kv_cache_method
from .parameter import RawParameter
from .unquant import UnquantizedConfig, UnquantizedFusedMoEMethod, UnquantizedLinearMethod
from .utils import adapt_int4_checkpoint
from .w8a8_fp8 import W8A8Fp8Config
from .w8a8_int8 import W8A8Int8Config

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Checkpoint suffix of the per-block dequantisation scale.
SCALE_SUFFIX = "weight_scale_inv"

# --------------------------------------------------------------------------- #
# Registry (mirrors sglang BASE_QUANTIZATION_METHODS)
# --------------------------------------------------------------------------- #
BASE_QUANTIZATION_METHODS: dict[str, type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "w8a8_int8": W8A8Int8Config,
    "blockwise_int8": BlockInt8Config,
    "awq": AWQConfig,
    "gptq": GPTQConfig,
    # Aliases
    "int8": BlockInt8Config,
    "smoothquant": W8A8Int8Config,
}

#: Runtime quantisation schemes accepted by ``--quantization``.
RUNTIME_SCHEMES: dict[str, type[QuantizationConfig]] = {
    "int8": BlockInt8Config,
    "int8-blockwise": BlockInt8Config,
    "fp8": W8A8Fp8Config,
    "int4": AWQConfig,
    "smoothquant": W8A8Int8Config,
}


def get_quantization_config(name: str) -> type[QuantizationConfig]:
    """Look up a QuantizationConfig class by name."""
    cls = BASE_QUANTIZATION_METHODS.get(name.lower())
    if cls is None:
        raise ValueError(
            f"unknown quantisation method {name!r}; supported: {sorted(BASE_QUANTIZATION_METHODS)}"
        )
    return cls


def get_quant_config_from_hf(hf_config: Any) -> QuantizationConfig | None:
    """Parse ``config.json``'s ``quantization_config`` into a QuantizationConfig.

    Returns None if the checkpoint is unquantised.
    Raises ValueError for unsupported formats.
    """
    raw = getattr(hf_config, "quantization_config", None)
    if not raw:
        return None
    params = raw if isinstance(raw, dict) else raw.to_dict()

    method_name = str(params.get("quant_method", "")).lower()
    cls = BASE_QUANTIZATION_METHODS.get(method_name)
    if cls is None:
        raise ValueError(
            f"unsupported quant_method {method_name!r}; "
            f"supported: {sorted(BASE_QUANTIZATION_METHODS)}"
        )
    return cls.from_config(params)


def for_runtime_scheme(name: str) -> QuantizationConfig:
    """Build a QuantizationConfig for ``--quantization <name>``.

    Raises ValueError on an unrecognised scheme name.
    """
    cls = RUNTIME_SCHEMES.get(name.lower())
    if cls is None:
        raise ValueError(
            f"unknown runtime quantisation {name!r}; supported: {sorted(RUNTIME_SCHEMES)}"
        )
    # Each config has sensible defaults for its runtime variant.
    if cls is BlockInt8Config:
        if name.lower() == "int8-blockwise":
            return BlockInt8Config.groupwise()
        return BlockInt8Config.per_channel()
    if cls is W8A8Fp8Config:
        return W8A8Fp8Config()
    if cls is W8A8Int8Config:
        return W8A8Int8Config()
    if cls is AWQConfig:
        return AWQConfig()
    # Fallback (shouldn't reach here).
    return cls.from_config({})


# Grouped by role rather than sorted: the groups say what each name is for,
# which a flat alphabetical list cannot.
__all__ = [  # noqa: RUF022
    # ABC
    "QuantizationConfig",
    "QuantizeMethodBase",
    "LinearMethodBase",
    "FusedMoEMethodBase",
    # Configs
    "Fp8Config",
    "W8A8Fp8Config",
    "W8A8Int8Config",
    "BlockInt8Config",
    "AWQConfig",
    "GPTQConfig",
    "UnquantizedConfig",
    # KV cache
    "BaseKVCacheMethod",
    "Fp8KVCacheMethod",
    "get_kv_cache_method",
    # Methods (convenience re-exports)
    "UnquantizedLinearMethod",
    "UnquantizedFusedMoEMethod",
    # Infrastructure
    "RawParameter",
    "FP8_BLOCK",
    "SCALE_SUFFIX",
    "BASE_QUANTIZATION_METHODS",
    "RUNTIME_SCHEMES",
    "adapt_int4_checkpoint",
    # Factories
    "get_quantization_config",
    "get_quant_config_from_hf",
    "for_runtime_scheme",
]
