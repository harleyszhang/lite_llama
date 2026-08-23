"""Quantisation sub-package: sglang-aligned architecture.

Layout (mirrors sglang ``srt/layers/quantization/``)::

    quantization/
    ├── __init__.py          # Registry + public API
    ├── base_config.py       # QuantizeMethodBase / LinearMethodBase / FusedMoEMethodBase / QuantizationConfig ABC
    ├── fp8.py               # Fp8Config (weight-only fp8, block-wise scales)
    ├── w8a8_fp8.py          # W8A8Fp8Config (true W8A8: fp8 weights + per-token fp8 activations)
    ├── w8a8_int8.py         # W8A8Int8Config (SmoothQuant: int8 W8A8)
    ├── blockwise_int8.py    # BlockInt8Config (int8 weight-only, per-channel / group-wise)
    ├── awq.py               # AWQConfig (int4 AWQ checkpoints)
    ├── gptq.py              # GPTQConfig (int4 GPTQ checkpoints)
    ├── unquant.py           # UnquantizedConfig / Methods (fp16 default)
    ├── kv_cache.py          # BaseKVCacheMethod / Fp8KVCacheMethod
    ├── parameter.py         # RawParameter (loader must not cast to fp16)
    └── utils.py             # Quantise helpers + checkpoint layout adapters

Public API:
    from lite_llama.modules.quantization import (
        QuantizationConfig, QuantizeMethodBase, LinearMethodBase, FusedMoEMethodBase,
        BASE_QUANTIZATION_METHODS,
        get_quantization_config, get_quant_config_from_hf, for_runtime_scheme,
        RawParameter, adapt_int4_checkpoint,
    )
"""

from __future__ import annotations

from typing import Any, Type

from .awq import AWQConfig
from .base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from .blockwise_int8 import BlockInt8Config
from .fp8 import Fp8Config
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

#: Block size of the fine-grained FP8 format (Qwen/DeepSeek checkpoints).
FP8_BLOCK = 128

#: Checkpoint suffix of the per-block dequantisation scale.
SCALE_SUFFIX = "weight_scale_inv"

# --------------------------------------------------------------------------- #
# Registry (mirrors sglang BASE_QUANTIZATION_METHODS)
# --------------------------------------------------------------------------- #
BASE_QUANTIZATION_METHODS: dict[str, Type[QuantizationConfig]] = {
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
RUNTIME_SCHEMES: dict[str, Type[QuantizationConfig]] = {
    "int8": BlockInt8Config,
    "int8-blockwise": BlockInt8Config,
    "fp8": W8A8Fp8Config,
    "int4": AWQConfig,
    "smoothquant": W8A8Int8Config,
}


def get_quantization_config(name: str) -> Type[QuantizationConfig]:
    """Look up a QuantizationConfig class by name."""
    cls = BASE_QUANTIZATION_METHODS.get(name.lower())
    if cls is None:
        raise ValueError(
            f"unknown quantisation method {name!r}; "
            f"supported: {sorted(BASE_QUANTIZATION_METHODS)}"
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


__all__ = [
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
