"""Quantisation sub-package: config, parameter factories, and quant methods.

Layout::

    quantization/
    ├── config.py       # QuantConfig, format constants, checkpoint/runtime registries
    ├── parameter.py    # RawParameter (loader must not cast to fp16)
    ├── params/         # Parameter factory per precision (int8 / int4 / fp8)
    ├── methods/        # Quant-method strategies per format (vLLM-style)
    └── _layout/        # Weight layout rearrangement per backend (private)

Import the public API from here:
    ``from lite_llama.models.quantization import QuantConfig, RawParameter``
"""

from .config import (
    FP8,
    FP8_BLOCK,
    INT4,
    INT8,
    SCALE_SUFFIX,
    SMOOTHQUANT,
    RUNTIME_SCHEMES,
    QuantConfig,
    register_quant_method,
)
from .methods import get_linear_method, get_moe_method
from .parameter import RawParameter
from .params import (
    quantize_fp8_per_channel,
    quantize_int4_groupwise,
    quantize_int8_groupwise,
    quantize_int8_per_channel,
)

__all__ = [
    "FP8",
    "FP8_BLOCK",
    "INT4",
    "INT8",
    "SCALE_SUFFIX",
    "SMOOTHQUANT",
    "RUNTIME_SCHEMES",
    "QuantConfig",
    "RawParameter",
    "get_linear_method",
    "get_moe_method",
    "quantize_fp8_per_channel",
    "quantize_int4_groupwise",
    "quantize_int8_groupwise",
    "quantize_int8_per_channel",
    "register_quant_method",
]
