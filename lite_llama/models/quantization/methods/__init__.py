"""Quantisation methods: one strategy class per weight format (vLLM-style).

The Linear/MoE modules own tensor-parallel sharding and routing; the method
object returned here owns the storage format. Adding a scheme is one file in
this package plus one line in the registries below — the layers do not change.

Usage:
    method = get_linear_method(quant)   # None -> UnquantizedLinearMethod
    method.create_weights(layer, input_size, output_size)
"""

from __future__ import annotations

from ..config import FP8, INT4, INT8, SMOOTHQUANT, QuantConfig
from .auto_awq import AWQLinearMethod
from .auto_gptq import GPTQLinearMethod
from .base import LinearQuantMethod, MoeQuantMethod
from .fp8 import Fp8LinearMethod
from .unquantized import UnquantizedLinearMethod, UnquantizedMoeMethod
from .w4a16 import W4A16LinearMethod
from .w8a16 import W8A16LinearMethod, W8A16MoeMethod
from .w8a8 import SmoothQuantLinearMethod

_LINEAR_METHODS: dict[str, type[LinearQuantMethod]] = {
    FP8: Fp8LinearMethod,
    INT8: W8A16LinearMethod,
    SMOOTHQUANT: SmoothQuantLinearMethod,
    INT4: W4A16LinearMethod,
}

#: Int4 checkpoints keep their producer's name so the loader can pick the
#: matching layout adapter; both run the shared w4a16 kernel afterwards.
_INT4_CHECKPOINT_METHODS: dict[str, type[LinearQuantMethod]] = {
    "awq": AWQLinearMethod,
    "gptq": GPTQLinearMethod,
}

# The fused MoE kernel consumes fp16 activations, so fp8/int8 experts stay
# weight-only (w8a16) even though the linear layers of the same model run
# true W8A8. INT4 experts are rejected: there is no grouped int4 GEMM kernel,
# and a silent fallback would produce wrong logits.
_MOE_METHODS: dict[str, type[MoeQuantMethod]] = {
    FP8: W8A16MoeMethod,
    INT8: W8A16MoeMethod,
    SMOOTHQUANT: W8A16MoeMethod,
}


def get_linear_method(quant: QuantConfig | None) -> LinearQuantMethod:
    """Return the strategy implementing ``quant``'s storage format."""
    if quant is None:
        return UnquantizedLinearMethod()
    cls = _LINEAR_METHODS.get(quant.format)
    if cls is None:
        raise ValueError(f"no linear quant method for format {quant.format!r}")
    if quant.format == INT4 and quant.method:
        cls = _INT4_CHECKPOINT_METHODS.get(quant.method, cls)
    return cls()


def get_moe_method(quant: QuantConfig | None) -> MoeQuantMethod:
    """Return the strategy implementing ``quant``'s format for stacked experts.

    Raises:
        ValueError: If the format has no grouped-GEMM kernel (int4).
    """
    if quant is None:
        return UnquantizedMoeMethod()
    cls = _MOE_METHODS.get(quant.format)
    if cls is None:
        raise ValueError(
            f"format {quant.format!r} is not supported for MoE experts; "
            "add the layer to modules_to_not_convert or use an 8-bit scheme"
        )
    return cls()


__all__ = [
    "AWQLinearMethod",
    "Fp8LinearMethod",
    "GPTQLinearMethod",
    "LinearQuantMethod",
    "MoeQuantMethod",
    "SmoothQuantLinearMethod",
    "UnquantizedLinearMethod",
    "UnquantizedMoeMethod",
    "W4A16LinearMethod",
    "W8A16LinearMethod",
    "W8A16MoeMethod",
    "get_linear_method",
    "get_moe_method",
]
