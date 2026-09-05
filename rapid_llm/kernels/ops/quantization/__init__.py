"""Quantisation GEMM kernels: one Triton kernel per numeric format.

Re-exports the five matmul entry points — fp8, nvfp4, w8a16, w4a16 and
smoothquant int8, each the registered implementation of the ``linear`` op
under its scheme — plus the one-shot word→byte repack helpers, the
per-token-group activation quantiser and the scale-grid layout descriptors it
allocates to, so the quant methods can pull them from the package root.

Usage:
    from rapid_llm.kernels.ops.quantization import fp8_matmul

``nvfp4`` is weight-only despite the narrow weight: sm90 has no fp4 MMA, so
its return is bytes, not FLOPs.
"""

from importlib import import_module

from .scale_layout import (
    COLUMN_MAJOR,
    COLUMN_MAJOR_TMA,
    ROW_MAJOR,
    TMA_SCALE_ALIGNMENT,
    ScaleLayout,
    create_scale_output,
    infer_scale_layout,
)

_MODULES = {
    "per_token_group_quant": "activation",
    "fp8_matmul": "fp8",
    "fp8_quantize_per_token": "fp8",
    "repack_int4_experts": "int4_repack",
    "unpack_int8_experts": "int8_repack",
    "NVFP4_BLOCK": "nvfp4",
    "nvfp4_matmul": "nvfp4",
    "quantize_nvfp4_blockwise": "nvfp4",
    "w4a16_matmul": "w4a16",
    "int8_quantize_per_token": "w8a8",
    "smoothquant_matmul": "w8a8",
    "w8a16_matmul": "w8a16",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "COLUMN_MAJOR",
    "COLUMN_MAJOR_TMA",
    "NVFP4_BLOCK",
    "ROW_MAJOR",
    "TMA_SCALE_ALIGNMENT",
    "ScaleLayout",
    "create_scale_output",
    "fp8_matmul",
    "fp8_quantize_per_token",
    "infer_scale_layout",
    "int8_quantize_per_token",
    "nvfp4_matmul",
    "per_token_group_quant",
    "quantize_nvfp4_blockwise",
    "repack_int4_experts",
    "smoothquant_matmul",
    "unpack_int8_experts",
    "w4a16_matmul",
    "w8a16_matmul",
]
