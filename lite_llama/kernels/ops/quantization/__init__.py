"""Quantisation GEMM kernels: one Triton kernel per numeric format.

Re-exports the five matmul entry points — fp8, nvfp4, w8a16, w4a16
and smoothquant int8 — each the registered implementation of the
``linear`` op under its scheme.

Usage:
    from lite_llama.kernels.ops.quantization import fp8_matmul

``nvfp4`` breaks the naming convention's implied progression: it is
weight-only despite the narrow weight, because sm90 has no fp4 MMA —
its return is bytes, not FLOPs.
"""

from .fp8 import fp8_matmul, fp8_quantize_per_token
from .nvfp4 import NVFP4_BLOCK, nvfp4_matmul, quantize_nvfp4_blockwise
from .w4a16 import w4a16_matmul
from .w8a8 import int8_quantize_per_token, smoothquant_matmul
from .w8a16 import w8a16_matmul

__all__ = [
    "NVFP4_BLOCK",
    "fp8_matmul",
    "fp8_quantize_per_token",
    "int8_quantize_per_token",
    "nvfp4_matmul",
    "quantize_nvfp4_blockwise",
    "smoothquant_matmul",
    "w4a16_matmul",
    "w8a16_matmul",
]
