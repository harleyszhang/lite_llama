"""Quantisation GEMM kernels: one Triton kernel per numeric format.

Re-exports the four matmul entry points — fp8, w8a16, w4a16 and
smoothquant int8 — each the registered implementation of the
``linear`` op under its scheme.

Usage:
    from lite_llama.kernels.ops.quantization import fp8_matmul
"""

from .fp8 import fp8_matmul
from .w4a16 import w4a16_matmul
from .w8a8 import smoothquant_matmul
from .w8a16 import w8a16_matmul

__all__ = ["fp8_matmul", "smoothquant_matmul", "w4a16_matmul", "w8a16_matmul"]
