"""Quantisation GEMM kernels: one Triton kernel per weight format.

All three are "weight stays quantised in HBM, widened in the inner loop" —
the decode step is bound by how many bytes of weight it must stream, so the
saving is proportional to the compression ratio.

Naming follows the *weight-activation* bit-width convention:
    w8a16 — 8-bit weight (fp8-e4m3 or int8), fp16 activation
    w4a16 — 4-bit weight (AWQ/GPTQ packed int4), fp16 activation
    w8a8  — 8-bit weight + 8-bit dynamic activation (SmoothQuant)
"""

from .w8a16 import w8a16_matmul
from .w4a16 import w4a16_matmul
from .w8a8 import smoothquant_matmul

__all__ = ["w8a16_matmul", "w4a16_matmul", "smoothquant_matmul"]
