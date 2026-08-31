"""Quantisation GEMM kernels: one Triton kernel per numeric format.

The weight-only kernels (w8a16, w4a16) keep the weight quantised in HBM and
widen it in the inner loop — the decode step is bound by how many bytes of
weight it must stream, so the saving is proportional to the compression ratio.
The W8A8 kernels (w8a8, fp8) quantise the activations too, so the whole GEMM
runs in the low precision.

Naming follows the *weight-activation* bit-width convention:
    w8a16 — 8-bit weight (fp8-e4m3 or int8), fp16 activation
    w4a16 — 4-bit weight (AWQ/GPTQ packed int4), fp16 activation
    w8a8  — int8 weight + dynamic per-token int8 activation (SmoothQuant)
    fp8   — fp8-e4m3 weight + dynamic per-token fp8-e4m3 activation
"""

from .fp8 import fp8_matmul
from .w4a16 import w4a16_matmul
from .w8a8 import smoothquant_matmul
from .w8a16 import w8a16_matmul

__all__ = ["fp8_matmul", "smoothquant_matmul", "w4a16_matmul", "w8a16_matmul"]
