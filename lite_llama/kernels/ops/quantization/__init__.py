"""Quantisation GEMM kernels: one Triton kernel per numeric format.

Re-exports the five matmul entry points — fp8, nvfp4, w8a16, w4a16 and
smoothquant int8, each the registered implementation of the ``linear`` op
under its scheme — plus the one-shot word→byte repack helpers, the
per-token-group activation quantiser and the scale-grid layout descriptors it
allocates to, so the quant methods can pull them from the package root.

Usage:
    from lite_llama.kernels.ops.quantization import fp8_matmul

``nvfp4`` is weight-only despite the narrow weight: sm90 has no fp4 MMA, so
its return is bytes, not FLOPs.
"""

from .activation import per_token_group_quant
from .fp8 import fp8_matmul, fp8_quantize_per_token
from .int4_repack import repack_int4_experts
from .int8_repack import unpack_int8_experts
from .nvfp4 import NVFP4_BLOCK, nvfp4_matmul, quantize_nvfp4_blockwise
from .scale_layout import (
    COLUMN_MAJOR,
    COLUMN_MAJOR_TMA,
    ROW_MAJOR,
    TMA_SCALE_ALIGNMENT,
    ScaleLayout,
    create_scale_output,
    infer_scale_layout,
)
from .w4a16 import w4a16_matmul
from .w8a8 import int8_quantize_per_token, smoothquant_matmul
from .w8a16 import w8a16_matmul

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
