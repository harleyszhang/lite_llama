"""MoE domain: the fused expert GEMMs, one row per implementation.

Registers the domain's spec rows and re-exports the grouped-GEMM entry
point :func:`~lite_llama.kernels.ops.moe.fused_moe.fused_moe` plus its
alignment helper.

Usage:
    from lite_llama.kernels import fused_moe
"""

from lite_llama.kernels.backend.deepgemm import CUDA_SM90
from lite_llama.kernels.dispatcher import (
    NATIVE_BASELINE,
    UNMEASURED,
    GoldenRecord,
    KernelSpec,
    register,
)

register(
    KernelSpec(
        name="native/fused_moe",
        op="moe",
        backend="native",
        target="lite_llama.kernels.ops.moe.fused_moe:fused_moe",
        dtypes=("bf16", "fp16"),
        schemes=(
            "unquantized",
            "fp8",
            "w8a8_fp8",
            "w8a8_int8",
            "blockwise_int8",
            "awq",
            "gptq",
        ),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="deepgemm/grouped_fp8_moe",
        op="moe",
        backend="deepgemm",
        target="lite_llama.kernels.backend.deepgemm.moe:grouped_moe",
        available="lite_llama.kernels.backend.deepgemm:available",
        capability=CUDA_SM90,
        dtypes=("bf16", "fp16"),
        schemes=("w8a8_fp8",),
        # Untested on hardware (needs sm90+; CI is sm86) — the golden gate
        # keeps this row out of default dispatch until a Hopper run.
        golden=GoldenRecord(verified=False, max_abs_diff=None, baseline="native/fused_moe"),
        priority=UNMEASURED,
    )
)
