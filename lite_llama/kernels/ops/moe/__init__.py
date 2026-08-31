"""MoE domain: the fused expert GEMMs, one row per implementation.

The native row covers every quantisation scheme on purpose: ``fused_moe``
derives the expert format from ``w1.dtype`` (uint8 fp8-e4m3 / int8 / packed
int32) rather than from a flag, so splitting the row per scheme would be one
spec claim per branch of the same dispatch the kernel already does internally.
DeepGEMM's grouped variant is the sm90+ contender — same grouped-contiguous
semantics, Hopper tensor cores, unverified until it runs on real hardware.

Usage:
    from lite_llama.kernels.ops import moe  # noqa: F401  (registers the rows)
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
