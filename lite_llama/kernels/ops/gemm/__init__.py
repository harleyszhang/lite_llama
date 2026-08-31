"""GEMM domain: the linear projections, one row per quantisation scheme.

The group's whole surface is the five native Triton/torch entry points of
:mod:`~lite_llama.kernels.ops.gemm.linear` plus DeepGEMM's Hopper fp8 path.
Splitting linear by scheme (rather than one row with branches) is what lets
dispatch answer "which kernel for w8a16_fp8 on this shape?" without the kernel
itself re-deriving the answer internally.

DeepGEMM registers ``verified=False`` on purpose: its wrappers are written
against the upstream API but have no hardware run on record yet (the CI box is
an sm86 A10, DeepGEMM needs sm90+), so the golden gate keeps the row out of
default dispatch until an H100 box produces a max-abs-diff — an explicit
``backend="deepgemm"`` may still force it.

Usage:
    from lite_llama.kernels.ops import gemm  # noqa: F401  (registers the rows)
"""

from lite_llama.kernels.backend.deepgemm import CUDA_SM90
from lite_llama.kernels.dispatcher import (
    NATIVE_BASELINE,
    UNMEASURED,
    GoldenRecord,
    KernelSpec,
    LayoutRequirement,
    register,
)

# --------------------------------------------------------------------------- #
# native — one row per quantisation scheme, so dispatch never needs a branch
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="native/linear_torch",
        op="linear",
        backend="native",
        target="lite_llama.kernels.ops.gemm.linear:linear_torch",
        schemes=("unquantized",),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w8a16",
        op="linear",
        backend="native",
        target="lite_llama.kernels.ops.gemm.linear:linear_w8a16",
        dtypes=("bf16", "fp16"),
        schemes=("fp8", "blockwise_int8"),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w4a16",
        op="linear",
        backend="native",
        target="lite_llama.kernels.ops.gemm.linear:linear_w4a16",
        dtypes=("bf16", "fp16"),
        schemes=("awq", "gptq"),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w8a8_int8",
        op="linear",
        backend="native",
        target="lite_llama.kernels.ops.gemm.linear:linear_w8a8_int8",
        dtypes=("bf16", "fp16"),
        schemes=("w8a8_int8",),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w8a8_fp8",
        op="linear",
        backend="native",
        target="lite_llama.kernels.ops.gemm.linear:linear_w8a8_fp8",
        dtypes=("bf16", "fp16"),
        schemes=("w8a8_fp8",),
        golden=NATIVE_BASELINE,
    )
)

# --------------------------------------------------------------------------- #
# deepgemm — fp8 dense GEMM on Hopper, NT weights with per-128 block scales
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="deepgemm/fp8_gemm_nt",
        op="linear",
        backend="deepgemm",
        target="lite_llama.kernels.backend.deepgemm.linear:fp8_gemm_nt",
        available="lite_llama.kernels.backend.deepgemm:available",
        capability=CUDA_SM90,
        dtypes=("bf16", "fp16"),
        schemes=("w8a8_fp8",),
        # NT weights and 128-block scales, transposed/cached once by the
        # wrapper rather than assumed inside the kernel.
        layout=LayoutRequirement(required=("weight:nt", "scale:block_128")),
        # Untested on hardware: no Hopper box has produced a max-abs-diff yet,
        # so the golden gate excludes the row from default dispatch.
        golden=GoldenRecord(
            verified=False, max_abs_diff=None, baseline="F.linear dequant reference"
        ),
        priority=UNMEASURED,
    )
)
