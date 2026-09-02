"""GEMM domain: the linear projections, one row per quantisation scheme.

Registers the spec rows and re-exports the entry points — ``linear_torch``
plus one function per quant scheme (w8a16, w4a16, w8a8 int8/fp8, nvfp4).

Usage:
    from lite_llama.kernels import linear_torch, linear_w8a16
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
        # ``gptq_int8`` is the asymmetric half of the row: GPTQ ``bits=8``
        # checkpoints land here (with zero points) after the load-time unpack,
        # while the int4 GPTQ scheme keeps its own packed-words row below.
        schemes=("fp8", "blockwise_int8", "gptq_int8"),
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
register(
    KernelSpec(
        name="native/linear_nvfp4",
        op="linear",
        backend="native",
        target="lite_llama.kernels.ops.gemm.linear:linear_nvfp4",
        # Weight-only, so the activation dtype is the model's and the row serves
        # both. No capability floor beyond what the fp8 *scale* decode needs:
        # there is no fp4 MMA involved on any device, which is the whole point.
        dtypes=("bf16", "fp16"),
        schemes=("nvfp4",),
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
