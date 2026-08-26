"""KernelSpec rows for the native backend — declared torch-free, loaded lazily.

The rows below are the floor every op falls back to: wide domains (any
dtype/shape/layout the kernels actually accept), golden-verified (they *are*
the baseline), no capability window. Registering here must stay free of
torch/triton imports — targets are ``"module:attr"`` strings resolved by
dispatch on first use, which keeps cold start at import speed.

Rows arrive per logical-op family as the restructure proceeds; M1.4 covers
``linear`` (one row per quantisation scheme so dispatch never needs a
runtime branch).
"""

from lite_llama.kernels.ops import GoldenRecord, KernelSpec, register

#: The native rows define the golden baseline themselves.
BASELINE = GoldenRecord(verified=True, max_abs_diff=0.0, baseline="native")

register(
    KernelSpec(
        name="native/linear_torch",
        op="linear",
        backend="native",
        target="lite_llama.kernels.impls.native.linear_torch:linear_torch",
        schemes=("unquantized",),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w8a16",
        op="linear",
        backend="native",
        target="lite_llama.kernels.impls.native.linear_triton:linear_w8a16",
        dtypes=("bf16", "fp16"),
        schemes=("fp8", "blockwise_int8"),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w4a16",
        op="linear",
        backend="native",
        target="lite_llama.kernels.impls.native.linear_triton:linear_w4a16",
        dtypes=("bf16", "fp16"),
        schemes=("awq", "gptq"),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w8a8_int8",
        op="linear",
        backend="native",
        target="lite_llama.kernels.impls.native.linear_triton:linear_w8a8_int8",
        dtypes=("bf16", "fp16"),
        schemes=("w8a8_int8",),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w8a8_fp8",
        op="linear",
        backend="native",
        target="lite_llama.kernels.impls.native.linear_triton:linear_w8a8_fp8",
        dtypes=("bf16", "fp16"),
        schemes=("w8a8_fp8",),
        golden=BASELINE,
    )
)
