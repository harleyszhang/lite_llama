"""Activation domain: the SwiGLU pair, fused and split.

Registers the domain's spec rows and points at the implementations in
:mod:`~lite_llama.kernels.ops.activation.swiglu`: the two-input
``swiglu_forward`` and the fused single-input variant.

Usage:
    from lite_llama.kernels.ops.activation.swiglu import swiglu_forward
"""

from lite_llama.kernels.dispatcher import NATIVE_BASELINE, KernelSpec, register

register(
    KernelSpec(
        name="native/swiglu_forward_fused",
        op="elementwise.swiglu",
        backend="native",
        target="lite_llama.kernels.ops.activation.swiglu:swiglu_forward_fused",
        dtypes=("bf16", "fp16"),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="native/swiglu_forward",
        op="elementwise.swiglu_split",
        backend="native",
        target="lite_llama.kernels.ops.activation.swiglu:swiglu_forward",
        dtypes=("bf16", "fp16"),
        golden=NATIVE_BASELINE,
    )
)
