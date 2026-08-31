"""Activation domain: the SwiGLU pair, fused and split.

Two rows because the two entry points are genuinely different contracts:
``elementwise.swiglu`` consumes a pre-concatenated fused gate/up tensor while
``elementwise.swiglu_split`` takes gate and up separately — a caller holding
two tensors cannot serve the fused kernel and dispatch should say so, not have
the kernel silently transpose around it. No external row yet: activation
kernels are bandwidth-bound to the point where the win is in the surrounding
fusion, not in the elementwise op itself.

Usage:
    from lite_llama.kernels.ops import activation  # noqa: F401  (registers rows)
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
