"""RoPE domain: applying rotary position embeddings in place.

The native kernel takes the position ids the engine already computed; the
FlashInfer row does the same with its own CUDA implementation. Both are
verified against each other, and both rank below a hypothetical measured
winner until the frozen store says otherwise — position embedding is cheap
enough that the wrong choice is invisible in a profile and only shows up in
aggregate, which is exactly what the ranking tier is for.

Usage:
    from lite_llama.kernels.ops import rope  # noqa: F401  (registers rows)
"""

from lite_llama.kernels.backend.flashinfer import CUDA_SM75
from lite_llama.kernels.dispatcher import (
    NATIVE_BASELINE,
    UNMEASURED,
    GoldenRecord,
    KernelSpec,
    register,
)

register(
    KernelSpec(
        name="native/rope_emb_forward",
        op="rope",
        backend="native",
        target="lite_llama.kernels.ops.rope.rope_emb:rope_emb_forward",
        dtypes=("bf16", "fp16"),
        golden=NATIVE_BASELINE,
    )
)
# Golden snapshot from bench_flashinfer.py (b8_s128_h128, bf16): the two
# implementations differ only in intermediate precision (native multiplies
# through the bf16 tables, FlashInfer keeps fp32 angles). The 3.125e-2
# max-abs-diff is one bf16 output ulp at a magnitude of ~4; the surviving
# tolerance-edge drift (~1.4e-2) sits on small outputs where the rotation
# cancels two large products — both sides sit equally close to the fp64
# rotation there.
register(
    KernelSpec(
        name="flashinfer/rope",
        op="rope",
        backend="flashinfer",
        target="lite_llama.kernels.backend.flashinfer.rope:rope",
        available="lite_llama.kernels.backend.flashinfer:available",
        capability=CUDA_SM75,
        dtypes=("bf16", "fp16"),
        golden=GoldenRecord(
            verified=True, max_abs_diff=3.125e-2, baseline="native/rope_emb_forward"
        ),
        priority=UNMEASURED,
    )
)
