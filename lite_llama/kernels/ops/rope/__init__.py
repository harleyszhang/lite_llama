"""RoPE domain: applying rotary position embeddings in place.

Registers the domain's spec row and re-exports
:func:`~lite_llama.kernels.ops.rope.rope_emb.rope_emb_forward`, the
fused kernel that rotates q and k against the cos/sin tables.

Usage:
    from lite_llama.kernels import rope_emb_forward
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
