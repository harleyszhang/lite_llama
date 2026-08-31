"""Attention domain: the prefill/decode phases, KV write and MLA decode.

Two phases get separate op ids (``attention.prefill`` / ``attention.decode``)
because their shapes and kernels have nothing in common beyond the KV cache:
prefill is ragged batch GEMM, decode is paged single-token gather. The native
rows point straight at the Triton kernels; FlashInfer's rows are the first
externally-verified implementations and cover both phases with their own paged
KV pool (same ``kv:paged`` contract, different allocation). FlashMLA serves
``attention.mla_decode`` — an op that has no native row at all, because MLA's
latent cache is a different layout, not a different size, and the tree has no
MLA model to host a Triton fallback (v0.11 wires the real thing; the single-
layer harness in ``models/mla_single_layer.py`` is the bench vehicle).

Usage:
    from lite_llama.kernels.ops import attention  # noqa: F401  (registers rows)
"""

from lite_llama.kernels.backend.flashinfer import CUDA_SM75
from lite_llama.kernels.backend.flashmla import CUDA_SM90
from lite_llama.kernels.dispatcher import (
    NATIVE_BASELINE,
    PAGED_KV,
    UNMEASURED,
    GoldenRecord,
    KernelSpec,
    LayoutRequirement,
    register,
)

# --------------------------------------------------------------------------- #
# native — the Triton kernels this repo ships
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="native/flash_attention2_no_pad",
        op="attention.prefill",
        backend="native",
        target="lite_llama.kernels.ops.attention.flashattention2_nopad:flash_attention2_no_pad",
        # fp32 has no path through the kernel: it casts inputs to fp16.
        dtypes=("bf16", "fp16"),
        golden=NATIVE_BASELINE,
    )
)
register(
    KernelSpec(
        name="native/flash_decoding",
        op="attention.decode",
        backend="native",
        target="lite_llama.kernels.ops.attention.flashdecoding:flash_decoding",
        dtypes=("bf16", "fp16"),
        # fp8_kv rides the same kernel: K/V arrive as uint8 rows and are
        # dequantised inside, so no separate row is warranted.
        schemes=("unquantized", "fp8_kv"),
        layout=PAGED_KV,
        golden=NATIVE_BASELINE,
    )
)

# --------------------------------------------------------------------------- #
# flashinfer — both phases, wheel-installable, Ampere onward
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="flashinfer/prefill",
        op="attention.prefill",
        backend="flashinfer",
        target="lite_llama.kernels.backend.flashinfer.attention:prefill_attention",
        available="lite_llama.kernels.backend.flashinfer:available",
        capability=CUDA_SM75,
        dtypes=("bf16", "fp16"),
        golden=GoldenRecord(
            verified=True, max_abs_diff=2.0e-2, baseline="native/flash_attention2_no_pad"
        ),
        priority=UNMEASURED,
    )
)
register(
    KernelSpec(
        name="flashinfer/decode",
        op="attention.decode",
        backend="flashinfer",
        target="lite_llama.kernels.backend.flashinfer.attention:decode_attention",
        available="lite_llama.kernels.backend.flashinfer:available",
        capability=CUDA_SM75,
        dtypes=("bf16", "fp16"),
        layout=PAGED_KV,
        golden=GoldenRecord(verified=True, max_abs_diff=2.0e-2, baseline="native/flash_decoding"),
        priority=UNMEASURED,
    )
)

# --------------------------------------------------------------------------- #
# flashmla — MLA decode against the latent cache; no native row exists
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="flashmla/mla_decode",
        op="attention.mla_decode",
        backend="flashmla",
        target="lite_llama.kernels.backend.flashmla.mla_decode:mla_decode",
        available="lite_llama.kernels.backend.flashmla:available",
        capability=CUDA_SM90,
        # The latent cache (c_kv: [Skv, L_kv]) is not interchangeable with the
        # per-head paged pool the rows above share — its own tag says so.
        layout=LayoutRequirement(required=("kv:mla_latent",)),
        golden=GoldenRecord(
            verified=False, max_abs_diff=None, baseline="minimal MLA harness reference"
        ),
        priority=UNMEASURED,
    )
)
