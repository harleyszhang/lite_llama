"""Triton kernels used by the lite_llama model implementations.

The public surface is intentionally small: exactly the kernels the model and
engine layers call, plus the standalone activation kernels that are useful on
their own. ``flash_attention2_no_pad`` serves the prefill (context) phase on
variable-length batches, while ``flash_decoding`` serves the single-token decode
phase against the paged KV buffer. The ``linear_*`` entry points cover the
projection GEMM, one per quantisation scheme.

Which implementation actually runs is a separate question, answered by
:mod:`lite_llama.kernels.ops` (logical-op contracts and deterministic dispatch)
from the per-backend spec rows in :mod:`lite_llama.kernels.backends`.
"""

from .activations import gelu, leaky_relu, relu, silu, tanh

# Registers the native KernelSpec rows before anything can dispatch; the rows
# are data pointing at the kernel modules below, so nothing loads eagerly.
from .backends import native as _native_specs  # noqa: F401
from .flashattention2_nopad import flash_attention2_no_pad
from .flashdecoding import flash_decoding
from .fused_moe import fused_moe, moe_align_block_size
from .linear import (
    linear_torch,
    linear_w4a16,
    linear_w8a8_fp8,
    linear_w8a8_int8,
    linear_w8a16,
)
from .quantization import smoothquant_matmul, w4a16_matmul, w8a16_matmul
from .rope_emb import rope_emb_forward
from .skip_rmsnorm import skip_rmsnorm
from .swiglu import swiglu_forward, swiglu_forward_fused
from .update_kv_buffer import update_kv_buffer
from .update_kv_index import update_kv_index
from .vocab_embedding import vocab_parallel_embedding

__all__ = [
    "flash_attention2_no_pad",
    "flash_decoding",
    "fused_moe",
    "gelu",
    "leaky_relu",
    "linear_torch",
    "linear_w4a16",
    "linear_w8a8_fp8",
    "linear_w8a8_int8",
    "linear_w8a16",
    "moe_align_block_size",
    "relu",
    "rope_emb_forward",
    "silu",
    "skip_rmsnorm",
    "smoothquant_matmul",
    "swiglu_forward",
    "swiglu_forward_fused",
    "tanh",
    "update_kv_buffer",
    "update_kv_index",
    "vocab_parallel_embedding",
    "w4a16_matmul",
    "w8a16_matmul",
]
