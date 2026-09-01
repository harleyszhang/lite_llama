"""Kernel layer: implementations in ``ops/``, policy in ``dispatcher/``.

Three layers, one direction of knowledge:

* :mod:`lite_llama.kernels.ops` — what lite_llama computes. One directory per
  operator domain, each holding the native Triton implementations beside the
  registration rows that put them (and the external-library contenders) into
  the registry as data.
* :mod:`lite_llama.kernels.dispatcher` — which row runs here. Torch-free
  machinery (spec rows, registry, deterministic dispatch, autotune store).
* :mod:`lite_llama.kernels.backend` — one package per external library:
  install metadata, an availability probe, and the adapters that translate
  the op contracts into each library's calling convention.

This facade re-exports exactly what the model and engine layers call: the
kernels they invoke directly (the ``linear_*`` entries, paged attention,
MoE, …) and the dispatch entry point for the ops that have contenders.
Importing it registers every spec row; nothing loads a kernel eagerly.

Usage:
    from lite_llama.kernels import dispatch, flash_decoding
    sel = dispatch("attention.decode", dtype="bf16")
    fn = sel.load()
"""

from . import backend as backend
from . import dispatcher as dispatcher
from . import ops as ops

# Dispatch machinery: the ops with contenders go through these.
from .dispatcher import Selected, dispatch, explain, invalidate_cache, op_backend_env
from .dispatcher.autotune import install_frozen_perf_provider

# The kernels the model/engine layers call directly.
from .ops.activation.activations import gelu, leaky_relu, relu, silu, tanh
from .ops.activation.swiglu import swiglu_forward, swiglu_forward_fused
from .ops.attention.flashattention2_nopad import flash_attention2_no_pad
from .ops.attention.flashdecoding import flash_decoding
from .ops.embeddings.vocab_embedding import vocab_parallel_embedding
from .ops.gemm.linear import (
    linear_torch,
    linear_w4a16,
    linear_w8a8_fp8,
    linear_w8a8_int8,
    linear_w8a16,
)
from .ops.kvcache.update_kv_buffer import update_kv_buffer
from .ops.kvcache.update_kv_index import update_kv_index
from .ops.layernorm.skip_rmsnorm import skip_rmsnorm
from .ops.moe.fused_moe import fused_moe, moe_align_block_size
from .ops.quantization import smoothquant_matmul, w4a16_matmul, w8a16_matmul
from .ops.rope.rope_emb import rope_emb_forward

# Frozen measured ranking (ROADMAP v0.10): records under the autotune cache's
# frozen/ dir become the rank step's perf input. Nothing is read until the
# first dispatch asks, and LITE_LLAMA_FROZEN_RANK=0 turns the lookup off.
install_frozen_perf_provider()

__all__ = [
    "Selected",
    "backend",
    "dispatch",
    "explain",
    "flash_attention2_no_pad",
    "flash_decoding",
    "fused_moe",
    "gelu",
    "invalidate_cache",
    "leaky_relu",
    "linear_torch",
    "linear_w4a16",
    "linear_w8a8_fp8",
    "linear_w8a8_int8",
    "linear_w8a16",
    "moe_align_block_size",
    "op_backend_env",
    "ops",
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
