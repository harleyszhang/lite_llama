"""Kernel layer: implementations in ``ops/``, policy in ``dispatcher/``.

The public names re-export the dispatch tier (``dispatch``) and the
individual op entry points; importing the package loads no CUDA library
until a kernel is actually dispatched.

Usage:
    from lite_llama.kernels import dispatch, fused_moe
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from . import backend as backend
from . import dispatcher as dispatcher
from . import ops as ops

# Dispatch machinery: the ops with contenders go through these.
from .dispatcher import Selected, dispatch, explain, invalidate_cache, op_backend_env
from .dispatcher.autotune import install_frozen_perf_provider

# The kernels the model/engine layers call directly, resolved on first use:
# importing an implementation module pulls in Triton, while callers such as
# modules/attention only need ``dispatch`` from the torch-free machinery above.
_EXPORTS: dict[str, tuple[str, str]] = {
    "gelu": (".ops.activation.activations", "gelu"),
    "leaky_relu": (".ops.activation.activations", "leaky_relu"),
    "relu": (".ops.activation.activations", "relu"),
    "silu": (".ops.activation.activations", "silu"),
    "tanh": (".ops.activation.activations", "tanh"),
    "swiglu_forward": (".ops.activation.swiglu", "swiglu_forward"),
    "swiglu_forward_fused": (".ops.activation.swiglu", "swiglu_forward_fused"),
    "flash_attention2_no_pad": (".ops.attention.flashattention2_nopad", "flash_attention2_no_pad"),
    "flash_attention2_chunked": (
        ".ops.attention.flashattention2_nopad",
        "flash_attention2_chunked",
    ),
    "flash_decoding": (".ops.attention.flashdecoding", "flash_decoding"),
    "vocab_parallel_embedding": (".ops.embeddings.vocab_embedding", "vocab_parallel_embedding"),
    "linear_torch": (".ops.gemm.linear", "linear_torch"),
    "linear_w4a16": (".ops.gemm.linear", "linear_w4a16"),
    "linear_w8a8_fp8": (".ops.gemm.linear", "linear_w8a8_fp8"),
    "linear_w8a8_int8": (".ops.gemm.linear", "linear_w8a8_int8"),
    "linear_w8a16": (".ops.gemm.linear", "linear_w8a16"),
    "update_kv_buffer": (".ops.kvcache.update_kv_buffer", "update_kv_buffer"),
    "update_kv_index": (".ops.kvcache.update_kv_index", "update_kv_index"),
    "skip_rmsnorm": (".ops.layernorm.skip_rmsnorm", "skip_rmsnorm"),
    "fused_moe": (".ops.moe.fused_moe", "fused_moe"),
    "fused_moe_w8a8_fp8": (".ops.moe.fused_moe", "fused_moe_w8a8_fp8"),
    "fused_moe_w8a8_int8": (".ops.moe.fused_moe", "fused_moe_w8a8_int8"),
    "moe_align_block_size": (".ops.moe.fused_moe", "moe_align_block_size"),
    "grouped_topk": (".ops.moe.grouped_topk", "grouped_topk"),
    "grouped_topk_torch": (".ops.moe.grouped_topk", "grouped_topk_torch"),
    "smoothquant_matmul": (".ops.quantization", "smoothquant_matmul"),
    "w4a16_matmul": (".ops.quantization", "w4a16_matmul"),
    "w8a16_matmul": (".ops.quantization", "w8a16_matmul"),
    "rope_emb_forward": (".ops.rope.rope_emb", "rope_emb_forward"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS.keys())


# Frozen measured ranking (ROADMAP v0.10): records under the autotune cache's
# frozen/ dir become the rank step's perf input. Nothing is read until the
# first dispatch asks, and LITE_LLAMA_FROZEN_RANK=0 turns the lookup off.
install_frozen_perf_provider()

__all__ = [
    "Selected",
    "backend",
    "dispatch",
    "explain",
    "flash_attention2_chunked",
    "flash_attention2_no_pad",
    "flash_decoding",
    "fused_moe",
    "fused_moe_w8a8_fp8",
    "fused_moe_w8a8_int8",
    "gelu",
    "grouped_topk",
    "grouped_topk_torch",
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
