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

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from .backends import native as _native_specs  # noqa: F401

# Registers the native KernelSpec rows before anything can dispatch; the rows
# are data pointing at the kernel modules below, so nothing loads eagerly.


_EXPORTS: dict[str, tuple[str, str]] = {
    "flash_attention2_no_pad": (".flashattention2_nopad", "flash_attention2_no_pad"),
    "flash_decoding": (".flashdecoding", "flash_decoding"),
    "fused_moe": (".fused_moe", "fused_moe"),
    "gelu": (".activations", "gelu"),
    "leaky_relu": (".activations", "leaky_relu"),
    "linear_torch": (".linear", "linear_torch"),
    "linear_w4a16": (".linear", "linear_w4a16"),
    "linear_w8a8_fp8": (".linear", "linear_w8a8_fp8"),
    "linear_w8a8_int8": (".linear", "linear_w8a8_int8"),
    "linear_w8a16": (".linear", "linear_w8a16"),
    "moe_align_block_size": (".fused_moe", "moe_align_block_size"),
    "relu": (".activations", "relu"),
    "rope_emb_forward": (".rope_emb", "rope_emb_forward"),
    "silu": (".activations", "silu"),
    "skip_rmsnorm": (".skip_rmsnorm", "skip_rmsnorm"),
    "smoothquant_matmul": (".quantization", "smoothquant_matmul"),
    "swiglu_forward": (".swiglu", "swiglu_forward"),
    "swiglu_forward_fused": (".swiglu", "swiglu_forward_fused"),
    "tanh": (".activations", "tanh"),
    "update_kv_buffer": (".update_kv_buffer", "update_kv_buffer"),
    "update_kv_index": (".update_kv_index", "update_kv_index"),
    "vocab_parallel_embedding": (".vocab_embedding", "vocab_parallel_embedding"),
    "w4a16_matmul": (".quantization", "w4a16_matmul"),
    "w8a16_matmul": (".quantization", "w8a16_matmul"),
}


def _lazy_kernel(name: str) -> Callable[..., Any]:
    """Return a stable call site that resolves its Triton target once."""
    target: Callable[..., Any] | None = None

    def call(*args: Any, **kwargs: Any) -> Any:
        nonlocal target
        if target is None:
            module_name, attribute = _EXPORTS[name]
            target = getattr(import_module(module_name, __name__), attribute)
        return target(*args, **kwargs)

    call.__name__ = name
    call.__qualname__ = name
    call.__module__ = __name__
    return call


# Model modules bind these names at import time. Bind lightweight trampolines,
# so importing a model or inspecting its config never imports Triton; the first
# real kernel call resolves and caches the implementation behind the trampoline.
globals().update({name: _lazy_kernel(name) for name in _EXPORTS})

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
