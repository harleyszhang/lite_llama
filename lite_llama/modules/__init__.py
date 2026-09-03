"""``nn.Module`` building blocks shared by the decoder models.

Re-exports the parallel-aware layers (linear, embedding, LM head),
:class:`PagedAttention`, :class:`FusedMLP`, :class:`SparseMoeBlock` and
the RoPE tables — everything a model definition composes.

Usage:
    from lite_llama.modules import QKVParallelLinear, FusedMLP
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .attention import PagedAttention
    from .deepseek_v4.attention import DeepseekV4Attention
    from .deepseek_v4.hyper_connection import (
        DeepseekV4HyperConnection,
        DeepseekV4HyperHead,
    )
    from .deepseek_v4.rope import DeepseekV4RotaryEmbedding
    from .linear import (
        ColumnParallelLinear,
        LinearBase,
        QKVParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )
    from .mla import DeepseekV2MLAAttention
    from .mlp import FusedMLP
    from .moe import SparseMoeBlock
    from .rotary_embedding import MRotaryEmbedding, RotaryEmbedding
    from .vocab_parallel import ParallelLMHead, VocabParallelEmbedding, vocab_shard


# Some components (attention, MoE, fused MLP) import the Triton kernels;
# resolving the facade lazily keeps ``from lite_llama.modules import
# RotaryEmbedding`` CPU-only.
_EXPORTS: dict[str, tuple[str, str]] = {
    "ColumnParallelLinear": (".linear", "ColumnParallelLinear"),
    "DeepseekV4Attention": (".deepseek_v4.attention", "DeepseekV4Attention"),
    "DeepseekV4HyperConnection": (
        ".deepseek_v4.hyper_connection",
        "DeepseekV4HyperConnection",
    ),
    "DeepseekV4HyperHead": (".deepseek_v4.hyper_connection", "DeepseekV4HyperHead"),
    "DeepseekV4RotaryEmbedding": (".deepseek_v4.rope", "DeepseekV4RotaryEmbedding"),
    "DeepseekV2MLAAttention": (".mla", "DeepseekV2MLAAttention"),
    "FusedMLP": (".mlp", "FusedMLP"),
    "LinearBase": (".linear", "LinearBase"),
    "MRotaryEmbedding": (".rotary_embedding", "MRotaryEmbedding"),
    "PagedAttention": (".attention", "PagedAttention"),
    "ParallelLMHead": (".vocab_parallel", "ParallelLMHead"),
    "QKVParallelLinear": (".linear", "QKVParallelLinear"),
    "ReplicatedLinear": (".linear", "ReplicatedLinear"),
    "RotaryEmbedding": (".rotary_embedding", "RotaryEmbedding"),
    "RowParallelLinear": (".linear", "RowParallelLinear"),
    "SparseMoeBlock": (".moe", "SparseMoeBlock"),
    "VocabParallelEmbedding": (".vocab_parallel", "VocabParallelEmbedding"),
    "vocab_shard": (".vocab_parallel", "vocab_shard"),
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


__all__ = [
    "ColumnParallelLinear",
    "DeepseekV2MLAAttention",
    "DeepseekV4Attention",
    "DeepseekV4HyperConnection",
    "DeepseekV4HyperHead",
    "DeepseekV4RotaryEmbedding",
    "FusedMLP",
    "LinearBase",
    "MRotaryEmbedding",
    "PagedAttention",
    "ParallelLMHead",
    "QKVParallelLinear",
    "ReplicatedLinear",
    "RotaryEmbedding",
    "RowParallelLinear",
    "SparseMoeBlock",
    "VocabParallelEmbedding",
    "vocab_shard",
]
