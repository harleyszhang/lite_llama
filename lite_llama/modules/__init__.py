"""``nn.Module`` building blocks shared by the decoder models.

Everything here is a *component* — a projection, an attention block, an MLP,
an MoE block, a RoPE table — with no notion of how blocks stack into a model.
The layer/model level (:class:`~lite_llama.models.base.DecoderLayer`,
:class:`~lite_llama.models.base.CausalLM`, the concrete model classes) lives in
:mod:`lite_llama.models`.

Import the public API from here:
    ``from lite_llama.modules import ColumnParallelLinear, PagedAttention``
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .attention import PagedAttention
    from .linear import (
        ColumnParallelLinear,
        LinearBase,
        QKVParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )
    from .mlp import FusedMLP
    from .moe import SparseMoeBlock
    from .rotary_embedding import MRotaryEmbedding, RotaryEmbedding
    from .vocab_parallel import ParallelLMHead, VocabParallelEmbedding, vocab_shard


_EXPORTS: dict[str, tuple[str, str]] = {
    "ColumnParallelLinear": (".linear", "ColumnParallelLinear"),
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
