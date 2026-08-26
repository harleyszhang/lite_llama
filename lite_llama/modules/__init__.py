"""``nn.Module`` building blocks shared by the decoder models.

Everything here is a *component* — a projection, an attention block, an MLP,
an MoE block, a RoPE table — with no notion of how blocks stack into a model.
The layer/model level (:class:`~lite_llama.models.base.DecoderLayer`,
:class:`~lite_llama.models.base.CausalLM`, the concrete model classes) lives in
:mod:`lite_llama.models`.

Import the public API from here:
    ``from lite_llama.modules import ColumnParallelLinear, PagedAttention``
"""

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
