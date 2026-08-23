"""``nn.Module`` building blocks shared by the decoder models.

Everything here is a *component* — a projection, an attention block, an MLP,
an MoE block, a RoPE table — with no notion of how blocks stack into a model.
The layer/model level (:class:`~lite_llama.models.base.DecoderLayer`,
:class:`~lite_llama.models.base.CausalLM`, the concrete model classes) lives in
:mod:`lite_llama.models`.

Import the public API from here:
    ``from lite_llama.modules import Attention, ColumnParallelLinear``
"""

from .attention import Attention, PagedAttention
from .linear import ColumnParallelLinear, LinearBase, ReplicatedLinear, RowParallelLinear
from .mlp import FusedMLP
from .moe import SparseMoeBlock
from .rotary_embedding import MRotaryEmbedding, RotaryEmbedding

__all__ = [
    "Attention",
    "ColumnParallelLinear",
    "FusedMLP",
    "LinearBase",
    "MRotaryEmbedding",
    "PagedAttention",
    "ReplicatedLinear",
    "RotaryEmbedding",
    "RowParallelLinear",
    "SparseMoeBlock",
]
