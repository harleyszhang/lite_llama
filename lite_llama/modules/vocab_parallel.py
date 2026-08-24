"""Vocabulary-parallel embedding and LM head: the two ends of the token axis, sharded.

Both tensors are ``[vocab_size, hidden_size]``, so tensor parallelism cuts them along
the *vocabulary*: rank ``r`` owns rows ``[r * vocab/tp, (r+1) * vocab/tp)``. The
embedding is a gather, so each rank looks up only the ids it owns and one
``all_reduce`` sums the (mostly zero) contributions into the full hidden vector. The LM
head is a GEMM against those same rows, so each rank produces a *slice* of the logits
and deliberately does **not** gather them: the sampler consumes local logits directly
(:mod:`lite_llama.engine.sampler`), which keeps the per-step transfer independent of
the vocabulary size and never materialises a full logits tensor.

Why shard at all: for a 151K-token vocabulary at 8192 hidden these two tensors are
~4.9 GB per rank in fp16, and the decode-step ``lm_head`` GEMM is ``batch x vocab x
hidden`` — the dominant matmul of a large-vocabulary model. Sharding divides both by
``tp``. For tied models it is also the only *correct* option: an unsharded head over a
sharded embedding would be two different tensors claiming to be one.

Usage:
    self.embed_tokens = VocabParallelEmbedding(vocab, hidden)
    self.lm_head = ParallelLMHead(vocab, hidden)
    self.lm_head.weight = self.embed_tokens.weight   # tie_word_embeddings
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributed.parallel_state import all_reduce_tp, divide, get_tp_rank, get_tp_world_size
from ..kernels import vocab_parallel_embedding


def vocab_shard(vocab_size: int, *, rank: int | None = None, tp_size: int | None = None) -> range:
    """Vocabulary ids owned by ``rank``, as a half-open ``range``.

    Kept as a free function of plain integers — no module, no device — because it is
    the whole of the layout's logic: a test can ask which rank owns id 40 000 of a
    151 936-token vocabulary without building anything.

    Raises:
        ValueError: If ``vocab_size`` does not divide across the ranks.
    """
    tp_size = get_tp_world_size() if tp_size is None else tp_size
    rank = get_tp_rank() if rank is None else rank
    local = divide(vocab_size, tp_size, "vocabulary")
    return range(rank * local, (rank + 1) * local)


class VocabParallelEmbedding(nn.Module):
    """Token embedding whose rows are split across TP ranks.

    Each rank holds ``vocab_size / tp`` rows, gathers the ids that fall inside its
    range and zeroes the rest, then one ``all_reduce`` over the hidden dimension
    makes every rank hold the same complete embedding. An unmasked
    ``F.embedding`` would happily return row ``id - start`` for an id this rank
    does not own, and the all-reduce would sum that garbage in — the zeroing is
    the whole subtlety.

    The mapping, the gather and the zeroing are one fused Triton kernel
    (:func:`~lite_llama.kernels.vocab_embedding.vocab_parallel_embedding`):
    the id->row arithmetic that used to run as an eager chain of seven kernels
    per lookup is two scalar register ops inside it. That matters here more
    than anywhere else in the model because TP disables CUDA graphs, so there
    is no replay to hide launch overhead behind on the decode path.

    Args:
        vocab_size: Full vocabulary size (split across ranks).
        hidden_size: Width of the residual stream (not split).
        dtype: Storage type of the weight.
    """

    def __init__(
        self, vocab_size: int, hidden_size: int, dtype: torch.dtype = torch.float16
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.shard = vocab_shard(vocab_size)
        self.weight = nn.Parameter(
            torch.empty(len(self.shard), hidden_size, dtype=dtype), requires_grad=False
        )

    @property
    def local_vocab_size(self) -> int:
        """Rows of the vocabulary this rank owns; ``vocab_size`` when TP is off."""
        return len(self.shard)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if get_tp_world_size() == 1:
            return F.embedding(input_ids, self.weight)
        # One fused kernel — map, gather, zero — where the eager path launched
        # seven, then the all-reduce sums every rank's contribution into the
        # complete embedding.
        out = vocab_parallel_embedding(
            input_ids, self.weight, self.shard.start, self.local_vocab_size
        )
        return all_reduce_tp(out.view(*input_ids.shape, self.hidden_size))

    def extra_repr(self) -> str:
        return (
            f"vocab={self.vocab_size} (local {self.local_vocab_size}), "
            f"hidden={self.hidden_size}, collective=all_reduce"
        )


class ParallelLMHead(VocabParallelEmbedding):
    """Output projection over this rank's slice of the vocabulary.

    Shares :class:`VocabParallelEmbedding`'s storage and shard arithmetic — same tensor,
    read the other way round — which is what makes ``tie_word_embeddings`` a single
    assignment rather than a special case. :meth:`forward` returns **local** logits
    ``[*, vocab_size / tp]``; :class:`lite_llama.engine.sampler.Sampler` reconstructs
    the global distribution from a scalar per row (see there), so no logits collective
    happens here.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)

    def extra_repr(self) -> str:
        return (
            f"vocab={self.vocab_size} (local {self.local_vocab_size}), "
            f"hidden={self.hidden_size}, collective=none (local logits)"
        )
