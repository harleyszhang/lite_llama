"""Vocabulary-parallel embedding and LM head: both token-axis ends, sharded.

:class:`VocabParallelEmbedding` owns one vocab shard and masks foreign ids;
:class:`ParallelLMHead` reuses the same sharded weight for the output
projection, so TP never materialises the full vocab matrix.

Usage:
    embed = VocabParallelEmbedding(vocab_size, hidden_size, dtype)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributed.parallel_state import all_reduce, divide, get_tp_rank, get_tp_world_size
from ..kernels import vocab_parallel_embedding


def vocab_shard(vocab_size: int, *, rank: int | None = None, tp_size: int | None = None) -> range:
    """Vocabulary ids owned by ``rank``, as a half-open ``range``.

    A free function of plain integers — no module, no device — so a test can
    ask which rank owns id 40 000 of a 151 936-token vocabulary without
    building anything.

    Raises:
        ValueError: If ``vocab_size`` does not divide across the ranks.
    """
    tp_size = get_tp_world_size() if tp_size is None else tp_size
    rank = get_tp_rank() if rank is None else rank
    local = divide(vocab_size, tp_size, "vocabulary")
    return range(rank * local, (rank + 1) * local)


class VocabParallelEmbedding(nn.Module):
    """Token embedding whose rows are split across TP ranks.

    Each rank holds ``vocab_size / tp`` rows, gathers the ids that fall inside
    its range and zeroes the rest, then one ``all_reduce`` over the hidden
    dimension makes every rank hold the same complete embedding. The zeroing
    is the whole subtlety: an unmasked ``F.embedding`` would return row
    ``id - start`` for an id this rank does not own, and the all-reduce would
    sum that garbage in.

    The mapping, the gather and the zeroing are one fused Triton kernel
    (:func:`~lite_llama.kernels.ops.embeddings.vocab_embedding.vocab_parallel_embedding`):
    the id->row arithmetic that used to run as an eager chain of seven kernels
    per lookup is two scalar register ops inside it. Decode may replay from a
    CUDA graph, which would hide those launches, but the saving still has to
    hold without one: prefill is always eager, and the graphs are dropped
    whenever the startup checks in
    :meth:`~lite_llama.executor.model_runner.ModelRunner.enable_cuda_graph`
    fail on any rank.

    Args:
        vocab_size: Full vocabulary size (split across ranks).
        hidden_size: Width of the residual stream (not split).
        dtype: Storage type of the weight.
    """

    def __init__(
        self, vocab_size: int, hidden_size: int, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.shard = vocab_shard(vocab_size)
        self.weight = nn.Parameter(
            torch.empty(len(self.shard), hidden_size, dtype=dtype), requires_grad=False
        )
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(
        self, param: torch.Tensor, loaded: torch.Tensor, shard_id=None
    ) -> torch.Tensor:
        """Fill this rank's vocabulary rows from the full table; return the view written.

        Both the embedding and the LM head are ``[vocab, hidden]`` split along
        the vocabulary, so the same rule serves them; the incoming tensor is
        narrowed to this rank's rows — the same :attr:`shard` the gather masks
        with. Never packed, so ``shard_id`` is unused.
        """
        world_size = get_tp_world_size()
        if world_size > 1:
            size = loaded.shape[0] // world_size
            loaded = loaded.narrow(0, get_tp_rank() * size, size)
        if param.shape != loaded.shape:
            raise ValueError(
                f"checkpoint tensor of shape {tuple(loaded.shape)} does not fit "
                f"parameter view of shape {tuple(param.shape)}"
            )
        param.data.copy_(loaded)
        return param.data

    @property
    def local_vocab_size(self) -> int:
        """Rows of the vocabulary this rank owns; ``vocab_size`` when TP is off."""
        return len(self.shard)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if get_tp_world_size() == 1:
            return F.embedding(input_ids, self.weight)
        out = vocab_parallel_embedding(
            input_ids, self.weight, self.shard.start, self.local_vocab_size
        )
        return all_reduce(out.view(*input_ids.shape, self.hidden_size))

    def extra_repr(self) -> str:
        return (
            f"vocab={self.vocab_size} (local {self.local_vocab_size}), "
            f"hidden={self.hidden_size}, collective=all_reduce"
        )


class ParallelLMHead(VocabParallelEmbedding):
    """Output projection over this rank's slice of the vocabulary.

    Shares :class:`VocabParallelEmbedding`'s storage and shard arithmetic —
    same tensor read the other way round — which makes ``tie_word_embeddings``
    a single assignment rather than a special case. :meth:`forward` returns
    **local** logits ``[*, vocab_size / tp]``;
    :class:`lite_llama.engine.sampler.Sampler` reconstructs the global
    distribution from a scalar per row, so no logits collective happens here.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)

    def extra_repr(self) -> str:
        return (
            f"vocab={self.vocab_size} (local {self.local_vocab_size}), "
            f"hidden={self.hidden_size}, collective=none (local logits)"
        )
