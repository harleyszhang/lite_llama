"""Tests for the vocabulary-sharded embedding and LM head.

Shards must partition the vocabulary exactly; foreign ids contribute
zeros so the all-reduce sums to the unsharded result. Runs on CPU with
a filled fake table.

Usage:
    pytest tests/distributed/test_vocab_parallel.py
"""

from __future__ import annotations

import pytest
import torch

from lite_llama.distributed import parallel_state as ps
from lite_llama.modules import ParallelLMHead, VocabParallelEmbedding, vocab_parallel
from tests.distributed.tp_harness import needs_gpus, run_on_tp_ranks

VOCAB, HIDDEN = 256, 128
TOKENS = [0, 1, 65, 127, 128, 129, 255, 5]
BATCH = 3


def _table() -> torch.Tensor:
    """The full ``[VOCAB, HIDDEN]`` weight table, reproducible in every worker."""
    return torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN) / 997.0


def _hidden_states() -> torch.Tensor:
    """Activations for the head. TP replicates activations, so every rank builds these."""
    return torch.linspace(-2.0, 2.0, BATCH * HIDDEN, dtype=torch.float32).reshape(BATCH, HIDDEN)


def _fill(module: VocabParallelEmbedding) -> VocabParallelEmbedding:
    """Load this rank's rows of :func:`_table` onto its device, as the loader would."""
    rows = _table()[module.shard.start : module.shard.stop]
    module.weight.data.copy_(rows.to(module.weight.device))
    return module


def _build(cls, device: torch.device) -> VocabParallelEmbedding:
    module = cls(VOCAB, HIDDEN, dtype=torch.float32)
    return _fill(module.to(device))


# --------------------------------------------------------------------------- #
# Layout arithmetic (no processes, no device)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_the_shards_partition_the_vocabulary(tp_size: int):
    """Every id belongs to exactly one rank — no gap (a token no rank can embed) and no
    overlap (a token counted twice by the all-reduce)."""
    covered = [
        token
        for rank in range(tp_size)
        for token in vocab_parallel.vocab_shard(VOCAB, rank=rank, tp_size=tp_size)
    ]

    assert sorted(covered) == list(range(VOCAB))


def test_a_vocabulary_that_does_not_divide_is_rejected():
    """Padding the table to a multiple of ``tp`` would make ids past the end samplable."""
    with pytest.raises(ValueError, match="vocabulary 151936 does not divide across 5"):
        vocab_parallel.vocab_shard(151_936, rank=0, tp_size=5)


def test_shard_reads_the_ambient_parallel_state_when_asked_nothing():
    """The default arguments are the running grid, which is how the layers call it."""
    assert vocab_parallel.vocab_shard(VOCAB) == range(0, VOCAB)


# --------------------------------------------------------------------------- #
# Equivalence on a real process group
# --------------------------------------------------------------------------- #
def _embedding_payload(rank: int) -> list[list[float]]:
    device = torch.device("cuda", rank)
    embedding = _build(VocabParallelEmbedding, device)
    return embedding(torch.tensor(TOKENS, device=device)).tolist()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_sharded_embedding_matches_the_unsharded_table_exactly(tp_size: int):
    """Bit-exact, not merely close: each rank contributes either the real row or zeros,
    and adding zeros in float32 changes nothing. An approximate match would mean some
    rank contributed a row it does not own."""
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"needs {tp_size} CUDA devices")
    expected = _table()[TOKENS]

    for rank, rows in enumerate(run_on_tp_ranks(_embedding_payload, tp_size=tp_size)):
        torch.testing.assert_close(torch.tensor(rows), expected, rtol=0, atol=0, msg=f"r{rank}")


def _head_payload(rank: int) -> list[list[float]]:
    device = torch.device("cuda", rank)
    head = _build(ParallelLMHead, device)
    return head(_hidden_states().to(device)).tolist()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_concatenated_local_logits_reproduce_the_full_logits(tp_size: int):
    """The head deliberately does not gather, so the test does the gather instead: rank
    order concatenation must be the full ``[batch, vocab]`` logits."""
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"needs {tp_size} CUDA devices")
    shards = run_on_tp_ranks(_head_payload, tp_size=tp_size)
    expected = _hidden_states() @ _table().T

    assert all(len(row) == VOCAB // tp_size for row in shards[0])
    gathered = torch.cat([torch.tensor(shard) for shard in shards], dim=-1)
    torch.testing.assert_close(gathered, expected, rtol=1e-5, atol=1e-4)


def _local_contribution_payload(rank: int) -> list[list[float]]:
    """What this rank alone puts into the sum, with the all-reduce stubbed out.

    Patching the module global is safe inside a worker process, and it is the only way to
    see the masked tensor: after the collective every rank holds the same answer, which is
    exactly what would hide a mask bug.
    """
    device = torch.device("cuda", rank)
    embedding = _build(VocabParallelEmbedding, device)
    vocab_parallel.all_reduce_tp = lambda tensor: tensor
    return embedding(torch.tensor(TOKENS, device=device)).tolist()


@needs_gpus(2)
def test_a_rank_contributes_zeros_for_ids_outside_its_shard():
    """The Triton kernel's mask in isolation: owned ids give the real row, foreign ids give
    exact zeros. Drop the mask and foreign ids come back as row ``id - start`` — a
    plausible embedding of the wrong token, which the all-reduce then makes unanimous."""
    table = _table()

    for rank, rows in enumerate(run_on_tp_ranks(_local_contribution_payload, tp_size=2)):
        shard = vocab_parallel.vocab_shard(VOCAB, rank=rank, tp_size=2)
        for token, row in zip(TOKENS, rows, strict=True):
            expected = table[token] if token in shard else torch.zeros(HIDDEN)
            torch.testing.assert_close(torch.tensor(row), expected, rtol=0, atol=0)


def _tie_payload(rank: int) -> tuple[bool, int, int]:
    embedding = VocabParallelEmbedding(VOCAB, HIDDEN, dtype=torch.float32)
    head = ParallelLMHead(VOCAB, HIDDEN, dtype=torch.float32)
    head.weight = embedding.weight
    return head.weight is embedding.weight, head.local_vocab_size, ps.get_tp_world_size()


@needs_gpus(2)
def test_tying_stays_one_tensor_because_both_ends_own_the_same_shard():
    """``tie_word_embeddings`` is a single assignment only because the two modules agree on
    the split. A head sharded differently from the embedding could not be tied at all —
    which is the real reason the vocabulary is split, ahead of the memory saved."""
    for tied, local, world in run_on_tp_ranks(_tie_payload, tp_size=2):
        assert tied
        assert world == 2
        assert local == VOCAB // 2


@needs_gpus(2)
def test_the_parent_process_keeps_its_world_of_one():
    """A spawned rank must not be able to leave a grid behind: every layer built after
    these tests reads ``parallel_state`` at construction time."""
    run_on_tp_ranks(_tie_payload, tp_size=2)

    assert ps.get_tp_world_size() == 1
    assert ps.get_dp_world_size() == 1
