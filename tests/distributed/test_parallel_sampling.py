"""Tests for sampling from a vocabulary that no single rank holds.

The sharded head hands the sampler ``[batch, vocab / tp]``. Reconstructing the *global*
decision from that without gathering logits is the original piece of this design, so the
tests are equivalence tests against the unsharded computation rather than sanity checks:

* :func:`vocab_logsumexp` must equal ``torch.logsumexp`` over the full row, which makes
  ``local_logits - log_z`` **exactly** the global ``log_softmax`` restricted to this
  rank's slice. That identity is the whole reason two scalars per row suffice, so it is
  asserted element-wise on the concatenated slices, not sampled.
* :func:`global_argmax` must equal ``full.argmax`` — and must break ties by lowest id, or
  two ranks holding the same maximum would decode different tokens.
* :func:`sharded_top_p` must draw from the true global nucleus. Driving ``top_p`` to zero
  collapses the nucleus to one token, which turns a sampler into a deterministic function
  and makes the candidate gather testable at all.

Ranks are *not* asserted to draw the same token at ordinary ``top_p``: they legitimately
do not, since each calls ``multinomial`` with its own RNG, and the engines broadcast the
winner (``broadcast_tp``) rather than relying on synchronised randomness.

Usage:
    pytest tests/distributed/test_parallel_sampling.py     # needs 2+ CUDA devices
"""

from __future__ import annotations

import torch

from lite_llama.distributed import parallel_state as ps
from lite_llama.engine.sampler import (
    global_argmax,
    local_vocab_offset,
    sharded_top_p,
    vocab_logsumexp,
)
from lite_llama.modules import vocab_parallel
from tests.distributed.tp_harness import needs_gpus, run_on_tp_ranks

VOCAB, BATCH = 256, 4
TEMPERATURE = 0.8


def _full_logits() -> torch.Tensor:
    """The unsharded logits, rebuilt identically in every worker from a fixed seed."""
    generator = torch.Generator().manual_seed(7)
    return torch.randn(BATCH, VOCAB, generator=generator)


def _tied_logits() -> torch.Tensor:
    """Logits whose maximum occurs twice, once in each half of a 2-way split."""
    logits = torch.full((1, VOCAB), -1.0)
    logits[0, 5] = 10.0
    logits[0, 5 + VOCAB // 2] = 10.0
    return logits


def _local(rank: int, full: torch.Tensor) -> torch.Tensor:
    """This rank's slice of ``full``, on this rank's device — what the head would emit."""
    shard = vocab_parallel.vocab_shard(full.shape[-1])
    return full[:, shard.start : shard.stop].to(torch.device("cuda", rank))


# --------------------------------------------------------------------------- #
# The shard offset
# --------------------------------------------------------------------------- #
def test_no_offset_without_tensor_parallelism():
    """``None`` rather than ``0``: the sampler branches on it, and rank 0 of a sharded
    world does have offset 0 while also needing the collective path."""
    assert local_vocab_offset(VOCAB) is None


def _offset_payload(rank: int) -> int | None:
    return local_vocab_offset(VOCAB // ps.get_tp_world_size())


@needs_gpus(2)
def test_the_offset_is_this_ranks_first_global_token_id():
    assert run_on_tp_ranks(_offset_payload, tp_size=2) == [0, VOCAB // 2]


# --------------------------------------------------------------------------- #
# Decentralised log_softmax
# --------------------------------------------------------------------------- #
def _log_softmax_payload(rank: int) -> tuple[list[list[float]], list[list[float]]]:
    scaled = _local(rank, _full_logits()) / TEMPERATURE
    log_z = vocab_logsumexp(scaled)
    return log_z.tolist(), (scaled - log_z).tolist()


@needs_gpus(2)
def test_two_scalars_per_row_reconstruct_the_global_log_softmax():
    """The identity the design rests on, asserted element-wise.

    A wrong ``logsumexp`` here does not crash or produce NaN — it produces a distribution
    that is off by a constant factor, which biases sampling in a way no smoke test sees.
    """
    shards = run_on_tp_ranks(_log_softmax_payload, tp_size=2)
    full = _full_logits() / TEMPERATURE
    expected_z = torch.logsumexp(full, dim=-1, keepdim=True)

    for rank, (log_z, _) in enumerate(shards):
        torch.testing.assert_close(
            torch.tensor(log_z), expected_z, rtol=1e-6, atol=1e-6, msg=f"r{rank}"
        )

    gathered = torch.cat([torch.tensor(log_probs) for _, log_probs in shards], dim=-1)
    torch.testing.assert_close(gathered, torch.log_softmax(full, dim=-1), rtol=1e-6, atol=1e-6)


@needs_gpus(2)
def test_every_rank_agrees_on_the_normaliser():
    """Ranks that disagree by an epsilon would rank candidates differently once the pool
    is gathered, so the collectives have to leave every rank with the same number."""
    (first, _), (second, _) = run_on_tp_ranks(_log_softmax_payload, tp_size=2)

    assert first == second


# --------------------------------------------------------------------------- #
# Greedy
# --------------------------------------------------------------------------- #
def _argmax_payload(rank: int) -> list[list[int]]:
    local = _local(rank, _full_logits())
    return global_argmax(local, local_vocab_offset(local.shape[-1])).tolist()


@needs_gpus(2)
def test_greedy_over_shards_picks_the_global_argmax():
    """Exact ids, not approximate logits: the answer is a token, and one wrong shard
    offset makes it the wrong token by exactly ``vocab / tp``."""
    expected = _full_logits().argmax(dim=-1, keepdim=True)

    for rank, ids in enumerate(run_on_tp_ranks(_argmax_payload, tp_size=2)):
        assert torch.tensor(ids).tolist() == expected.tolist(), f"rank {rank}"


def _tie_payload(rank: int) -> list[list[int]]:
    local = _local(rank, _tied_logits())
    return global_argmax(local, local_vocab_offset(local.shape[-1])).tolist()


@needs_gpus(2)
def test_a_tie_between_ranks_resolves_to_the_lowest_id():
    """Two ranks holding the same maximum must not decide by kernel ordering: greedy
    decoding has to be reproducible, and every rank must reach the same token."""
    ids = run_on_tp_ranks(_tie_payload, tp_size=2)

    assert ids[0] == ids[1] == [[5]]


# --------------------------------------------------------------------------- #
# Nucleus sampling
# --------------------------------------------------------------------------- #
def _collapsed_nucleus_payload(rank: int) -> list[list[int]]:
    """``top_p`` small enough that the nucleus holds one token: the argmax, drawn through
    the real candidate gather with a deliberately small ``k``."""
    local = _local(rank, _full_logits())
    return sharded_top_p(
        local, TEMPERATURE, top_p=1e-9, vocab_offset=local_vocab_offset(local.shape[-1]), k=8
    ).tolist()


@needs_gpus(2)
def test_a_collapsed_nucleus_draws_the_global_argmax():
    """What this really tests is the candidate gather: with ``k=8`` per rank, the winner
    can only be found if the union of the per-rank top-k contains the global best."""
    expected = _full_logits().argmax(dim=-1, keepdim=True).tolist()

    for rank, ids in enumerate(run_on_tp_ranks(_collapsed_nucleus_payload, tp_size=2)):
        assert ids == expected, f"rank {rank}"


def _nucleus_pool_payload(rank: int) -> list[list[int]]:
    local = _local(rank, _full_logits())
    return sharded_top_p(
        local, TEMPERATURE, top_p=0.9, vocab_offset=local_vocab_offset(local.shape[-1]), k=8
    ).tolist()


@needs_gpus(2)
def test_a_drawn_token_always_comes_from_the_global_top_k():
    """The gather is ``O(k * tp)`` precisely because the union of per-rank top-k contains
    the global top-k; a draw from outside it would mean the pool was assembled wrong."""
    full = _full_logits()
    allowed = {int(token) for token in torch.topk(full, 8 * 2, dim=-1).indices.flatten()}

    for rank, ids in enumerate(run_on_tp_ranks(_nucleus_pool_payload, tp_size=2)):
        for row in ids:
            assert row[0] in allowed, f"rank {rank} drew {row[0]} from outside the pool"


def test_the_local_top_k_union_contains_the_global_top_k():
    """The claim the gather size rests on, as arithmetic rather than as a process test.

    A token in the global top ``k`` has fewer than ``k`` tokens above it anywhere, so it
    certainly has fewer than ``k`` above it on its own shard — which is why each rank only
    has to offer ``k`` candidates, no matter how large the vocabulary is.
    """
    full = _full_logits()
    k = 8

    for tp_size in (2, 4, 8):
        width = VOCAB // tp_size
        for row in range(BATCH):
            union = {
                int(token) + rank * width
                for rank in range(tp_size)
                for token in torch.topk(full[row, rank * width : (rank + 1) * width], k).indices
            }
            assert {int(token) for token in torch.topk(full[row], k).indices} <= union
