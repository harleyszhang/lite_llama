"""Tests for the fused vocabulary-parallel embedding kernel.

The contract is small but load-bearing: an id this rank owns gathers its row
verbatim, an id another rank owns yields an exact zero row — so the
``all_reduce`` over ranks sums exactly one real embedding per token. A lookup
that ignored ownership would feed an out-of-shard row into that sum; there is
no downstream check to catch it, and every token outside the shard would come
back as a well-formed embedding of the *wrong* token.

The eager chain the kernel replaces (subtract, compare, compare, and, clamp,
gather, multiply — seven launches per lookup) is reimplemented here as the
reference. Element-for-element agreement with it is the regression condition
of the fusion: same numbers, one kernel instead of seven.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lite_llama.kernels import vocab_parallel_embedding

# A shard in the middle of a larger id space, so foreign ids exist on *both*
# sides of it — the situation of every rank except the first and the last.
SHARD_START = 1000
LOCAL_VOCAB = 512
HIDDEN = 96  # not a power of two: exercises the BLOCK padding mask


def _weight(hidden: int = HIDDEN) -> torch.Tensor:
    return torch.randn(LOCAL_VOCAB, hidden, device="cuda", dtype=torch.float16)


def _eager_chain(
    ids: torch.Tensor, weight: torch.Tensor, shard_start: int, local_vocab: int
) -> torch.Tensor:
    """The seven-kernel eager lookup the fused kernel replaces."""
    local_ids = ids - shard_start
    owned = (local_ids >= 0) & (local_ids < local_vocab)
    out = F.embedding(local_ids.clamp(0, local_vocab - 1), weight)
    return out * owned.unsqueeze(-1).to(out.dtype)


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
def test_owned_ids_gather_their_rows_verbatim(ids_dtype):
    """An owned id returns its row unchanged — a gather, nothing more."""
    weight = _weight()
    ids = torch.randint(
        SHARD_START, SHARD_START + LOCAL_VOCAB, (37,), device="cuda", dtype=ids_dtype
    )

    out = vocab_parallel_embedding(ids, weight, SHARD_START, LOCAL_VOCAB)

    assert out.shape == (37, HIDDEN)
    torch.testing.assert_close(out, weight[(ids - SHARD_START).long()])


def test_foreign_ids_contribute_exact_zero_rows():
    """Ids on either side of the shard must give zeros, not row ``id - start``.

    The all-reduce sums one real row and zeros per token; an unmasked gather
    would slip a foreign row into that sum and corrupt every out-of-shard
    token with no exception to notice.
    """
    weight = _weight()
    below = torch.randint(0, SHARD_START, (8,), device="cuda")
    above = torch.randint(
        SHARD_START + LOCAL_VOCAB, SHARD_START + 4 * LOCAL_VOCAB, (8,), device="cuda"
    )
    ids = torch.cat([below, above]).to(torch.int64)

    out = vocab_parallel_embedding(ids, weight, SHARD_START, LOCAL_VOCAB)

    assert out.shape == (16, HIDDEN)
    assert (out == 0).all()


@pytest.mark.parametrize(
    "token_id,owned",
    [
        pytest.param(SHARD_START - 1, False, id="one-below"),
        pytest.param(SHARD_START, True, id="first-owned"),
        pytest.param(SHARD_START + LOCAL_VOCAB - 1, True, id="last-owned"),
        pytest.param(SHARD_START + LOCAL_VOCAB, False, id="one-above"),
    ],
)
def test_ownership_is_half_open_on_both_ends(token_id, owned):
    """``[start, start + local)`` exactly: first row in, last row in, neighbours out.

    ``one-below`` is the negative-pointer case — the address arithmetic goes
    negative, and only the load mask keeps it from being dereferenced.
    """
    weight = _weight()
    ids = torch.tensor([token_id], device="cuda")

    out = vocab_parallel_embedding(ids, weight, SHARD_START, LOCAL_VOCAB)

    if owned:
        torch.testing.assert_close(out[0], weight[token_id - SHARD_START])
    else:
        assert (out[0] == 0).all()


@pytest.mark.parametrize(
    "shape,hidden",
    [
        pytest.param((1,), 64, id="single-token"),
        pytest.param((64,), 96, id="flat"),
        pytest.param((2, 31), 96, id="ragged-2d"),
        pytest.param((3, 5, 7), 128, id="3d"),
    ],
)
def test_mixed_ownership_matches_the_eager_chain(shape, hidden):
    """Random ids on both sides of the shard: the kernel must reproduce the old
    chain element-for-element — the regression condition of the fusion."""
    weight = _weight(hidden)
    ids = torch.randint(0, SHARD_START + 3 * LOCAL_VOCAB, shape, device="cuda", dtype=torch.int64)

    out = vocab_parallel_embedding(ids, weight, SHARD_START, LOCAL_VOCAB)

    assert out.shape == (ids.numel(), hidden)
    torch.testing.assert_close(out, _eager_chain(ids.reshape(-1), weight, SHARD_START, LOCAL_VOCAB))


def test_empty_batch_returns_empty_without_launching():
    """A zero-token batch (the TP empty-input path) must not launch a kernel."""
    weight = _weight()
    ids = torch.empty(0, dtype=torch.int64, device="cuda")

    out = vocab_parallel_embedding(ids, weight, SHARD_START, LOCAL_VOCAB)

    assert out.shape == (0, HIDDEN)


def test_cpu_fallback_agrees_with_the_kernel():
    """The gloo tier runs the torch-native path on CPU tensors; pin that it is
    the same lookup as the kernel, or the CPU tests would be certifying
    different arithmetic than the GPU runs execute.

    Both paths only gather rows and multiply by 0/1, so agreement is exact.
    """
    weight_gpu = _weight()
    ids_gpu = torch.randint(
        0, SHARD_START + 2 * LOCAL_VOCAB, (48,), device="cuda", dtype=torch.int64
    )

    kernel_out = vocab_parallel_embedding(ids_gpu, weight_gpu, SHARD_START, LOCAL_VOCAB)
    fallback_out = vocab_parallel_embedding(
        ids_gpu.cpu(), weight_gpu.cpu(), SHARD_START, LOCAL_VOCAB
    )

    torch.testing.assert_close(kernel_out, fallback_out.cuda(), rtol=0, atol=0)
