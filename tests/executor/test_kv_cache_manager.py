"""Tests for :class:`KVCacheManager`.

The manager hands out cache rows and tracks refcounts. Two things make it worth
testing carefully:

* **The bump-allocator fast path.** ``alloc_kvcache_index`` normally answers from
  a cursor with no device reads, because ``generate()`` opens with ``free_all``
  and the cache is then append-only. The cursor is only valid while that holds:
  any partial free leaves holes, so it must fall back to searching and must not
  resume until ``free_all``. A cursor that stayed "exact" after a partial free
  would hand out rows that are still in use -- two sequences sharing KV, which
  shows up as garbled text far from here.
* **Accounting.** ``can_use_mem_size`` drives the "can I admit this request?"
  decision. If it drifts from the real occupancy the engine either OOMs or
  refuses work it could do.

These run on CPU when no GPU is present: the manager allocates real KV buffers,
so the tiny sizes here keep it cheap either way.
"""

from __future__ import annotations

import pytest
import torch

from lite_llama.executor.kv_cache_manager import KVCacheManager

_BLOCKS = 9


@pytest.fixture
def manager() -> KVCacheManager:
    """A 9-row pool; small enough that exhaustion cases are easy to express."""
    return KVCacheManager(
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        gpu_num_blocks=_BLOCKS,
        dtype=torch.float32,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


# --------------------------------------------------------------------------- #
# Initial state and buffers
# --------------------------------------------------------------------------- #
def test_starts_empty(manager):
    assert manager.can_use_mem_size == _BLOCKS
    assert manager.kv_mem_use_state.numel() == _BLOCKS
    assert not manager.kv_mem_use_state.any()


def test_allocates_one_kv_buffer_per_layer(manager):
    """Layers must not share a buffer, or layer N would read layer N-1's K/V."""
    assert len(manager.gpu_kv_buffer) == 2
    # 2x heads: K heads then V heads live in one tensor.
    assert manager.gpu_kv_buffer[0].shape == (_BLOCKS, 8, 64)
    assert manager.gpu_kv_buffer[0].data_ptr() != manager.gpu_kv_buffer[1].data_ptr()


# --------------------------------------------------------------------------- #
# alloc_kvcache / alloc_contiguous_kvcache
# --------------------------------------------------------------------------- #
def test_alloc_marks_rows_used_and_debits_the_counter(manager):
    index = manager.alloc_kvcache(3)
    assert index is not None
    assert index.numel() == 3
    assert (manager.kv_mem_use_state[index] == 1).all()
    assert manager.can_use_mem_size == _BLOCKS - 3


def test_alloc_beyond_capacity_returns_none_and_changes_nothing(manager):
    """Refusal must be total: a partial allocation would leak rows."""
    assert manager.alloc_kvcache(_BLOCKS + 1) is None
    assert not manager.kv_mem_use_state.any()
    assert manager.can_use_mem_size == _BLOCKS


def test_alloc_contiguous_returns_a_consecutive_run(manager):
    result = manager.alloc_contiguous_kvcache(4)
    assert result is not None
    index, start, end = result
    assert index.numel() == 4
    assert end - start == 4
    assert index.tolist() == list(range(start, end))
    assert manager.can_use_mem_size == _BLOCKS - 4


def test_alloc_contiguous_fails_when_the_pool_is_fragmented(manager):
    """Enough free rows, but no run long enough: the request must be refused.

    Distinguishes a real contiguity search from a plain free-count check.
    """
    manager.alloc_kvcache(5)  # takes rows 0-4
    manager.kv_mem_use_state[7] = 1  # hole at 7 leaves runs of length 2 and 1
    manager.can_use_mem_size -= 1

    assert manager.alloc_contiguous_kvcache(3) is None


def test_alloc_contiguous_reuses_rows_after_a_partial_release(manager):
    index, _, _ = manager.alloc_contiguous_kvcache(4)
    manager.release_ref(index[:2])  # free the first two rows

    reused = manager.alloc_contiguous_kvcache(2)
    assert reused is not None
    assert reused[0].numel() == 2
    assert manager.can_use_mem_size == _BLOCKS - 4


# --------------------------------------------------------------------------- #
# Refcounting
# --------------------------------------------------------------------------- #
def test_release_frees_rows_and_credits_the_counter(manager):
    index = manager.alloc_kvcache(3)
    manager.release_ref(index)
    assert not manager.kv_mem_use_state[index].any()
    assert manager.can_use_mem_size == _BLOCKS


def test_second_reference_keeps_the_row_alive(manager):
    """A row shared by two owners must survive the first release.

    This is what makes prefix sharing safe; freeing on the first release would
    hand a live row to the next request.
    """
    index = manager.alloc_kvcache(2)
    manager.add_ref(index)
    assert (manager.kv_mem_use_state[index] == 2).all()

    manager.release_ref(index)
    assert (manager.kv_mem_use_state[index] == 1).all()
    assert manager.can_use_mem_size == _BLOCKS - 2

    manager.release_ref(index)
    assert not manager.kv_mem_use_state[index].any()
    assert manager.can_use_mem_size == _BLOCKS


def test_free_all_resets_every_row(manager):
    manager.alloc_kvcache(5)
    manager.free_all()
    assert not manager.kv_mem_use_state.any()
    assert manager.can_use_mem_size == _BLOCKS


# --------------------------------------------------------------------------- #
# Bump allocator (alloc_kvcache_index)
# --------------------------------------------------------------------------- #
def test_bump_hands_out_consecutive_rows_from_zero(manager):
    """The fast path is append-only, so successive calls must not overlap."""
    first = manager.alloc_kvcache_index(3)
    second = manager.alloc_kvcache_index(2)

    assert first.tolist() == [0, 1, 2]
    assert second.tolist() == [3, 4]
    assert manager.can_use_mem_size == _BLOCKS - 5


def test_bump_returns_int32_rows(manager):
    """Callers index the cache with int32; a silent int64 would break the kernels."""
    assert manager.alloc_kvcache_index(2).dtype == torch.int32


def test_bump_marks_rows_used(manager):
    index = manager.alloc_kvcache_index(4)
    assert (manager.kv_mem_use_state[index.long()] == 1).all()


def test_partial_free_disables_the_bump_fast_path(manager):
    """After a hole appears the cursor is no longer the free list.

    Continuing to bump would re-hand rows that are still referenced. The
    fallback search must reuse the freed rows instead.
    """
    first = manager.alloc_kvcache_index(4)  # rows 0-3
    manager.release_ref(first[:2].long())  # free rows 0-1
    assert manager._bump_is_exact is False

    reused = manager.alloc_kvcache_index(2)
    assert sorted(reused.tolist()) == [0, 1]


def test_free_all_restores_the_bump_fast_path(manager):
    """``generate()`` opens with ``free_all``, which is what makes bumping valid."""
    manager.alloc_kvcache_index(3)
    manager.release_ref(torch.tensor([0], device=manager.device))
    assert manager._bump_is_exact is False

    manager.free_all()
    assert manager._bump_is_exact is True
    assert manager._bump_cursor == 0
    assert manager.alloc_kvcache_index(2).tolist() == [0, 1]


def test_bump_falls_back_when_the_cursor_reaches_the_end(manager):
    """Exhausting the cursor must fall back to searching, not run off the end."""
    manager.alloc_kvcache_index(_BLOCKS)  # cursor now at capacity
    manager.free_all()
    manager.alloc_kvcache_index(_BLOCKS - 2)

    # Two rows remain and the cursor cannot serve three, so the search path runs.
    assert manager.alloc_kvcache_index(2).numel() == 2


def test_repeated_generate_cycles_do_not_leak(manager):
    """Ten free_all/alloc rounds must return the pool to its initial state.

    Mirrors what ten ``generate()`` calls do, which is the leak the end-to-end
    test can only detect as an eventual OOM.
    """
    for _ in range(10):
        manager.free_all()
        index = manager.alloc_kvcache_index(4)
        assert index.numel() == 4

    manager.free_all()
    assert manager.can_use_mem_size == _BLOCKS
    assert not manager.kv_mem_use_state.any()
