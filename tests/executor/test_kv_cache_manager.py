"""Tests for :class:`~lite_llama.executor.kv_cache_manager.KVCacheManager`.

Pure CPU against a tiny pool: alloc marks rows and debits the counter,
over-allocation returns None and changes nothing, contiguous allocs
find runs or fail on fragmentation, and ref-counts free the last holder.

Usage:
    pytest tests/executor/test_kv_cache_manager.py
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
        kv_row=(8, 64),  # 4 kv heads, K and V
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


def test_mla_cache_rows_hold_one_latent_vector():
    """MLA: K and V share one latent row — no head axis, no factor of two."""
    mgr = KVCacheManager(
        num_layers=2,
        kv_row=(1, 576),  # kv_lora_rank 512 + qk_rope_head_dim 64
        gpu_num_blocks=_BLOCKS,
        dtype=torch.float32,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    assert mgr.gpu_kv_buffer[0].shape == (_BLOCKS, 1, 576)
    per_token = mgr.kv_row[0] * mgr.kv_row[1] * mgr.num_layers
    assert per_token == 2 * 576  # not 2 * kv_heads * head_dim: the row *is* K and V


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


# --------------------------------------------------------------------------- #
# Admission watermark with hysteresis (O9)
# --------------------------------------------------------------------------- #
def _pressured_manager() -> KVCacheManager:
    """100 rows, watermark 30%, recovery band 10% — thresholds at 30 and 40."""
    return KVCacheManager(
        num_layers=1,
        kv_row=(8, 64),
        gpu_num_blocks=100,
        dtype=torch.float32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        watermark=0.3,
        hysteresis=0.1,
    )


def test_admission_refuses_only_under_the_watermark():
    """Above the watermark the single threshold still governs."""
    mgr = _pressured_manager()
    assert mgr.can_admit(65)  # 100 - 65 = 35 >= 30
    assert not mgr.can_admit(71)  # 100 - 71 = 29 < 30


def test_dip_under_the_watermark_raises_the_bar_until_recovery():
    """After a dip, admission resumes only above watermark + recovery band."""
    mgr = _pressured_manager()
    mgr.alloc_kvcache(75)  # 25 free, under the 30-row watermark
    assert not mgr.can_admit(0)

    mgr.free(torch.arange(0, 10, device=mgr.device))  # 35 free: above the watermark...
    assert not mgr.can_admit(0)  # ...but under watermark + band (40): still refused

    mgr.free(torch.arange(10, 15, device=mgr.device))  # 40 free: recovered
    assert mgr.can_admit(0)


def test_level_oscillating_around_the_watermark_cannot_flap_admission():
    """A level bouncing 25 ↔ 35 stays refused; a single threshold would flap."""
    mgr = _pressured_manager()
    mgr.alloc_kvcache(75)  # 25 free
    assert not mgr.can_admit(0)  # trips the pressure latch
    for _ in range(2):
        mgr.free(torch.arange(0, 10, device=mgr.device))  # 35
        assert not mgr.can_admit(0)
        mgr.alloc_kvcache(10)  # 25 again
        assert not mgr.can_admit(0)

    mgr.free(torch.arange(10, 25, device=mgr.device))  # 40: recovered
    assert mgr.can_admit(0)
