"""Tests for the physical KV block pool — pure CPU, no GPU or checkpoint.

One class per behaviour of :mod:`lite_llama.engine.block_pool`: the free
queue's O(1) surgery, reference counting, the LRU order that decides what is
evicted, hash indexing, and the reset safety check.

Usage:
    pytest tests/engine/test_block_pool.py
"""

from __future__ import annotations

import pytest

from lite_llama.engine.block_pool import (
    NULL_BLOCK_ID,
    BlockPool,
    FreeBlockQueue,
    KVCacheBlock,
)


def _ids(blocks) -> list[int]:
    return [block.block_id for block in blocks]


# --------------------------------------------------------------------------- #
# 1. The free queue
# --------------------------------------------------------------------------- #
class TestFreeBlockQueue:
    """A doubly linked list whose two ends encode the eviction policy."""

    def test_append_order_is_iteration_order(self):
        blocks = [KVCacheBlock(block_id=i) for i in range(4)]
        queue = FreeBlockQueue(blocks)
        assert _ids(queue) == [0, 1, 2, 3]
        assert queue.num_free_blocks == 4

    def test_popleft_takes_the_front(self):
        queue = FreeBlockQueue([KVCacheBlock(block_id=i) for i in range(3)])
        assert queue.popleft().block_id == 0
        assert _ids(queue) == [1, 2]

    def test_prepend_jumps_the_queue(self):
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        queue = FreeBlockQueue(blocks)
        spare = KVCacheBlock(block_id=9)
        queue.prepend(spare)
        assert _ids(queue) == [9, 0, 1, 2]

    def test_remove_unlinks_from_the_middle(self):
        blocks = [KVCacheBlock(block_id=i) for i in range(4)]
        queue = FreeBlockQueue(blocks)
        queue.remove(blocks[2])
        assert _ids(queue) == [0, 1, 3]
        assert queue.num_free_blocks == 3

    def test_removed_block_can_be_appended_again(self):
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        queue = FreeBlockQueue(blocks)
        queue.remove(blocks[0])
        queue.append(blocks[0])
        assert _ids(queue) == [1, 2, 0]

    def test_popleft_on_empty_queue_raises(self):
        queue = FreeBlockQueue([])
        with pytest.raises(IndexError):
            queue.popleft()

    def test_removing_an_absent_block_raises(self):
        queue = FreeBlockQueue([KVCacheBlock(block_id=0)])
        with pytest.raises(ValueError, match="not in the free queue"):
            queue.remove(KVCacheBlock(block_id=7))


# --------------------------------------------------------------------------- #
# 2. Construction and the null block
# --------------------------------------------------------------------------- #
class TestPoolConstruction:
    """Block 0 is reserved; everything else starts free."""

    def test_null_block_is_never_free(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        assert pool.null_block.block_id == NULL_BLOCK_ID
        assert pool.null_block.ref_cnt == 1
        assert pool.num_free_blocks == 7
        assert NULL_BLOCK_ID not in _ids(pool.free_block_queue)

    def test_null_block_is_never_handed_out(self):
        pool = BlockPool(num_blocks=4, block_size=4)
        blocks = pool.get_new_blocks(3)
        assert blocks is not None
        assert NULL_BLOCK_ID not in _ids(blocks)
        assert pool.get_new_blocks(1) is None

    def test_rows_of_maps_a_block_to_its_cache_rows(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        assert list(pool.rows_of(2)) == list(range(32, 48))

    @pytest.mark.parametrize("num_blocks", [0, 1])
    def test_pool_needs_a_spare_block_beyond_the_null_one(self, num_blocks):
        with pytest.raises(ValueError, match="at least 2 blocks"):
            BlockPool(num_blocks=num_blocks, block_size=4)

    def test_block_size_must_be_positive(self):
        with pytest.raises(ValueError, match="block_size must be"):
            BlockPool(num_blocks=4, block_size=0)


# --------------------------------------------------------------------------- #
# 3. Allocation and reference counting
# --------------------------------------------------------------------------- #
class TestAllocation:
    """Blocks are handed out at ref_cnt 1 and recycled at 0."""

    def test_new_blocks_start_referenced_once(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        blocks = pool.get_new_blocks(2)
        assert [block.ref_cnt for block in blocks] == [1, 1]
        assert pool.num_free_blocks == 5

    def test_short_pool_allocates_nothing_at_all(self):
        pool = BlockPool(num_blocks=4, block_size=4)
        assert pool.get_new_blocks(4) is None
        # The failed request must not have consumed anything on the way out.
        assert pool.num_free_blocks == 3

    def test_zero_blocks_is_a_valid_request(self):
        pool = BlockPool(num_blocks=4, block_size=4)
        assert pool.get_new_blocks(0) == []

    def test_negative_request_raises(self):
        pool = BlockPool(num_blocks=4, block_size=4)
        with pytest.raises(ValueError, match="cannot allocate"):
            pool.get_new_blocks(-1)

    def test_touch_shares_a_block_between_requests(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        blocks = pool.get_new_blocks(1)
        pool.touch(blocks)
        assert blocks[0].ref_cnt == 2
        pool.free_blocks(blocks)
        # Still held by the other request, so not reusable yet.
        assert blocks[0].ref_cnt == 1
        assert pool.num_free_blocks == 6

    def test_touching_a_free_block_pulls_it_out_of_the_queue(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        blocks = pool.get_new_blocks(1)
        pool.free_blocks(blocks)
        free_before = pool.num_free_blocks
        pool.touch(blocks)
        assert blocks[0].ref_cnt == 1
        assert pool.num_free_blocks == free_before - 1

    def test_touching_the_null_block_leaves_it_alone(self):
        """A windowed group's hit carries null blocks; they must not be counted."""
        pool = BlockPool(num_blocks=8, block_size=4)
        pool.touch([pool.null_block, pool.null_block])
        assert pool.null_block.ref_cnt == 1
        assert pool.num_free_blocks == 7

    def test_freeing_the_null_block_is_a_no_op(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        pool.free_blocks([pool.null_block])
        assert pool.null_block.ref_cnt == 1
        assert pool.num_free_blocks == 7

    def test_double_free_raises_rather_than_corrupting_the_count(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        blocks = pool.get_new_blocks(1)
        pool.free_blocks(blocks)
        with pytest.raises(ValueError, match="freed more times than referenced"):
            pool.free_blocks(blocks)


# --------------------------------------------------------------------------- #
# 4. LRU order
# --------------------------------------------------------------------------- #
class TestEvictionOrder:
    """Worthless blocks go first, cached ones least-recently-freed first."""

    def test_uncached_blocks_are_reused_before_cached_ones(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        cached, plain = pool.get_new_blocks(1), pool.get_new_blocks(1)
        pool.cache_full_blocks(cached, [1234])
        pool.free_blocks(cached)  # cached -> back of the queue
        pool.free_blocks(plain)  # hash-less -> front of the queue
        assert _ids(pool.free_block_queue)[0] == plain[0].block_id
        assert _ids(pool.free_block_queue)[-1] == cached[0].block_id

    def test_cached_blocks_evict_in_the_order_they_were_freed(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        first, second = pool.get_new_blocks(1), pool.get_new_blocks(1)
        pool.cache_full_blocks(first, [11])
        pool.cache_full_blocks(second, [22])
        pool.free_blocks(first)
        pool.free_blocks(second)
        order = _ids(pool.free_block_queue)
        assert order.index(first[0].block_id) < order.index(second[0].block_id)

    def test_allocation_evicts_the_least_recently_used_cached_block(self):
        pool = BlockPool(num_blocks=3, block_size=4)  # 2 usable blocks
        first, second = pool.get_new_blocks(1), pool.get_new_blocks(1)
        pool.cache_full_blocks(first, [11])
        pool.cache_full_blocks(second, [22])
        pool.free_blocks(first)
        pool.free_blocks(second)

        taken = pool.get_new_blocks(1)
        assert taken[0] is first[0]
        assert pool.get_cached_block(11) is None  # evicted
        assert pool.get_cached_block(22) is second[0]  # still hittable
        assert pool.stats.evictions == 1

    def test_eviction_forgets_the_block_hash(self):
        pool = BlockPool(num_blocks=3, block_size=4)
        block = pool.get_new_blocks(1)
        pool.cache_full_blocks(block, [11])
        pool.free_blocks(block)
        assert pool.get_new_blocks(1)[0].block_hash is None


# --------------------------------------------------------------------------- #
# 5. The hash index
# --------------------------------------------------------------------------- #
class TestHashIndex:
    """cache_full_blocks is what makes a block reusable, and only once."""

    def test_cached_block_is_found_by_hash(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        blocks = pool.get_new_blocks(2)
        pool.cache_full_blocks(blocks, [11, 22])
        assert pool.get_cached_block(11) is blocks[0]
        assert pool.get_cached_block(22) is blocks[1]
        assert pool.num_cached_blocks == 2

    def test_unknown_hash_misses(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        assert pool.get_cached_block(999) is None

    def test_recaching_an_indexed_block_is_a_no_op(self):
        """Committing every step must cost work only for the new blocks."""
        pool = BlockPool(num_blocks=8, block_size=4)
        blocks = pool.get_new_blocks(1)
        pool.cache_full_blocks(blocks, [11])
        pool.cache_full_blocks(blocks, [99])
        assert pool.get_cached_block(11) is blocks[0]
        assert pool.get_cached_block(99) is None

    def test_caching_is_disabled_when_the_pool_says_so(self):
        pool = BlockPool(num_blocks=8, block_size=4, enable_caching=False)
        blocks = pool.get_new_blocks(1)
        pool.cache_full_blocks(blocks, [11])
        assert pool.get_cached_block(11) is None
        assert pool.num_cached_blocks == 0
        # Allocation and recycling still work; only reuse is off.
        pool.free_blocks(blocks)
        assert pool.num_free_blocks == 7

    def test_two_blocks_may_share_a_hash_without_losing_either(self):
        """One block being freed while another already caches the same prefix."""
        pool = BlockPool(num_blocks=3, block_size=4)  # 2 usable blocks
        first, second = pool.get_new_blocks(1), pool.get_new_blocks(1)
        pool.cache_full_blocks(first, [11])
        pool.cache_full_blocks(second, [11])
        assert pool.num_cached_blocks == 2
        pool.free_blocks(first)
        pool.get_new_blocks(1)  # evicts `first`, whose hash `second` still holds
        assert pool.get_cached_block(11) is second[0]

    def test_hash_count_ignores_partial_blocks(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        blocks = pool.get_new_blocks(3)
        pool.cache_full_blocks(blocks[:2], [11, 22])
        assert pool.num_cached_blocks == 2


# --------------------------------------------------------------------------- #
# 6. Reset
# --------------------------------------------------------------------------- #
class TestReset:
    """A reset that cannot free capacity refuses instead of lying."""

    def test_reset_drops_every_cached_block(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        blocks = pool.get_new_blocks(2)
        pool.cache_full_blocks(blocks, [11, 22])
        pool.free_blocks(blocks)

        assert pool.reset_prefix_cache() is True
        assert pool.num_cached_blocks == 0
        assert pool.get_cached_block(11) is None
        assert pool.num_free_blocks == 7
        assert pool.stats.resets == 1

    def test_reset_refuses_while_a_request_holds_a_block(self):
        pool = BlockPool(num_blocks=8, block_size=4)
        blocks = pool.get_new_blocks(1)
        pool.cache_full_blocks(blocks, [11])

        assert pool.reset_prefix_cache() is False
        assert pool.get_cached_block(11) is blocks[0]
        assert pool.stats.resets == 0

    def test_reset_forgets_every_block_hash(self):
        pool = BlockPool(num_blocks=5, block_size=4)
        blocks = pool.get_new_blocks(4)
        pool.cache_full_blocks(blocks, [1, 2, 3, 4])
        pool.free_blocks(reversed(blocks))
        pool.reset_prefix_cache()
        assert all(block.block_hash is None for block in pool.blocks)
        # Capacity is untouched: a reset drops the index, not the blocks.
        assert pool.num_free_blocks == 4
        assert sorted(_ids(pool.free_block_queue)) == [1, 2, 3, 4]
