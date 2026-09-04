"""Tests for KV cache groups and their hit policies — pure CPU, no GPU.

Covers what :mod:`rapid_llm.engine.kv_cache_spec` decides: how far back a
prefix is reusable for each attention kind, how a prompt's hash chain projects
onto groups that page at different sizes, and how the coordinator reduces N
groups to the one number the scheduler wants.

Usage:
    pytest tests/engine/test_kv_cache_spec.py
"""

from __future__ import annotations

import pytest

from rapid_llm.engine.block_pool import BlockPool
from rapid_llm.engine.kv_cache_spec import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheCoordinator,
    KVCacheGroup,
    MLASpec,
    SlidingWindowSpec,
    cdiv,
)


def _cache(pool: BlockPool, hashes: list[int]) -> list[int]:
    """Fill the pool with one cached block per hash, and free them again.

    Returns the block ids in prefix order, so a test can assert *which*
    physical blocks a hit returned.
    """
    blocks = pool.get_new_blocks(len(hashes))
    assert blocks is not None
    pool.cache_full_blocks(blocks, hashes)
    pool.free_blocks(blocks)
    return [block.block_id for block in blocks]


def _ids(blocks) -> list[int]:
    return [block.block_id for block in blocks]


# --------------------------------------------------------------------------- #
# 1. Full attention: an unbroken run from token 0
# --------------------------------------------------------------------------- #
class TestFullAttentionHit:
    def test_whole_prompt_hits(self):
        pool = BlockPool(num_blocks=16, block_size=4)
        ids = _cache(pool, [11, 22, 33])
        spec = FullAttentionSpec(block_size=4)
        hit = spec.find_longest_cache_hit([11, 22, 33], max_length=12, pool=pool, hash_block_size=4)
        assert _ids(hit) == ids

    def test_first_miss_ends_the_run(self):
        """Attention reads from row 0 up, so a gap cannot be skipped over."""
        pool = BlockPool(num_blocks=16, block_size=4)
        _cache(pool, [11])
        _cache(pool, [33])  # cached, but block 2 (hash 22) is not
        spec = FullAttentionSpec(block_size=4)
        hit = spec.find_longest_cache_hit([11, 22, 33], max_length=12, pool=pool, hash_block_size=4)
        assert len(hit) == 1

    def test_max_length_caps_the_hit_at_whole_blocks(self):
        pool = BlockPool(num_blocks=16, block_size=4)
        _cache(pool, [11, 22, 33])
        spec = FullAttentionSpec(block_size=4)
        # 11 tokens is two whole blocks plus three: the partial block is not
        # shareable, so the hit stops at 8.
        hit = spec.find_longest_cache_hit([11, 22, 33], max_length=11, pool=pool, hash_block_size=4)
        assert len(hit) == 2

    def test_empty_cache_misses(self):
        pool = BlockPool(num_blocks=16, block_size=4)
        spec = FullAttentionSpec(block_size=4)
        assert spec.find_longest_cache_hit([11, 22], 8, pool, 4) == []

    def test_mla_uses_the_same_policy(self):
        """MLA differs in layout, not in what a hit means."""
        pool = BlockPool(num_blocks=16, block_size=4)
        ids = _cache(pool, [11, 22])
        spec = MLASpec(block_size=4, kv_row=(1, 576))
        hit = spec.find_longest_cache_hit([11, 22], 8, pool, 4)
        assert _ids(hit) == ids


# --------------------------------------------------------------------------- #
# 2. Hash-chain projection onto coarser groups
# --------------------------------------------------------------------------- #
class TestGroupBlockHashes:
    def test_same_block_size_passes_the_chain_through(self):
        spec = FullAttentionSpec(block_size=4)
        assert spec.group_block_hashes([1, 2, 3], 4) == [1, 2, 3]

    def test_coarser_group_takes_every_r_th_hash(self):
        """The chain already folds in the blocks before, so no rehashing."""
        spec = FullAttentionSpec(block_size=8)
        assert list(spec.group_block_hashes([1, 2, 3, 4, 5], 4)) == [2, 4]

    def test_indivisible_block_size_is_rejected(self):
        spec = FullAttentionSpec(block_size=6)
        with pytest.raises(ValueError, match="must be a multiple"):
            spec.group_block_hashes([1, 2], 4)

    def test_coarse_group_hits_only_whole_group_blocks(self):
        pool = BlockPool(num_blocks=16, block_size=8)
        # Hash blocks are 4 tokens; the group pages at 8, so its block 0 is
        # named by hash index 1.
        ids = _cache(pool, [2])
        spec = FullAttentionSpec(block_size=8)
        hit = spec.find_longest_cache_hit([1, 2, 3], max_length=24, pool=pool, hash_block_size=4)
        assert _ids(hit) == ids


# --------------------------------------------------------------------------- #
# 3. Sliding window: a hit is a window, and it may start anywhere
# --------------------------------------------------------------------------- #
class TestSlidingWindowHit:
    def test_window_block_count(self):
        spec = SlidingWindowSpec(block_size=4, sliding_window=9)
        # A query at p reads [p-8, p]: eight tokens back plus its own block.
        assert spec.num_window_blocks == cdiv(8, 4) + 1 == 3

    def test_a_complete_window_stops_the_scan_early(self):
        """Blocks the window cannot reach are left null rather than adopted."""
        pool = BlockPool(num_blocks=16, block_size=4)
        ids = _cache(pool, [11, 22, 33])
        spec = SlidingWindowSpec(block_size=4, sliding_window=5)  # 2 window blocks
        hit = spec.find_longest_cache_hit([11, 22, 33], 12, pool, 4)
        # A query at position 11 reads [7, 11], i.e. blocks 1 and 2; block 0 is
        # addressed but never read, so it costs no reference.
        assert hit[0] is pool.null_block
        assert _ids(hit[1:]) == ids[1:]

    def test_a_tail_run_hits_with_null_blocks_in_front(self):
        """Unlike full attention, an unmatched head does not kill the hit."""
        pool = BlockPool(num_blocks=16, block_size=4)
        _cache(pool, [22])  # block 0's hash (11) is not cached
        tail = _cache(pool, [33])
        spec = SlidingWindowSpec(block_size=4, sliding_window=5)  # 2 window blocks
        hit = spec.find_longest_cache_hit([11, 22, 33], 12, pool, 4)
        assert len(hit) == 3
        assert hit[0] is pool.null_block
        assert hit[2].block_id == tail[0]

    def test_a_tail_run_shorter_than_the_window_is_useless(self):
        """A query at the run's end still needs the positions before it."""
        pool = BlockPool(num_blocks=16, block_size=4)
        _cache(pool, [33])
        spec = SlidingWindowSpec(block_size=4, sliding_window=9)  # 3 window blocks
        assert spec.find_longest_cache_hit([11, 22, 33], 12, pool, 4) == []

    def test_a_prefix_run_shorter_than_the_window_still_hits(self):
        """Nothing precedes token 0, so a short run there *is* a whole window."""
        pool = BlockPool(num_blocks=16, block_size=4)
        ids = _cache(pool, [11])
        spec = SlidingWindowSpec(block_size=4, sliding_window=9)
        hit = spec.find_longest_cache_hit([11, 22, 33], 12, pool, 4)
        assert _ids(hit) == ids

    def test_the_run_is_trimmed_where_the_window_completes(self):
        """A hit stops at the first complete window; blocks past it are misses."""
        pool = BlockPool(num_blocks=16, block_size=4)
        _cache(pool, [11, 22])  # a complete 2-block window at blocks 0..1
        spec = SlidingWindowSpec(block_size=4, sliding_window=5)
        hit = spec.find_longest_cache_hit([11, 22, 33], 12, pool, 4)
        assert len(hit) == 2

    def test_nothing_cached_misses(self):
        pool = BlockPool(num_blocks=16, block_size=4)
        spec = SlidingWindowSpec(block_size=4, sliding_window=5)
        assert spec.find_longest_cache_hit([11, 22], 8, pool, 4) == []

    def test_window_must_be_positive(self):
        with pytest.raises(ValueError, match="sliding_window must be"):
            SlidingWindowSpec(block_size=4, sliding_window=0)

    def test_block_count_is_not_capped_by_the_window(self):
        """Table entries are indexed by absolute position, however far it moved."""
        spec = SlidingWindowSpec(block_size=4, sliding_window=5)
        assert spec.num_blocks_for(40) == 10


# --------------------------------------------------------------------------- #
# 4. Config derivation
# --------------------------------------------------------------------------- #
class _Config:
    """The bits of ModelConfig that decide the groups."""

    def __init__(self, **kwargs):
        self.num_layers = 4
        self.num_kv_heads = 8
        self.head_dim = 64
        self.is_mla = False
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64
        self.sliding_window = None
        self.sliding_window_layers = ()
        self.__dict__.update(kwargs)


class TestConfig:
    def test_homogeneous_is_one_full_attention_group(self):
        config = KVCacheConfig.homogeneous(block_size=16, num_layers=3)
        assert config.num_groups == 1
        assert config.is_homogeneous
        assert isinstance(config.groups[0].spec, FullAttentionSpec)
        assert config.groups[0].layer_ids == (0, 1, 2)

    def test_gqa_model_gets_one_full_attention_group(self):
        config = KVCacheConfig.from_model_config(_Config(), tp_size=2, block_size=16)
        assert config.num_groups == 1
        spec = config.groups[0].spec
        assert isinstance(spec, FullAttentionSpec)
        assert spec.kv_row == (2 * 4, 64)  # 8 kv heads over 2 ranks

    def test_mla_model_gets_one_latent_group_unsharded(self):
        config = KVCacheConfig.from_model_config(_Config(is_mla=True), tp_size=4)
        spec = config.groups[0].spec
        assert isinstance(spec, MLASpec)
        assert spec.kv_row == (1, 512 + 64)

    def test_kv_heads_must_divide_across_ranks(self):
        with pytest.raises(ValueError, match="do not divide"):
            KVCacheConfig.from_model_config(_Config(num_kv_heads=3), tp_size=2)

    def test_interleaved_window_model_splits_into_two_groups(self):
        config = KVCacheConfig.from_model_config(
            _Config(sliding_window=128, sliding_window_layers=(0, 2)), block_size=16
        )
        assert config.num_groups == 2
        assert not config.is_homogeneous
        window, full = config.groups
        assert isinstance(window.spec, SlidingWindowSpec)
        assert window.layer_ids == (0, 2)
        assert isinstance(full.spec, FullAttentionSpec)
        assert full.layer_ids == (1, 3)
        assert config.group_of_layer(2) == 0
        assert config.group_of_layer(3) == 1

    def test_window_without_named_layers_stays_one_group(self):
        config = KVCacheConfig.from_model_config(_Config(sliding_window=128))
        assert config.num_groups == 1
        assert isinstance(config.groups[0].spec, FullAttentionSpec)

    def test_unknown_layer_has_no_group(self):
        config = KVCacheConfig.homogeneous(block_size=16, num_layers=2)
        with pytest.raises(KeyError):
            config.group_of_layer(5)

    def test_group_block_size_must_be_a_multiple_of_the_hash_size(self):
        group = KVCacheGroup(0, FullAttentionSpec(block_size=6), (0,))
        with pytest.raises(ValueError, match="must be a multiple"):
            KVCacheConfig((group,), hash_block_size=4)

    def test_a_config_needs_a_group(self):
        with pytest.raises(ValueError, match="at least one group"):
            KVCacheConfig((), hash_block_size=4)

    def test_page_size_covers_every_layer(self):
        spec = FullAttentionSpec(block_size=16, kv_row=(8, 64))
        assert spec.page_size_bytes(num_layers=2, dtype_size=2) == 16 * 8 * 64 * 2 * 2


# --------------------------------------------------------------------------- #
# 5. The coordinator: N groups, one answer
# --------------------------------------------------------------------------- #
def _coordinator(pool: BlockPool, *specs, hash_block_size: int = 4) -> KVCacheCoordinator:
    groups = tuple(KVCacheGroup(i, spec, (i,)) for i, spec in enumerate(specs))
    return KVCacheCoordinator(pool, groups, hash_block_size)


class TestCoordinatorLookup:
    def test_single_group_hit_is_reported_in_tokens(self):
        pool = BlockPool(num_blocks=16, block_size=4)
        _cache(pool, [11, 22])
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        blocks, length = coord.find_longest_cache_hit([11, 22, 33], max_length=12)
        assert length == 8
        assert len(blocks[0]) == 2

    def test_hit_is_the_shortest_across_groups(self):
        """A step computes every layer or none, so the weakest group decides."""
        pool = BlockPool(num_blocks=32, block_size=4)
        _cache(pool, [11, 22])  # nothing cached for the third block
        coord = _coordinator(
            pool,
            FullAttentionSpec(block_size=4),
            SlidingWindowSpec(block_size=4, sliding_window=5),
        )
        _, length = coord.find_longest_cache_hit([11, 22, 33], max_length=12)
        assert length == 8

    def test_a_windowed_group_reports_the_span_it_addresses(self):
        """Its early blocks are null -- addressed, never read, never referenced."""
        pool = BlockPool(num_blocks=32, block_size=4)
        _cache(pool, [11, 22, 33])
        coord = _coordinator(
            pool,
            FullAttentionSpec(block_size=4),
            SlidingWindowSpec(block_size=4, sliding_window=5),
        )
        blocks, length = coord.find_longest_cache_hit([11, 22, 33], max_length=12)
        assert length == 12
        assert blocks[0][0] is not pool.null_block
        assert blocks[1][0] is pool.null_block

    def test_hit_is_aligned_to_the_coarsest_group(self):
        pool = BlockPool(num_blocks=32, block_size=8)
        # Group 0 pages at 4 and hits 3 blocks (12 tokens); group 1 pages at 8
        # and hits 1 block (8 tokens). 8 is a whole number of blocks in both.
        fine = FullAttentionSpec(block_size=4)
        coarse = FullAttentionSpec(block_size=8)
        pool_fine = BlockPool(num_blocks=32, block_size=4)
        _cache(pool_fine, [11, 22, 33])
        coord = _coordinator(pool_fine, fine, coarse)
        blocks, length = coord.find_longest_cache_hit([11, 22, 33], max_length=12)
        assert length == 8
        assert len(blocks[0]) == 2  # 8 tokens at 4 per block
        assert len(blocks[1]) == 1  # 8 tokens at 8 per block
        del pool

    def test_a_group_that_misses_zeroes_the_hit(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        _cache(pool, [11, 22])
        coord = _coordinator(
            pool,
            FullAttentionSpec(block_size=4),
            FullAttentionSpec(block_size=8),  # needs hash index 1 at its block 0
        )
        blocks, length = coord.find_longest_cache_hit([11], max_length=4)
        assert length == 0
        assert blocks == [[], []]


class TestCoordinatorAllocation:
    def test_allocation_covers_every_group(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4), FullAttentionSpec(block_size=8))
        assert coord.allocate("a", num_tokens=16) is True
        ids = coord.block_ids("a")
        assert len(ids[0]) == 4  # 16 tokens at 4 per block
        assert len(ids[1]) == 2  # 16 tokens at 8 per block

    def test_growing_a_request_only_adds_what_is_missing(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        coord.allocate("a", 8)
        first = coord.block_ids("a")[0]
        coord.allocate("a", 9)  # crosses into a third block
        grown = coord.block_ids("a")[0]
        assert grown[:2] == first
        assert len(grown) == 3

    def test_allocation_is_idempotent_within_a_block(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        coord.allocate("a", 5)
        before = pool.num_free_blocks
        assert coord.allocate("a", 8) is True
        assert pool.num_free_blocks == before

    def test_a_short_pool_allocates_nothing_at_all(self):
        pool = BlockPool(num_blocks=4, block_size=4)  # 3 usable
        coord = _coordinator(pool, FullAttentionSpec(block_size=4), FullAttentionSpec(block_size=4))
        assert coord.allocate("a", num_tokens=8) is False  # would need 4
        assert pool.num_free_blocks == 3
        assert coord.block_ids("a") == ((), ())

    def test_adopting_a_hit_shares_the_physical_blocks(self):
        """The whole point: reuse is a reference, not a copy."""
        pool = BlockPool(num_blocks=32, block_size=4)
        ids = _cache(pool, [11, 22])
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        blocks, length = coord.find_longest_cache_hit([11, 22], max_length=8)
        assert coord.allocate("a", length, blocks) is True
        assert list(coord.block_ids("a")[0]) == ids
        assert all(pool.blocks[i].ref_cnt == 1 for i in ids)

    def test_two_requests_share_the_same_blocks(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        ids = _cache(pool, [11, 22])
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        for request_id in ("a", "b"):
            blocks, length = coord.find_longest_cache_hit([11, 22], max_length=8)
            assert coord.allocate(request_id, length, blocks) is True
        assert coord.block_ids("a") == coord.block_ids("b")
        assert all(pool.blocks[i].ref_cnt == 2 for i in ids)

    def test_a_failed_adoption_leaves_the_pool_untouched(self):
        pool = BlockPool(num_blocks=4, block_size=4)  # 3 usable
        ids = _cache(pool, [11])
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        blocks, _ = coord.find_longest_cache_hit([11], max_length=4)
        # Wants 4 blocks in total but only 3 exist: the adopted reference must
        # be handed back, or a rejected admission would pin it forever.
        assert coord.allocate("a", 16, blocks) is False
        assert pool.blocks[ids[0]].ref_cnt == 0
        assert pool.num_free_blocks == 3
        assert pool.get_cached_block(11) is not None

    def test_adopting_twice_is_a_programming_error(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        _cache(pool, [11])
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        blocks, length = coord.find_longest_cache_hit([11], max_length=4)
        coord.allocate("a", length, blocks)
        with pytest.raises(ValueError, match="already holds blocks"):
            coord.allocate("a", length, blocks)

    def test_adoption_must_cover_every_group(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4), FullAttentionSpec(block_size=4))
        with pytest.raises(ValueError, match="one entry per group"):
            coord.allocate("a", 4, [[pool.null_block]])


class TestCoordinatorCaching:
    def test_only_whole_computed_blocks_are_indexed(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        coord.allocate("a", 10)
        coord.cache_blocks("a", [11, 22], num_computed_tokens=6)
        assert pool.get_cached_block(11) is not None
        assert pool.get_cached_block(22) is None

    def test_caching_resumes_where_it_left_off(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        coord.allocate("a", 8)
        coord.cache_blocks("a", [11, 22], 4)
        coord.cache_blocks("a", [11, 22], 8)
        assert pool.get_cached_block(22) is not None
        assert pool.num_cached_blocks == 2

    def test_caching_an_untracked_request_is_a_no_op(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        coord.cache_blocks("ghost", [11], 4)
        assert pool.num_cached_blocks == 0

    def test_each_group_indexes_at_its_own_granularity(self):
        pool = BlockPool(num_blocks=32, block_size=8)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4), FullAttentionSpec(block_size=8))
        coord.allocate("a", 8)
        coord.cache_blocks("a", [11, 22], 8)
        # Group 0 indexed two 4-token blocks, group 1 one 8-token block under
        # the chained hash of the second: three entries, two hashes.
        assert pool.get_cached_block(11) is not None
        assert pool.num_cached_blocks == 3


class TestCoordinatorRelease:
    def test_free_returns_every_block(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        coord.allocate("a", 16)
        coord.free("a")
        assert pool.num_free_blocks == 31
        assert coord.num_tracked_requests() == 0
        assert coord.block_ids("a") == ((),)

    def test_freeing_tail_first_puts_the_tail_nearer_eviction(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        coord.allocate("a", 8)
        head, tail = coord.block_ids("a")[0]
        coord.cache_blocks("a", [11, 22], 8)
        coord.free("a")
        order = [block.block_id for block in pool.free_block_queue]
        assert order.index(tail) < order.index(head)

    def test_freeing_an_unknown_request_is_a_no_op(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        coord.free("ghost")

    def test_a_shared_block_survives_one_holder_leaving(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        ids = _cache(pool, [11])
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        for request_id in ("a", "b"):
            blocks, length = coord.find_longest_cache_hit([11], 4)
            coord.allocate(request_id, length, blocks)
        coord.free("a")
        assert pool.blocks[ids[0]].ref_cnt == 1
        assert list(coord.block_ids("b")[0]) == ids


class TestWindowTrimming:
    def test_blocks_below_the_window_are_released_and_nulled(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, SlidingWindowSpec(block_size=4, sliding_window=5))
        coord.allocate("a", 20)
        before = pool.num_free_blocks
        coord.remove_skipped_blocks("a", num_computed_tokens=20)
        # Positions below 20 - 5 = 15 are dead, i.e. blocks 0..2 (rows < 12).
        assert pool.num_free_blocks == before + 3
        assert coord.block_ids("a")[0][:3] == (0, 0, 0)

    def test_trimming_is_idempotent(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, SlidingWindowSpec(block_size=4, sliding_window=5))
        coord.allocate("a", 20)
        coord.remove_skipped_blocks("a", 20)
        free = pool.num_free_blocks
        coord.remove_skipped_blocks("a", 20)
        assert pool.num_free_blocks == free

    def test_full_attention_groups_keep_everything(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, FullAttentionSpec(block_size=4))
        coord.allocate("a", 20)
        before = pool.num_free_blocks
        coord.remove_skipped_blocks("a", 20)
        assert pool.num_free_blocks == before

    def test_trimming_an_unknown_request_is_a_no_op(self):
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, SlidingWindowSpec(block_size=4, sliding_window=5))
        coord.remove_skipped_blocks("ghost", 20)

    def test_a_windowed_request_grows_past_its_freed_blocks(self):
        """The regression that a window-capped block count would cause."""
        pool = BlockPool(num_blocks=32, block_size=4)
        coord = _coordinator(pool, SlidingWindowSpec(block_size=4, sliding_window=5))
        coord.allocate("a", 20)
        coord.remove_skipped_blocks("a", 20)
        assert coord.allocate("a", 24) is True
        ids = coord.block_ids("a")[0]
        assert len(ids) == 6
        assert ids[-1] != 0
