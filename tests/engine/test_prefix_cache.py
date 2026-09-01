"""Tests for prefix caching — pure CPU, no GPU or checkpoint required.

One test class per behaviour: hashing, matching, reference counting,
LRU eviction, capacity boundaries, cache isolation, and the scheduler
integration that reuses copied blocks.

Usage:
    pytest tests/engine/test_prefix_cache.py
"""

from __future__ import annotations

import pytest

from lite_llama.engine.prefix_cache import PrefixCache, PrefixCacheStats
from lite_llama.engine.sampler import SamplingParams
from lite_llama.engine.scheduler import Request, Scheduler, SchedulerConfig


def _req(request_id: str, token_ids: list[int]) -> Request:
    return Request(
        request_id=request_id,
        prompt="x" * len(token_ids),
        prompt_token_ids=token_ids,
        params=SamplingParams(temperature=0.0, max_gen_len=32),
    )


# --------------------------------------------------------------------------- #
# 1. Block hashing & boundary conditions
# --------------------------------------------------------------------------- #
class TestPrefixHashing:
    """Block granularity: full blocks only, trailing partial is ignored."""

    @pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16])
    def test_empty_cache_is_always_zero_hit(self, block_size):
        cache = PrefixCache(block_size=block_size)
        assert cache.query(list(range(64))) == 0

    @pytest.mark.parametrize(
        ("prompt_len", "block_size", "expected_blocks"),
        [(16, 4, 4), (17, 4, 4), (15, 4, 3), (32, 8, 4), (33, 8, 4)],
    )
    def test_register_creates_exactly_full_blocks(self, prompt_len, block_size, expected_blocks):
        cache = PrefixCache(block_size=block_size)
        cache.register(list(range(prompt_len)))
        assert cache.num_cached_blocks == expected_blocks

    def test_trailing_partial_block_is_not_cached(self):
        """A half-filled block's KV is not reusable until it is complete."""
        cache = PrefixCache(block_size=4)
        cache.register([0, 1, 2, 3, 4, 5])  # 1 full + 2 leftover
        assert cache.query([0, 1, 2, 3, 4, 5]) == 4  # only the full block

    def test_prompt_shorter_than_one_block_caches_nothing(self):
        cache = PrefixCache(block_size=16)
        cache.register([0, 1, 2])
        assert cache.num_cached_blocks == 0
        assert cache.query([0, 1, 2]) == 0

    def test_single_token_prompt(self):
        cache = PrefixCache(block_size=1)
        cache.register([42])
        assert cache.query([42]) == 1

    def test_empty_prompt_is_a_no_op(self):
        cache = PrefixCache(block_size=4)
        cache.register([])
        assert cache.num_cached_blocks == 0
        assert cache.query([]) == 0


# --------------------------------------------------------------------------- #
# 2. Prefix matching semantics
# --------------------------------------------------------------------------- #
class TestPrefixMatching:
    """Prefix reuse is contiguous from token 0, exactly as a KV cache read is."""

    def test_full_hit_after_register(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(16))
        cache.register(tokens)
        assert cache.query(tokens) == 16

    def test_shared_prefix_partial_hit(self):
        """Same first 8 tokens, then diverges — 2 shared blocks hit."""
        cache = PrefixCache(block_size=4)
        a = list(range(16))
        cache.register(a)
        b = [*range(8), 99, 98, 97, 96, 95, 94, 93, 92]
        assert cache.query(b) == 8

    @pytest.mark.parametrize("diverge_at", [0, 1, 2, 3])
    def test_divergence_at_various_depths(self, diverge_at):
        """Hit length must equal diverge_at * block_size (or 0 at depth 0)."""
        block_size = 4
        cache = PrefixCache(block_size=block_size)
        original = list(range(16))
        cache.register(original)
        # Replace tokens starting at the given block boundary.
        modified = original[: diverge_at * block_size] + [
            200 + i for i in range(16 - diverge_at * block_size)
        ]
        expected = diverge_at * block_size
        assert cache.query(modified) == expected

    def test_divergence_at_first_block_is_zero(self):
        cache = PrefixCache(block_size=4)
        cache.register(list(range(16)))
        assert cache.query([100, 101, 102, 103, 0, 1, 2, 3]) == 0

    def test_repeating_tokens_dont_false_match(self):
        """Identical blocks at different offsets must not cross-hit.

        Mirrors SGLang's ``test_hash_value_repeating_tokens``: a block of all-7s
        at position 0 has a different chained hash than one at position 2.
        """
        cache = PrefixCache(block_size=4)
        cache.register([7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7])  # blocks 0,1,2
        # A prompt whose first block is all-7s must NOT match block 2's all-7s,
        # because block 2's hash chains block 1's hash (all-8s).
        assert cache.query([7, 7, 7, 7, 9, 9, 9, 9]) == 4  # only block 0

    def test_long_prefix_chain_hit(self):
        """A deep prefix (50 blocks) is fully hittable after one register."""
        block_size = 4
        cache = PrefixCache(block_size=block_size)
        tokens = list(range(200))
        cache.register(tokens)
        assert cache.query(tokens) == 200


# --------------------------------------------------------------------------- #
# 3. Reference counting
# --------------------------------------------------------------------------- #
class TestReferenceCounting:
    """Multiple live holders protect blocks; release decrements without evicting."""

    def test_multiple_registers_accumulate_ref_cnt(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(8))
        cache.register(tokens)
        cache.register(tokens)  # two holders
        assert cache.num_referenced_blocks == 2
        assert cache.num_evictable_blocks == 0

    def test_one_release_among_multiple_holders_keeps_blocks(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(8))
        cache.register(tokens)
        cache.register(tokens)
        cache.release(tokens)  # one holder drops; the other still holds
        assert cache.query(tokens) == 8
        assert cache.num_referenced_blocks == 2

    def test_all_releases_make_blocks_evictable_but_not_evicted(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(8))
        cache.register(tokens)
        cache.register(tokens)
        cache.release(tokens)
        cache.release(tokens)  # ref_cnt -> 0 for both blocks
        assert cache.num_referenced_blocks == 0
        assert cache.num_evictable_blocks == 2
        assert cache.num_cached_blocks == 2  # still resident (LRU)

    def test_release_beyond_register_count_does_not_go_negative(self):
        """Over-releasing must be a safe no-op, not an underflow."""
        cache = PrefixCache(block_size=4)
        tokens = list(range(8))
        cache.register(tokens)
        cache.release(tokens)
        cache.release(tokens)  # already 0 — must not crash or go negative
        assert cache.num_referenced_blocks == 0
        assert cache.num_cached_blocks == 2

    def test_register_returns_preexisting_hit_length(self):
        """register() reports the reuse the caller gets (no separate query needed)."""
        cache = PrefixCache(block_size=4)
        tokens = list(range(16))
        assert cache.register(tokens) == 0  # cold
        assert cache.register(tokens) == 16  # warm: full prefix already cached


# --------------------------------------------------------------------------- #
# 4. LRU eviction
# --------------------------------------------------------------------------- #
class TestLRUEviction:
    """LRU order, touch-on-hit, capacity-based eviction of unreferenced blocks."""

    def test_release_keeps_blocks_cached_for_later_hits(self):
        """The core vLLM behaviour: a finished request's prefix stays warm."""
        cache = PrefixCache(block_size=4)
        tokens = list(range(8))
        cache.register(tokens)
        cache.release(tokens)
        assert cache.query(tokens) == 8
        assert cache.num_cached_blocks == 2
        assert cache.num_evictable_blocks == 2

    def test_query_touches_block_to_mru_so_it_surves_pressure(self):
        """A hit refreshes LRU position; a hot prefix is not evicted under pressure."""
        cache = PrefixCache(block_size=4, capacity=4)
        warm = list(range(8))  # 2 blocks
        cold = list(range(100, 108))  # 2 blocks
        cache.register(warm)
        cache.release(warm)  # warm -> evictable
        cache.register(cold)
        cache.release(cold)  # cold -> evictable; 4 blocks total
        # Touch warm blocks (move to MRU) so cold is LRU.
        cache.query(warm)
        # Add 2 new blocks -> 6 > capacity 4 -> evict 2 LRU unreferenced.
        cache.register(list(range(200, 208)))
        # Warm survived (touched to MRU); cold was evicted.
        assert cache.query(warm) == 8
        assert cache.query(cold) == 0

    def test_capacity_evicts_lru_unreferenced_blocks(self):
        cache = PrefixCache(block_size=4, capacity=2)
        tokens_a = list(range(8))
        tokens_b = list(range(8, 16))
        cache.register(tokens_a)
        cache.release(tokens_a)
        cache.register(tokens_b)  # 4 > 2 -> evicts a's blocks
        assert cache.query(tokens_a) == 0
        assert cache.query(tokens_b) == 8

    def test_referenced_blocks_are_never_evicted(self):
        """A live holder protects its blocks even under capacity pressure."""
        cache = PrefixCache(block_size=4, capacity=1)
        held = list(range(8))
        cache.register(held)  # ref_cnt = 1
        cache.register(list(range(100, 108)))
        cache.release(list(range(100, 108)))
        cache.register(list(range(200, 208)))
        assert cache.query(held) == 8
        assert cache.num_referenced_blocks >= 2

    def test_eviction_counter(self):
        cache = PrefixCache(block_size=4, capacity=1)
        cache.register(list(range(8)))
        cache.release(list(range(8)))
        cache.register(list(range(100, 108)))
        assert cache.stats.evictions >= 1


# --------------------------------------------------------------------------- #
# 5. Capacity boundary conditions
# --------------------------------------------------------------------------- #
class TestCapacityBoundary:
    """Extreme capacity values: 1 (minimum), None (unbounded), exact fit."""

    def test_capacity_one_evicts_aggressively(self):
        cache = PrefixCache(block_size=4, capacity=1)
        cache.register(list(range(8)))
        cache.release(list(range(8)))
        cache.register(list(range(100, 104)))  # 1 block -> fits exactly
        assert cache.num_cached_blocks == 1

    def test_capacity_none_is_unbounded(self):
        cache = PrefixCache(block_size=4, capacity=None)
        for i in range(20):
            cache.register(list(range(i * 4, i * 4 + 4)))
        assert cache.num_cached_blocks == 20
        assert cache.stats.evictions == 0

    def test_exact_capacity_no_eviction(self):
        cache = PrefixCache(block_size=4, capacity=3)
        cache.register(list(range(12)))  # 3 blocks
        assert cache.num_cached_blocks == 3
        assert cache.stats.evictions == 0

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            PrefixCache(block_size=4, capacity=0)

    def test_invalid_block_size_raises(self):
        with pytest.raises(ValueError):
            PrefixCache(block_size=0)


# --------------------------------------------------------------------------- #
# 6. Cache isolation & reset
# --------------------------------------------------------------------------- #
class TestCacheIsolation:
    """hash_seed salting and reset semantics."""

    def test_hash_seed_isolates_caches(self):
        """Different seeds must not cross-hit on identical token ids."""
        a = PrefixCache(block_size=4, hash_seed=1)
        b = PrefixCache(block_size=4, hash_seed=2)
        tokens = list(range(16))
        a.register(tokens)
        assert a.query(tokens) == 16
        assert b.query(tokens) == 0

    def test_default_seed_is_zero_and_isolates_from_nonzero(self):
        """Default seed=0 must not collide with a non-zero seed."""
        default = PrefixCache(block_size=4)
        seeded = PrefixCache(block_size=4, hash_seed=99)
        tokens = list(range(16))
        default.register(tokens)
        assert default.query(tokens) == 16
        assert seeded.query(tokens) == 0

    def test_reset_clears_blocks_and_stats(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(16))
        cache.register(tokens)
        cache.query(tokens)
        cache.reset()
        assert cache.num_cached_blocks == 0
        assert cache.hit_rate == 0.0
        assert cache.stats.num_requests == 0
        assert cache.stats.evictions == 0

    def test_reset_preserves_config(self):
        """reset() drops data but keeps block_size/capacity for reuse."""
        cache = PrefixCache(block_size=8, capacity=10)
        cache.register(list(range(64)))
        cache.reset()
        # Config is still valid; new registrations work.
        cache.register(list(range(16)))
        assert cache.num_cached_blocks == 2  # 16 / 8 = 2 blocks


# --------------------------------------------------------------------------- #
# 7. Statistics
# --------------------------------------------------------------------------- #
class TestStatistics:
    """Cumulative counters and derived metrics."""

    def test_stats_counters_after_single_query(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(16))
        cache.register(tokens)
        cache.query(tokens)
        assert cache.stats.num_requests == 1
        assert cache.stats.queried_tokens == 16
        assert cache.stats.hit_tokens == 16

    def test_hit_rate_full_hit(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(16))
        cache.register(tokens)
        cache.query(tokens)
        assert cache.hit_rate == 1.0

    def test_hit_rate_mixed_queries(self):
        cache = PrefixCache(block_size=4)
        shared = list(range(16))
        cache.register(shared)
        cache.query(shared)  # 16 hit / 16 queried
        cache.query(list(range(100, 116)))  # 0 hit / 16 queried
        assert cache.stats.num_requests == 2
        assert cache.stats.queried_tokens == 32
        assert cache.stats.hit_tokens == 16
        assert cache.hit_rate == 0.5

    def test_hit_rate_zero_when_no_queries(self):
        cache = PrefixCache(block_size=4)
        assert cache.hit_rate == 0.0

    def test_stats_is_prefixcachestats_instance(self):
        cache = PrefixCache(block_size=4)
        assert isinstance(cache.stats, PrefixCacheStats)


# --------------------------------------------------------------------------- #
# 8. Scheduler integration
# --------------------------------------------------------------------------- #
class TestSchedulerIntegration:
    """Prefix cache wired into the scheduler's admission/finish/preempt path."""

    def _sched(
        self,
        *,
        enable_prefix_cache: bool = True,
        enable_preemption: bool = False,
        max_chunk_size: int = 0,
        max_num_seqs: int = 8,
        num_slots: int = 8,
    ) -> Scheduler:
        config = SchedulerConfig(
            max_seq_len=4096,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=65536,
            max_chunk_size=max_chunk_size,
            enable_prefix_cache=enable_prefix_cache,
            enable_preemption=enable_preemption,
        )
        return Scheduler(config, num_slots=num_slots)

    def test_disabled_cache_means_zero_cached_tokens(self):
        sched = self._sched(enable_prefix_cache=False)
        shared = list(range(64))
        sched.add_request(_req("a", shared))
        sched.schedule()
        sched.add_request(_req("b", shared))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.num_cached_tokens == 0

    def test_second_request_hits_shared_prefix(self):
        sched = self._sched()
        shared = list(range(64))  # 4 blocks of 16
        sched.add_request(_req("a", shared))
        out_a = sched.schedule()
        a = next(r for r in out_a.prefill if r.request_id == "a")
        assert a.num_cached_tokens == 0  # first request: cold

        sched.add_request(_req("b", shared))
        out_b = sched.schedule()
        b = next(r for r in out_b.prefill if r.request_id == "b")
        assert b.num_cached_tokens >= 48  # at least 3 of 4 blocks
        assert b.num_cached_tokens < b.prompt_len  # never skips the whole prompt

    def test_divergent_prompts_get_zero_cached(self):
        sched = self._sched()
        sched.add_request(_req("a", list(range(64))))
        sched.schedule()
        sched.add_request(_req("b", list(range(1000, 1064))))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.num_cached_tokens == 0

    def test_hit_rate_grows_with_shared_requests(self):
        sched = self._sched()
        shared = list(range(64))
        rates: list[float] = []
        for name in ("a", "b", "c"):
            sched.add_request(_req(name, shared))
            sched.schedule()
            rates.append(sched.prefix_cache_hit_rate)
        assert rates[0] == 0.0  # first request: cold
        assert rates[1] > 0.0  # second: hit
        assert rates[2] >= rates[1]  # third: hit rate monotonically non-decreasing

    def test_finish_releases_prefix_but_it_stays_cached(self):
        """After the first request finishes, its prefix survives (LRU persistence)."""
        sched = self._sched()
        shared = list(range(64))
        sched.add_request(_req("a", shared))
        out = sched.schedule()
        a = out.prefill[0]
        sched.finish(a, "eos")

        # req b arrives after a finished — should still hit the shared prefix.
        sched.add_request(_req("b", shared))
        out_b = sched.schedule()
        b = next(r for r in out_b.prefill if r.request_id == "b")
        assert b.num_cached_tokens >= 48

    def test_preempted_request_resets_cached_tokens(self):
        """Preemption clears num_cached_tokens; recompute starts from scratch."""
        sched = self._sched(
            enable_preemption=True,
            max_num_seqs=3,
            num_slots=2,
        )
        shared = list(range(64))
        for i in range(3):
            sched.add_request(_req(f"r{i}", shared))
        # Run until a preemption occurs.
        for _ in range(6):
            out = sched.schedule()
            for r in out.decode:
                r.output_token_ids.append(999)
            if out.preempted:
                # Preempted requests must have their cached tokens reset.
                for p in out.preempted:
                    assert p.num_cached_tokens == 0
                return
        pytest.fail("expected at least one preemption within 6 steps")

    def test_prefix_cache_with_chunked_prefill(self):
        """Reuse is capped by what the sharer has actually executed, not hashed.

        ``a``'s whole prompt is hashed the moment it is admitted, so all 8 blocks
        are cache *hits* for ``b``. Only the first 4 are reusable: reuse means
        copying real K/V out of a slot, and when ``b`` arrives ``a`` has executed
        one 64-token chunk, so rows for its second chunk hold nothing yet. ``b``
        recomputes the rest and becomes their owner in turn.
        """
        sched = self._sched(max_chunk_size=64)
        shared = list(range(128))  # 8 blocks of 16
        sched.add_request(_req("a", shared))
        sched.schedule()  # plans a's first chunk and registers its prompt
        # req b shares prefix — its uncached remainder is smaller.
        sched.add_request(_req("b", shared + list(range(1000, 1064))))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.num_cached_tokens == 64  # a's executed chunk, 4 of 8 blocks
        assert b.prefix_copies == ((0, 0, 64),)  # copied out of a's slot
        assert b.num_computed_tokens == 64 + 64  # reused prefix + this chunk


# --------------------------------------------------------------------------- #
# 9. Block copy: which hits are actually reusable, and where from
# --------------------------------------------------------------------------- #
class TestBlockCopyReuse:
    """A hit is bookkeeping; reuse needs K/V that some slot still holds.

    Without a block table there is no indirection to share rows through, so a
    matched prefix is reused by copying it into the requester's own slot. That
    splits every hit in two: ``num_tokens`` (the hash matched) and
    ``copyable_tokens`` (a slot still holds the K/V). These tests pin the gap
    between them, since it is the whole reason prefill cannot simply be skipped.
    """

    BLOCK = 16

    def _cache(self) -> PrefixCache:
        return PrefixCache(block_size=self.BLOCK, capacity=256)

    def _sched(self) -> Scheduler:
        config = SchedulerConfig(
            max_seq_len=4096,
            max_num_seqs=8,
            max_num_batched_tokens=65536,
            max_chunk_size=0,
            enable_prefix_cache=True,
            enable_preemption=False,
        )
        return Scheduler(config, num_slots=8)

    # ------------------------------------------------------------ ownership #
    def test_a_hit_with_no_live_copy_is_not_reusable(self):
        """Hashes outlive K/V, so a full hit can still be worth nothing."""
        cache = self._cache()
        prompt = list(range(64))
        assert cache.register(prompt) == 0  # first sight: nothing to hit yet
        match = cache.admit(prompt)
        assert match.num_tokens == 64  # every block is a hit
        assert match.copyable_tokens == 0  # no slot ever claimed them
        assert match.segments == ()

    def test_owned_blocks_merge_into_one_run(self):
        """Adjacent blocks of one owner cost one copy, not one copy per block."""
        cache = self._cache()
        prompt = list(range(64))
        cache.register(prompt)
        cache.assign_owner(prompt, slot=3, upto_tokens=64)
        match = cache.admit(prompt)
        assert match.copyable_tokens == 64
        assert match.segments == ((3, 0, 64),)  # 4 blocks, 1 run

    def test_invalidating_the_slot_keeps_the_hit_but_kills_the_copy(self):
        """Handing a slot on costs the copy source, not the cache entry."""
        cache = self._cache()
        prompt = list(range(64))
        cache.register(prompt)
        cache.assign_owner(prompt, slot=3, upto_tokens=64)
        cache.invalidate_slot(3)
        match = cache.admit(prompt)
        assert match.num_tokens == 64  # the hash is still true
        assert match.copyable_tokens == 0  # nobody holds the K/V now

    def test_ownership_reaches_only_as_far_as_the_owner_executed(self):
        """Claims are bounded by computed tokens, not by prompt length."""
        cache = self._cache()
        prompt = list(range(64))
        cache.register(prompt)
        cache.assign_owner(prompt, slot=1, upto_tokens=32)  # two chunks in
        match = cache.admit(prompt)
        assert match.num_tokens == 64
        assert match.copyable_tokens == 32
        assert match.segments == ((1, 0, 32),)

    def test_a_run_ends_where_the_owner_changes(self):
        """Two owners of one prefix produce two copies, in prompt order."""
        cache = self._cache()
        prompt = list(range(64))
        cache.register(prompt)
        cache.assign_owner(prompt, slot=0, upto_tokens=32)  # blocks 0-1
        # Blocks 0-1 already have an owner and keep it; only 2-3 are up for grabs.
        cache.assign_owner(prompt, slot=1, upto_tokens=64)
        match = cache.admit(prompt)
        assert match.copyable_tokens == 64
        assert match.segments == ((0, 0, 32), (1, 32, 32))

    def test_reuse_stops_at_the_first_block_without_a_copy(self):
        """A slot is read from row 0 up, so a gap ends the reuse for good.

        Blocks 0-1 and 3 have live copies but block 2 does not. Block 3 cannot be
        used despite being both cached and resident: attention over rows
        ``[0, n)`` has no way to skip the hole in the middle.
        """
        cache = self._cache()
        prompt = list(range(64))
        cache.register(prompt)
        cache.assign_owner(prompt, slot=0, upto_tokens=32)  # blocks 0-1
        cache.assign_owner(prompt, slot=1, upto_tokens=48)  # block 2
        cache.assign_owner(prompt, slot=2, upto_tokens=64)  # block 3
        cache.invalidate_slot(1)  # block 2 loses its only copy
        match = cache.admit(prompt)
        assert match.num_tokens == 64  # all four still hit
        assert match.copyable_tokens == 32  # reuse stops at the hole
        assert match.segments == ((0, 0, 32),)

    # ------------------------------------------------------------- scheduler #
    def test_same_step_admissions_cannot_share(self):
        """K/V exists a step after it is planned, so the twin gets nothing.

        Both requests are admitted in one call, so when ``b`` is matched ``a``'s
        first chunk has been *planned* and nothing more. Offering ``a``'s rows
        here would have ``b`` attend over cache the model has not written.
        """
        sched = self._sched()
        shared = list(range(64))
        sched.add_request(_req("a", shared))
        sched.add_request(_req("b", shared))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.num_cached_tokens == 0
        assert b.prefix_copies == ()

    def test_landing_on_the_slot_holding_the_prefix_copies_nothing(self):
        """The cheapest hit: the rows are already where attention will look.

        ``a`` frees slot 0 and ``b`` is handed it straight back, so ``b``'s prefix
        is bit-for-bit already in place -- reuse without moving a byte. Note the
        one token short of the whole prompt: something must run to produce logits.
        """
        sched = self._sched()
        shared = list(range(64))
        sched.add_request(_req("a", shared))
        a = sched.schedule().prefill[0]
        freed = a.slot  # finish() clears it
        sched.finish(a, "eos")
        sched.add_request(_req("b", shared))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.slot == freed  # freed slot handed straight back
        assert b.num_cached_tokens == 63  # whole prompt bar the last token
        assert b.prefix_copies == ()  # nothing to move
        assert b.num_computed_tokens == 64  # reused 63 + the 1-token chunk

    def test_copying_from_a_running_sharer_names_its_slot(self):
        """A different slot means a real copy, sourced from the sharer's rows."""
        sched = self._sched()
        shared = list(range(64))
        sched.add_request(_req("a", shared))
        a = sched.schedule().prefill[0]  # plans a's whole prompt
        sched.add_request(_req("b", shared + list(range(100, 120))))
        out = sched.schedule()  # promotes a's ownership first
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.slot != a.slot
        assert b.num_cached_tokens == 64
        assert b.prefix_copies == ((a.slot, 0, 64),)
        assert b.num_computed_tokens == 84  # 64 reused + 20 own tokens

    def test_a_slot_reused_by_a_stranger_stops_being_a_source(self):
        """``c`` must not inherit rows that ``b`` overwrote with its own prompt.

        ``a`` computes a prefix in slot 0 and finishes; ``b`` takes slot 0 with an
        unrelated prompt and refills it. ``c`` shares ``a``'s prefix and still
        hits on hash, but there is no live copy left to take it from.
        """
        sched = self._sched()
        shared = list(range(64))
        sched.add_request(_req("a", shared))
        a = sched.schedule().prefill[0]
        freed = a.slot  # finish() clears it
        sched.finish(a, "eos")
        sched.add_request(_req("b", list(range(500, 564))))
        b = next(r for r in sched.schedule().prefill if r.request_id == "b")
        assert b.slot == freed  # b overwrites the rows a left
        sched.add_request(_req("c", shared))
        c = next(r for r in sched.schedule().prefill if r.request_id == "c")
        assert c.num_cached_tokens == 0
        assert c.prefix_copies == ()
