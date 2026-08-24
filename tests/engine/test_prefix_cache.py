"""Tests for prefix caching — pure CPU, no GPU or checkpoint required.

Covers the block-hash prefix cache in isolation and its integration with the
scheduler's admission path (shared prompt prefix reduces prefill work).
"""

from __future__ import annotations

from lite_llama.engine.prefix_cache import PrefixCache
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
# PrefixCache unit
# --------------------------------------------------------------------------- #
class TestPrefixCache:
    def test_empty_cache_zero_hit(self):
        cache = PrefixCache(block_size=4)
        assert cache.query(list(range(16))) == 0

    def test_register_then_full_hit_minus_partial(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(16))  # 4 full blocks
        cache.register(tokens)
        assert cache.query(tokens) == 16

    def test_shared_prefix_partial_hit(self):
        cache = PrefixCache(block_size=4)
        a = list(range(16))
        cache.register(a)
        # Same first 8 tokens, then diverges.
        b = list(range(8)) + [99, 98, 97, 96, 95, 94, 93, 92]
        assert cache.query(b) == 8  # 2 shared blocks

    def test_divergence_at_first_block_zero(self):
        cache = PrefixCache(block_size=4)
        cache.register(list(range(16)))
        assert cache.query([100, 101, 102, 103, 0, 1, 2, 3]) == 0

    def test_partial_trailing_block_ignored(self):
        cache = PrefixCache(block_size=4)
        cache.register([0, 1, 2, 3, 4, 5])  # 1 full block + 2 leftover
        assert cache.query([0, 1, 2, 3, 4, 5]) == 4  # only the full block

    def test_release_keeps_blocks_cached_for_later_hits(self):
        """Blocks survive release (LRU persistence); only capacity pressure evicts."""
        cache = PrefixCache(block_size=4)
        tokens = list(range(8))
        cache.register(tokens)
        cache.release(tokens)
        # Blocks stay resident (LRU) even after the holder releases.
        assert cache.query(tokens) == 8
        assert cache.num_cached_blocks == 2
        assert cache.num_evictable_blocks == 2  # zero ref_cnt

    def test_capacity_evicts_lru_unreferenced_blocks(self):
        """Capacity-bounded cache evicts LRU unreferenced blocks on overflow."""
        cache = PrefixCache(block_size=4, capacity=2)
        tokens_a = list(range(8))     # 2 blocks
        tokens_b = list(range(8, 16)) # 2 different blocks
        cache.register(tokens_a)
        cache.release(tokens_a)       # ref_cnt -> 0, stays resident
        cache.register(tokens_b)      # adds 2 more -> 4 > capacity 2 -> evicts a's
        assert cache.query(tokens_a) == 0  # evicted
        assert cache.query(tokens_b) == 8  # still resident

    def test_refcount_survives_one_release(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(8))
        cache.register(tokens)
        cache.register(tokens)  # two holders
        cache.release(tokens)
        assert cache.query(tokens) == 8  # still cached

    def test_hit_rate_metric(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(16))
        cache.register(tokens)
        cache.query(tokens)  # 16/16 cached
        assert cache.hit_rate == 1.0

    def test_referenced_blocks_are_never_evicted(self):
        """A live holder protects its blocks even under capacity pressure."""
        cache = PrefixCache(block_size=4, capacity=1)
        held = list(range(8))          # 2 blocks, kept referenced
        other = list(range(100, 108))  # 2 blocks
        cache.register(held)           # ref_cnt = 1, must survive
        cache.register(other)
        cache.release(other)           # only these are evictable
        cache.register(list(range(200, 208)))  # force more pressure
        # The referenced prefix is still fully hittable.
        assert cache.query(held) == 8
        assert cache.num_referenced_blocks >= 2

    def test_register_returns_preexisting_hit_length(self):
        """register() reports the reuse the caller gets, so callers need no 2nd query."""
        cache = PrefixCache(block_size=4)
        tokens = list(range(16))
        assert cache.register(tokens) == 0   # cold
        assert cache.register(tokens) == 16  # warm: full prefix already cached

    def test_hash_seed_isolates_caches(self):
        """Different seeds must not cross-hit on identical token ids."""
        a = PrefixCache(block_size=4, hash_seed=1)
        b = PrefixCache(block_size=4, hash_seed=2)
        tokens = list(range(16))
        a.register(tokens)
        assert a.query(tokens) == 16
        assert b.query(tokens) == 0

    def test_reset_clears_blocks_and_stats(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(16))
        cache.register(tokens)
        cache.query(tokens)
        cache.reset()
        assert cache.num_cached_blocks == 0
        assert cache.hit_rate == 0.0
        assert cache.stats.num_requests == 0

    def test_stats_counters(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(16))
        cache.register(tokens)
        cache.query(tokens)
        assert cache.stats.num_requests == 1
        assert cache.stats.queried_tokens == 16
        assert cache.stats.hit_tokens == 16

    def test_eviction_counter(self):
        cache = PrefixCache(block_size=4, capacity=1)
        cache.register(list(range(8)))
        cache.release(list(range(8)))
        cache.register(list(range(100, 108)))
        assert cache.stats.evictions >= 1


# --------------------------------------------------------------------------- #
# Scheduler integration
# --------------------------------------------------------------------------- #
class TestSchedulerPrefixCache:
    def _sched(self, enable: bool) -> Scheduler:
        config = SchedulerConfig(
            max_seq_len=4096,
            max_num_seqs=8,
            max_num_batched_tokens=65536,
            max_chunk_size=0,
            enable_prefix_cache=enable,
        )
        return Scheduler(config, num_slots=8)

    def test_disabled_no_caching(self):
        sched = self._sched(enable=False)
        shared = list(range(64))
        sched.add_request(_req("a", shared))
        sched.schedule()
        sched.add_request(_req("b", shared))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.num_cached_tokens == 0

    def test_second_request_hits_shared_prefix(self):
        sched = self._sched(enable=True)
        shared = list(range(64))  # 4 blocks of 16

        sched.add_request(_req("a", shared))
        out_a = sched.schedule()
        a = next(r for r in out_a.prefill if r.request_id == "a")
        # First request populates the cache; it cannot skip its own prefill.
        assert a.num_cached_tokens == 0

        sched.add_request(_req("b", shared))
        out_b = sched.schedule()
        b = next(r for r in out_b.prefill if r.request_id == "b")
        # Second request reuses the prefix (all but the last token guaranteed).
        assert b.num_cached_tokens >= 48
        assert b.num_cached_tokens < b.prompt_len

    def test_hit_rate_positive_after_shared(self):
        sched = self._sched(enable=True)
        shared = list(range(64))
        for name in ("a", "b", "c"):
            sched.add_request(_req(name, shared))
            sched.schedule()
        assert sched.prefix_cache_hit_rate > 0.0

    def test_divergent_prompts_no_hit(self):
        sched = self._sched(enable=True)
        sched.add_request(_req("a", list(range(64))))
        sched.schedule()
        sched.add_request(_req("b", list(range(1000, 1064))))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.num_cached_tokens == 0
