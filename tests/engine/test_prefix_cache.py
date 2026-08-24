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

    def test_release_evicts(self):
        cache = PrefixCache(block_size=4)
        tokens = list(range(8))
        cache.register(tokens)
        cache.release(tokens)
        assert cache.query(tokens) == 0
        assert cache.num_cached_blocks == 0

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
