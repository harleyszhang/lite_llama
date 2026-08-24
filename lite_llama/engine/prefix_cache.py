"""Prefix caching: reuse KV of shared prompt prefixes across requests.

Design mirrors vLLM v1's ``BlockPool`` + ``KVCacheBlock`` + free-block LRU queue:

* A prompt is split into fixed-size blocks. Each block's hash **chains** the
  previous block's hash, so one hash identifies a *prefix* (this block plus all
  before it), not just the block's own tokens -- the same trick as vLLM's
  ``hash_block_tokens(parent_hash, block_tokens)``.
* A cached block carries a **reference count**. ``ref_cnt == 0`` does NOT evict
  it: like vLLM, a released block stays resident and remains hittable, moving to
  the tail of an **LRU** order. It is only physically dropped when the pool is
  over ``capacity`` and the block is the least-recently-used unreferenced one.

That LRU persistence is what makes a shared system prompt pay off: the first
request populates the cache and *finishes*, yet the next request still hits the
prefix instead of re-prefilling it.

Usage:
    cache = PrefixCache(block_size=16, capacity=4096)
    cache.register(req_a.prompt_token_ids)     # ref_cnt +1 per block
    hit = cache.query(req_b.prompt_token_ids)  # -> cached leading tokens
    cache.release(req_a.prompt_token_ids)      # ref_cnt -1; stays cached (LRU)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


def _iter_block_hashes(token_ids: list[int], block_size: int, seed: int):
    """Yield one chained hash per full block of *token_ids*.

    Each block's hash folds in the previous block's hash (and a per-cache
    ``seed``), so identical hashes imply identical prefixes rather than merely
    identical block contents at some offset. A trailing partial block is skipped:
    a half-filled block's KV is not reusable until it is complete.
    """
    parent = seed
    num_full = len(token_ids) // block_size
    for b in range(num_full):
        block = tuple(token_ids[b * block_size : (b + 1) * block_size])
        parent = hash((parent, block))
        yield parent


@dataclass
class PrefixCacheStats:
    """Cumulative counters for prefix-cache effectiveness (mirrors vLLM's stats).

    Attributes:
        num_requests: How many prompts were queried.
        queried_tokens: Total prompt tokens looked up.
        hit_tokens: Prompt tokens served from cache.
        evictions: Blocks physically dropped under capacity pressure.
    """

    num_requests: int = 0
    queried_tokens: int = 0
    hit_tokens: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of queried prompt tokens served from cache (0.0 - 1.0)."""
        if self.queried_tokens == 0:
            return 0.0
        return self.hit_tokens / self.queried_tokens


@dataclass
class _CachedBlock:
    """One resident prefix block: its chained hash and live reference count."""

    block_hash: int
    ref_cnt: int = 0


class PrefixCache:
    """Ref-counted, LRU-evicted store of cached prefix blocks.

    The block map doubles as the LRU order: it is an ``OrderedDict`` whose tail
    is most-recently-used. A hit or a fresh reference moves a block to the tail;
    eviction (only when over ``capacity``) removes unreferenced blocks from the
    head. Referenced blocks (``ref_cnt > 0``) are never evicted.

    Args:
        block_size: Tokens per block. Larger blocks hash cheaper but match
            coarser; 16 mirrors vLLM's default page granularity.
        capacity: Maximum resident blocks. ``None`` means unbounded (no
            eviction) -- set it in memory-constrained deployments.
        hash_seed: Salt folded into every block hash, isolating one cache's
            hashes from another's (e.g. per-tenant) to avoid cross-hits.
    """

    def __init__(
        self,
        block_size: int = 16,
        capacity: int | None = None,
        hash_seed: int = 0,
    ) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be >= 1 or None, got {capacity}")
        self.block_size = block_size
        self.capacity = capacity
        self.hash_seed = hash_seed
        #: block-hash -> _CachedBlock, ordered least- to most-recently-used.
        self._blocks: OrderedDict[int, _CachedBlock] = OrderedDict()
        self.stats = PrefixCacheStats()

    # ------------------------------------------------------------------ lookup #
    def query(self, token_ids: list[int]) -> int:
        """Return the number of leading tokens already cached for this prompt.

        Walks the prompt's block hashes from the front and stops at the first
        block that is not cached: prefix reuse must be contiguous from token 0,
        exactly as a KV cache read is. Hit blocks are refreshed to most-recently
        -used so a hot prefix is not evicted out from under future requests.
        """
        self.stats.num_requests += 1
        self.stats.queried_tokens += len(token_ids)
        cached = 0
        for h in _iter_block_hashes(token_ids, self.block_size, self.hash_seed):
            block = self._blocks.get(h)
            if block is None:
                break
            self._blocks.move_to_end(h)  # LRU touch
            cached += self.block_size
        self.stats.hit_tokens += cached
        return cached

    # -------------------------------------------------------------- mutation #
    def register(self, token_ids: list[int]) -> int:
        """Reference every full block of *token_ids*, creating missing ones.

        Returns the cached-prefix length (in tokens) that existed *before* this
        call -- i.e. the reuse this request enjoys. Newly created blocks may
        trigger LRU eviction of unreferenced blocks when over capacity.
        """
        hit = 0
        counting_hit = True
        for h in _iter_block_hashes(token_ids, self.block_size, self.hash_seed):
            block = self._blocks.get(h)
            if block is None:
                counting_hit = False
                block = _CachedBlock(block_hash=h)
                self._blocks[h] = block
            elif counting_hit:
                hit += self.block_size
            block.ref_cnt += 1
            self._blocks.move_to_end(h)  # newest / just-touched -> MRU
        self._evict_to_capacity()
        return hit

    def release(self, token_ids: list[int]) -> None:
        """Drop one reference per block; blocks stay cached (LRU) at zero.

        Unlike a naive cache, hitting ``ref_cnt == 0`` does not evict: the block
        remains resident and hittable until capacity pressure reclaims it, so a
        finished request still leaves its prefix warm for the next one.
        """
        for h in _iter_block_hashes(token_ids, self.block_size, self.hash_seed):
            block = self._blocks.get(h)
            if block is not None and block.ref_cnt > 0:
                block.ref_cnt -= 1

    def _evict_to_capacity(self) -> None:
        """Evict least-recently-used *unreferenced* blocks until within capacity."""
        if self.capacity is None:
            return
        # Iterate a snapshot of keys from LRU (front) to MRU (back).
        for h in list(self._blocks.keys()):
            if len(self._blocks) <= self.capacity:
                break
            if self._blocks[h].ref_cnt == 0:
                del self._blocks[h]
                self.stats.evictions += 1

    def reset(self) -> None:
        """Drop all cached blocks and zero the stats (e.g. between benchmarks)."""
        self._blocks.clear()
        self.stats = PrefixCacheStats()

    # ------------------------------------------------------------------ views #
    @property
    def hit_rate(self) -> float:
        """Cumulative fraction of queried prompt tokens served from cache."""
        return self.stats.hit_rate

    @property
    def num_cached_blocks(self) -> int:
        """Total resident blocks (referenced + evictable)."""
        return len(self._blocks)

    @property
    def num_referenced_blocks(self) -> int:
        """Blocks a live request still holds (``ref_cnt > 0``, never evicted)."""
        return sum(1 for b in self._blocks.values() if b.ref_cnt > 0)

    @property
    def num_evictable_blocks(self) -> int:
        """Resident blocks with no live reference (LRU-eviction candidates)."""
        return sum(1 for b in self._blocks.values() if b.ref_cnt == 0)
