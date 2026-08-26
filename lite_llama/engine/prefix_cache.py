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

import hashlib
import struct
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

#: Tokens per prefix-cache block; 16 mirrors vLLM's default page granularity.
#: It lives here rather than on the scheduler because everyone who computes a
#: block hash has to agree on it -- the replica's own cache and the DP router's
#: affinity index -- and only one of those owns a scheduler.
PREFIX_CACHE_BLOCK_SIZE = 16


def iter_block_hashes(token_ids: Sequence[int], block_size: int, seed: int = 0) -> Iterator[int]:
    """Yield one chained hash per full block of *token_ids*.

    Each block's hash folds in the previous block's hash (and a per-cache
    ``seed``), so identical hashes imply identical prefixes rather than merely
    identical block contents at some offset. A trailing partial block is skipped:
    a half-filled block's KV is not reusable until it is complete.

    The digest is ``blake2b`` rather than the builtin ``hash()`` because these
    values are a *cross-process contract*: the DP router hashes a prompt to find
    which replica already holds its prefix, and the replica hashes it again to
    look the blocks up. Builtin ``hash()`` is in fact stable for tuples of ints
    (``PYTHONHASHSEED`` randomises only str/bytes), but that is a property of one
    interpreter build rather than a promise, and a router that disagreed with its
    replicas would not raise -- it would quietly route every request as a miss.

    Args:
        token_ids: Prompt tokens, each fitting in 32 bits.
        block_size: Tokens per block. Must match every party that hashes.
        seed: Salt folded into the chain, isolating one cache's hashes from
            another's. Must match every party that hashes.

    Yields:
        One 64-bit hash per complete block, in prefix order.

    Raises:
        struct.error: A token id is negative or does not fit in 32 bits.
    """
    parent = seed & 0xFFFFFFFFFFFFFFFF
    layout = f"<Q{block_size}I"
    num_full = len(token_ids) // block_size
    for b in range(num_full):
        block = token_ids[b * block_size : (b + 1) * block_size]
        digest = hashlib.blake2b(struct.pack(layout, parent, *block), digest_size=8)
        parent = int.from_bytes(digest.digest(), "little")
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


@dataclass(frozen=True)
class PrefixMatch:
    """What a prompt may reuse, and where the reusable K/V must be copied from.

    Attributes:
        num_tokens: Leading prompt tokens whose blocks are cached, block-aligned.
            This is the hit-rate numerator, and it is *not* how much prefill can
            be skipped: a block can be cached (its hash is still true) while no
            slot holds its K/V any more.
        copyable_tokens: Leading prompt tokens whose K/V is both cached and still
            resident in some slot, hence reusable by copying instead of
            recomputing. Always ``<= num_tokens``, and always a prefix -- reuse
            stops at the first block without a live copy, because attention reads
            a slot's rows contiguously from 0.
        segments: ``(src_slot, start_token, num_tokens)`` runs covering exactly
            ``copyable_tokens``, merged across adjacent blocks sharing an owner.
            The destination offset is ``start_token`` again: a chained hash pins a
            block to one absolute prompt position, so source and destination rows
            always line up.
    """

    num_tokens: int = 0
    copyable_tokens: int = 0
    segments: tuple[tuple[int, int, int], ...] = ()


@dataclass
class _CachedBlock:
    """One resident prefix block: its chained hash, references, and live copy.

    Attributes:
        block_hash: Chained hash identifying this block *and* every block before
            it, so it names a prefix rather than a bag of tokens.
        ref_cnt: Live requests holding this block; zero does not evict.
        owner_slot: Cache slot whose rows currently hold this block's K/V, or
            ``None`` when no live copy exists. The hash alone is bookkeeping:
            reusing a block means reading real K/V out of some slot, and under the
            fixed-slot layout a slot's rows are overwritten wholesale when it is
            handed to a new request. So a block may be hittable yet unreadable,
            in which case the requester recomputes it and becomes the new owner.
    """

    block_hash: int
    ref_cnt: int = 0
    owner_slot: int | None = None


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
            hashes from another's (e.g. per-tenant) to avoid cross-hits. Under
            data parallelism every replica -- and the router's affinity index --
            must be given the same seed, or the router's guess about who holds a
            prefix never matches what the replicas actually cached.
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
        #: slot -> hashes it is the live copy of. A reverse index because
        #: :meth:`invalidate_slot` runs on every admission and would otherwise
        #: scan the whole pool, which is sized by the KV cache rather than by
        #: the prompt and so would dominate the admission path.
        self._owned: dict[int, set[int]] = {}
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
        for h in iter_block_hashes(token_ids, self.block_size, self.hash_seed):
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
        return self._reference_blocks(token_ids).num_tokens

    def admit(self, token_ids: list[int]) -> PrefixMatch:
        """One-pass :meth:`query` + :meth:`register`, for the admission path.

        The scheduler needs both answers about the same prompt in the same
        breath: how much of it is already cached, and a reference on every block
        so the prefix survives while the request runs. Asking as two calls hashes
        the prompt twice -- 256 chained block hashes for a 4 k-token prompt, all
        of it duplicate, on the critical path of every admission -- and the
        second traversal cannot see anything the first did not, because
        :meth:`register` already reports the hit length that preceded it.

        Counts towards the hit-rate statistics exactly as :meth:`query` does.

        Returns:
            The full :class:`PrefixMatch`, whose ``copyable_tokens`` (not
            ``num_tokens``) is what prefill may actually skip.
        """
        self.stats.num_requests += 1
        self.stats.queried_tokens += len(token_ids)
        match = self._reference_blocks(token_ids)
        self.stats.hit_tokens += match.num_tokens
        return match

    def invalidate_slot(self, slot: int) -> None:
        """Forget that *slot* holds any block's K/V, as it changes hands.

        A slot owns one contiguous ``max_seq_len`` region that its next occupant
        refills from that occupant's own token 0, so every block this slot was the
        live copy of becomes unreadable the moment the slot is handed over. The
        blocks stay cached and stay hittable -- their hashes are still true, and
        whoever recomputes them next becomes the new owner -- they merely stop
        being copy sources.

        Must run *after* the new occupant's match, not before: a freed slot keeps
        its rows until they are overwritten, so the commonest hit of all is the
        request that lands on the slot whose prefix it wanted. Invalidating first
        would throw that away; invalidating after leaves the new occupant's own
        blocks briefly ownerless, which costs nothing because it re-claims them
        one step later as the request that now genuinely holds them.
        """
        for block_hash in self._owned.pop(slot, ()):
            block = self._blocks.get(block_hash)
            if block is not None and block.owner_slot == slot:
                block.owner_slot = None

    def assign_owner(self, token_ids: list[int], slot: int, upto_tokens: int) -> None:
        """Record *slot* as the live copy of the ownerless blocks it now holds.

        Callers must have *executed* the prefill covering ``upto_tokens``, not
        merely scheduled it. Under a committing scheduler those are different
        moments: ``num_computed_tokens`` advances when a chunk is planned, one
        engine step before its K/V exists, and a block claimed at planning time
        would be offered as a copy source to the next admission in that same
        step -- which would read cache rows the model had not written yet.

        Blocks that already have an owner keep it: one live copy is all a copy
        needs, and leaving the incumbent alone keeps segment lists short.
        """
        blocks = upto_tokens // self.block_size
        if blocks <= 0:
            return
        for index, h in enumerate(iter_block_hashes(token_ids, self.block_size, self.hash_seed)):
            if index >= blocks:
                break
            block = self._blocks.get(h)
            if block is not None and block.owner_slot is None:
                block.owner_slot = slot
                self._owned.setdefault(slot, set()).add(h)

    def _reference_blocks(self, token_ids: list[int]) -> PrefixMatch:
        """Take a reference on every full block, creating the missing ones.

        Returns the leading reuse the caller inherits: the cached length, plus
        the shorter copyable length and the segments realising it.
        """
        hit = 0
        counting_hit = True
        copyable = 0
        segments: list[list[int]] = []
        for index, h in enumerate(iter_block_hashes(token_ids, self.block_size, self.hash_seed)):
            block = self._blocks.get(h)
            if block is None:
                counting_hit = False
                block = _CachedBlock(block_hash=h)
                self._blocks[h] = block
            elif counting_hit:
                hit += self.block_size
            block.ref_cnt += 1
            self._blocks.move_to_end(h)  # newest / just-touched -> MRU

            # The copy plan tracks the *unbroken* run of cached blocks that still
            # have a live copy: the first gap ends it, because a slot's rows are
            # only meaningful read from 0 up.
            if (
                copyable == index * self.block_size
                and counting_hit
                and block.owner_slot is not None
            ):
                start = index * self.block_size
                previous = segments[-1] if segments else None
                if previous is not None and previous[0] == block.owner_slot:
                    previous[2] += self.block_size  # extend the run in place
                else:
                    segments.append([block.owner_slot, start, self.block_size])
                copyable += self.block_size
        self._evict_to_capacity()
        return PrefixMatch(hit, copyable, tuple(tuple(run) for run in segments))

    def release(self, token_ids: list[int]) -> None:
        """Drop one reference per block; blocks stay cached (LRU) at zero.

        Unlike a naive cache, hitting ``ref_cnt == 0`` does not evict: the block
        remains resident and hittable until capacity pressure reclaims it, so a
        finished request still leaves its prefix warm for the next one.
        """
        for h in iter_block_hashes(token_ids, self.block_size, self.hash_seed):
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
            block = self._blocks[h]
            if block.ref_cnt == 0:
                if block.owner_slot is not None:
                    self._owned.get(block.owner_slot, set()).discard(h)
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
