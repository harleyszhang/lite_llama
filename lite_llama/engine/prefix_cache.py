"""Prefix caching: reuse KV of shared prompt prefixes across requests.

Aligned with vLLM's block-hash prefix caching. A prompt is split into fixed-size
blocks; each block's hash chains the previous block's hash, so a hash identifies
a *prefix* (this block plus everything before it), not just the block's own
tokens. When a new request's leading blocks are already cached, those tokens need
not be prefilled again.

The classic win is a shared system prompt: the first request pays full prefill,
every later request with the same prefix skips it.

Usage:
    cache = PrefixCache(block_size=16)
    cache.register(req_a.prompt_token_ids)          # first request populates
    hit = cache.query(req_b.prompt_token_ids)       # -> cached leading tokens
"""

from __future__ import annotations


def _hash_blocks(token_ids: list[int], block_size: int) -> list[int]:
    """Return one chained hash per full block of *token_ids*.

    Each block's hash folds in the previous block's hash, so identical hashes
    imply identical prefixes (not merely identical block contents at that
    offset). A trailing partial block is ignored: a half-filled block's KV is
    not reusable until it is complete.
    """
    hashes: list[int] = []
    parent = 0
    num_full = len(token_ids) // block_size
    for b in range(num_full):
        block = tuple(token_ids[b * block_size : (b + 1) * block_size])
        parent = hash((parent, block))
        hashes.append(parent)
    return hashes


class PrefixCache:
    """Ref-counted block-hash store mapping cached prefixes to hit lengths.

    Args:
        block_size: Tokens per block. Larger blocks hash cheaper but match
            coarser; 16 mirrors vLLM's default page granularity for KV reuse.
    """

    def __init__(self, block_size: int = 16) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.block_size = block_size
        #: block-hash -> how many live requests hold that cached block.
        self._blocks: dict[int, int] = {}
        # Metrics
        self.total_queries: int = 0
        self.total_prompt_tokens: int = 0
        self.total_cached_tokens: int = 0

    def query(self, token_ids: list[int]) -> int:
        """Return the number of leading tokens already cached for this prompt.

        Walks the prompt's block hashes from the front and stops at the first
        block that is not cached: prefix reuse must be contiguous from token 0,
        exactly as a KV cache read is.
        """
        self.total_queries += 1
        self.total_prompt_tokens += len(token_ids)
        cached = 0
        for h in _hash_blocks(token_ids, self.block_size):
            if self._blocks.get(h, 0) > 0:
                cached += self.block_size
            else:
                break
        self.total_cached_tokens += cached
        return cached

    def register(self, token_ids: list[int]) -> None:
        """Add every full block of *token_ids* to the cache (ref-count +1)."""
        for h in _hash_blocks(token_ids, self.block_size):
            self._blocks[h] = self._blocks.get(h, 0) + 1

    def release(self, token_ids: list[int]) -> None:
        """Drop one reference to each block; evict blocks that reach zero."""
        for h in _hash_blocks(token_ids, self.block_size):
            count = self._blocks.get(h, 0)
            if count <= 1:
                self._blocks.pop(h, None)
            else:
                self._blocks[h] = count - 1

    @property
    def hit_rate(self) -> float:
        """Fraction of queried prompt tokens served from cache (0.0 - 1.0)."""
        if self.total_prompt_tokens == 0:
            return 0.0
        return self.total_cached_tokens / self.total_prompt_tokens

    @property
    def num_cached_blocks(self) -> int:
        return len(self._blocks)
