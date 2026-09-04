"""Ngram proposer for speculative decoding (O5).

Scans the token history (prompt + generated so far) for n-gram matches and
proposes draft tokens that continue the matched pattern. The proposer is
called before each decode step; its drafts are verified by the model in a
single forward pass, accepting correct predictions and rejecting the rest.

Usage:
    proposer = NgramProposer(max_ngram_size=5, max_draft=6)
    drafts = proposer.propose(token_ids)  # prompt + generated so far
"""

from __future__ import annotations


class NgramProposer:
    """Propose draft tokens by matching n-grams in the token history.

    The proposer scans the token sequence from the longest n-gram down to
    bigrams, looking for the most recent occurrence of the trailing n tokens.
    When a match is found, it returns the tokens that followed the match
    (up to ``max_draft``). If no match is found for any n-gram size, it
    returns an empty list (no drafts).

    Args:
        max_ngram_size: Largest n-gram to try (default 5).
        max_draft: Maximum number of draft tokens to return (default 6).
    """

    def __init__(self, max_ngram_size: int = 5, max_draft: int = 6) -> None:
        self.max_ngram_size = max_ngram_size
        self.max_draft = max_draft

    def propose(self, token_ids: list[int]) -> list[int]:
        """Return up to ``max_draft`` draft tokens from n-gram matching.

        Args:
            token_ids: The full token sequence so far (prompt + generated).

        Returns:
            A list of draft token ids (possibly empty if no match found).
        """
        n = len(token_ids)
        if n < 2:
            return []

        # Try from longest n-gram down to bigram.
        for gram_size in range(min(self.max_ngram_size, n - 1), 1, -1):
            # The trailing `gram_size` tokens are the pattern to match.
            pattern = token_ids[n - gram_size : n]

            # Search backwards for the most recent earlier occurrence.
            # Start from n - gram_size - 1 and scan left.
            for start in range(n - gram_size - 1, -1, -1):
                if token_ids[start : start + gram_size] == pattern:
                    # Found a match at `start`. Return the tokens that
                    # followed it (up to max_draft).
                    match_end = start + gram_size
                    remaining = token_ids[match_end : match_end + self.max_draft]
                    if remaining:
                        return remaining
                    # If no tokens followed, keep searching for an earlier
                    # match that has followers.
                    break  # This match had no followers; try next gram_size.

        return []
