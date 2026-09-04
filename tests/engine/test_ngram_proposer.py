"""Tests for the ngram speculative decoding proposer (O5).

Verifies that the proposer finds repeated n-gram patterns and proposes
the correct continuation tokens.

Usage:
    pytest tests/engine/test_ngram_proposer.py
"""

from __future__ import annotations

import pytest

from rapid_llm.engine.ngram_proposer import NgramProposer


class TestNgramProposer:
    def test_empty_sequence(self):
        p = NgramProposer()
        assert p.propose([]) == []

    def test_too_short(self):
        p = NgramProposer()
        assert p.propose([1]) == []
        assert p.propose([1, 2]) == []

    def test_no_match(self):
        p = NgramProposer()
        # All unique tokens, no repeated n-gram.
        assert p.propose([1, 2, 3, 4, 5]) == []

    def test_bigram_match(self):
        p = NgramProposer(max_ngram_size=5, max_draft=6)
        # Pattern: [1, 2] appears at positions 0-1 and 5-6.
        # After the second occurrence at pos 5, continuation is [3, 4, 0, 1, 2]
        # (everything after pos 7, capped at max_draft=6).
        seq = [1, 2, 3, 4, 0, 1, 2]
        result = p.propose(seq)
        assert result == [3, 4, 0, 1, 2]

    def test_trigram_match(self):
        p = NgramProposer(max_ngram_size=5, max_draft=6)
        # Pattern: [1, 2, 3] at pos 0-2 and 6-8.
        # Match at pos 0, continuation after pos 2: [4, 5, 0, 1, 2, 3] (6 = max_draft).
        seq = [1, 2, 3, 4, 5, 0, 1, 2, 3]
        result = p.propose(seq)
        assert result == [4, 5, 0, 1, 2, 3]

    def test_longer_match_preferred(self):
        p = NgramProposer(max_ngram_size=5, max_draft=6)
        # Trigram [2, 3, 4] matches at pos 1-3, continuation is [5, 0, 2, 3, 4].
        seq = [1, 2, 3, 4, 5, 0, 2, 3, 4]
        result = p.propose(seq)
        assert result == [5, 0, 2, 3, 4]

    def test_max_draft_cap(self):
        p = NgramProposer(max_ngram_size=5, max_draft=2)
        # Pattern matches, but 5 tokens follow. Cap at max_draft=2.
        seq = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2]
        result = p.propose(seq)
        assert len(result) <= 2
        assert result == [3, 4]

    def test_no_match_at_end(self):
        p = NgramProposer(max_ngram_size=5, max_draft=6)
        # Trigram [1, 2, 3] matches at pos 0-2, continuation is [0, 1, 2, 3].
        seq = [1, 2, 3, 0, 1, 2, 3]
        result = p.propose(seq)
        assert result == [0, 1, 2, 3]

    def test_repetitive_pattern(self):
        p = NgramProposer(max_ngram_size=5, max_draft=6)
        # Highly repetitive: [1, 2, 1, 2, 1, 2, ...].
        seq = [1, 2, 1, 2, 1, 2, 1, 2]
        result = p.propose(seq)
        # Last 5 tokens: [2, 1, 2, 1, 2]. Looking for earlier match.
        # The proposer should find a match and propose continuation.
        assert len(result) > 0
