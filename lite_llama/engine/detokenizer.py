"""Incremental detokenisation: per-step token ids to per-step text deltas.

Decoding the whole span each step and diffing is ``O(n^2)``. Instead, per sequence
two offsets (``prefix_offset`` for context, ``read_offset`` for what was already
emitted) bound each step to decoding a small window — ``O(1)`` amortised. A window
rather than a lone token is required for correctness: SentencePiece strips a
leading ``▁`` from a single token (gluing words together), and multi-byte UTF-8
split across tokens must be held back until the character completes.

Usage:
    det = IncrementalDetokenizer(tokenizer, batch_size)
    delta = det.append(seq_index, token_id)   # "" until a character completes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# Emitted by tokenizers for a byte sequence that is not yet a complete
# character; text ending in it must be held back until the next token arrives.
_REPLACEMENT_CHAR = "\ufffd"


class Tokenizer(Protocol):
    """The one tokenizer method incremental detokenisation depends on."""

    def decode(self, token_ids: list[int], skip_special_tokens: bool = ...) -> str: ...


@dataclass
class _SequenceState:
    """Detokenisation state for a single sequence: token history and two offsets.

    ``prefix_offset`` starts the window decoded for context; ``read_offset`` is
    the boundary already emitted as text. ``prefix_text`` caches
    ``decode(tokens[prefix_offset:read_offset])`` so a step that holds text back —
    leaving both offsets unchanged — does not decode that same window again.
    """

    tokens: list[int] = field(default_factory=list)
    prefix_offset: int = 0
    read_offset: int = 0
    prefix_text: str | None = None


class IncrementalDetokenizer:
    """Turns per-step token ids into per-step text deltas, one state per sequence.

    Args:
        tokenizer: Any HuggingFace tokenizer exposing ``decode``.
        batch_size: Number of independent sequences to track.
    """

    def __init__(self, tokenizer: Tokenizer, batch_size: int) -> None:
        self._tokenizer = tokenizer
        self._states = [_SequenceState() for _ in range(batch_size)]

    def tokens(self, index: int) -> list[int]:
        """Token ids appended to sequence ``index`` so far."""
        return self._states[index].tokens

    def append(self, index: int, token_id: int) -> str:
        """Append one token to a sequence and return the newly readable text.

        Returns an empty string when the token does not complete a character
        yet; the held-back bytes are emitted with a later token.
        """
        state = self._states[index]
        state.tokens.append(token_id)

        prefix_text = state.prefix_text
        if prefix_text is None:
            prefix_text = self._decode(state.tokens[state.prefix_offset : state.read_offset])
        full_text = self._decode(state.tokens[state.prefix_offset :])

        if len(full_text) <= len(prefix_text) or full_text.endswith(_REPLACEMENT_CHAR):
            # Nothing new is safely readable yet: keep the window open so the next
            # token is decoded with the same context, and reuse this prefix decode.
            state.prefix_text = prefix_text
            return ""

        # Slide the window forward: what we just emitted becomes the context.
        state.prefix_offset = state.read_offset
        state.read_offset = len(state.tokens)
        state.prefix_text = None
        return full_text[len(prefix_text) :]

    def text(self, index: int) -> str:
        """Full text of a sequence, decoded in one pass.

        Used by callers that need the whole generated string (the repetition
        breaker, and the blocking API when it skips per-step streaming).
        """
        return self._decode(self._states[index].tokens)

    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)
