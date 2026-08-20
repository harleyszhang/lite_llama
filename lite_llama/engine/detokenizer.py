"""Incremental detokenisation for streaming output.

The naive way to stream text is to decode the whole generated span every step
and diff it against what was already emitted. That is ``O(n)`` tokenizer work
per step, so producing ``n`` tokens costs ``O(n^2)`` — at 256 tokens it measured
0.8 ms per step, comparable to the entire GPU decode step.

:class:`IncrementalDetokenizer` keeps the cost bounded instead. Per sequence it
tracks two offsets into the token list:

* ``prefix_offset`` — start of the window that is decoded for *context*;
* ``read_offset``  — everything before this has already been emitted as text.

Each step decodes only ``tokens[prefix_offset:]`` and the shorter
``tokens[prefix_offset:read_offset]``, then emits the difference. Both windows
stay small, which is what makes the step cost constant.

Decoding a window rather than a single token is not an optimisation detail, it
is required for correctness: SentencePiece tokenizers (LLaMA, LLaVA, Vicuna)
represent a leading space as ``▁`` and ``decode()`` strips it when the input is
a lone token, so per-token decoding silently glues words together
("A large black dog" -> "Alargeblackdog"). Multi-byte UTF-8 characters split
across tokens are handled by holding text back until the replacement character
disappears.
"""

from __future__ import annotations

from typing import Any

# Emitted by tokenizers for a byte sequence that is not yet a complete
# character; text ending in it must be held back until the next token arrives.
_REPLACEMENT_CHAR = "\ufffd"


class IncrementalDetokenizer:
    """Turns per-step token ids into per-step text deltas, one state per sequence.

    Args:
        tokenizer: Any HuggingFace tokenizer exposing ``decode``.
        batch_size: Number of independent sequences to track.
    """

    def __init__(self, tokenizer: Any, batch_size: int) -> None:
        self._tokenizer = tokenizer
        self._tokens: list[list[int]] = [[] for _ in range(batch_size)]
        self._prefix_offset = [0] * batch_size
        self._read_offset = [0] * batch_size

    def tokens(self, index: int) -> list[int]:
        """Token ids appended to sequence ``index`` so far."""
        return self._tokens[index]

    def append(self, index: int, token_id: int) -> str:
        """Append one token to a sequence and return the newly readable text.

        Returns an empty string when the token does not complete a character
        yet; the held-back bytes are emitted with a later token.
        """
        tokens = self._tokens[index]
        tokens.append(token_id)
        return self._flush(index)

    def _flush(self, index: int) -> str:
        tokens = self._tokens[index]
        prefix_offset = self._prefix_offset[index]
        read_offset = self._read_offset[index]

        prefix_text = self._decode(tokens[prefix_offset:read_offset])
        full_text = self._decode(tokens[prefix_offset:])

        if len(full_text) <= len(prefix_text) or full_text.endswith(_REPLACEMENT_CHAR):
            # Nothing new is safely readable yet: keep the window open so the
            # next token is decoded with the same context.
            return ""

        # Slide the window forward: what we just emitted becomes the context.
        self._prefix_offset[index] = read_offset
        self._read_offset[index] = len(tokens)
        return full_text[len(prefix_text) :]

    def text(self, index: int) -> str:
        """Full text of a sequence, decoded in one pass.

        Used by callers that need the whole generated string (the repetition
        breaker, and the blocking API when it skips per-step streaming).
        """
        return self._decode(self._tokens[index])

    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)
