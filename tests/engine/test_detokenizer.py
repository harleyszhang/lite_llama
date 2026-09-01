"""Tests for :class:`IncrementalDetokenizer`.

Fake SentencePiece-like and byte-level tokenizers drive the edge cases:
leading spaces, multibyte characters held back until complete, and the
invariant stream == full decode.

Usage:
    pytest tests/engine/test_detokenizer.py
"""

from __future__ import annotations

import pytest

from lite_llama.engine.detokenizer import IncrementalDetokenizer

_SPACE = "\u2581"  # SentencePiece word-start marker


class SentencePieceLike:
    """Fake SentencePiece tokenizer: joins pieces, then strips the leading space.

    That final strip is the behaviour that breaks per-token decoding, and it is
    real: ``LlamaTokenizer.decode(["_A"])`` returns ``"A"``, not ``" A"``.
    """

    def __init__(self, pieces: dict[int, str]) -> None:
        self.pieces = pieces

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        text = "".join(self.pieces[i] for i in token_ids).replace(_SPACE, " ")
        return text[1:] if text.startswith(" ") else text


class ByteLevelLike:
    """Fake byte-level BPE: concatenates raw bytes and decodes with replacement.

    A token boundary inside a UTF-8 sequence produces U+FFFD, exactly as
    ``tokenizer.decode`` does on a truncated byte run.
    """

    def __init__(self, pieces: dict[int, bytes]) -> None:
        self.pieces = pieces

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return b"".join(self.pieces[i] for i in token_ids).decode("utf-8", errors="replace")


@pytest.fixture
def sp_tokenizer() -> SentencePieceLike:
    return SentencePieceLike(
        {
            1: f"{_SPACE}A",
            2: f"{_SPACE}large",
            3: f"{_SPACE}black",
            4: f"{_SPACE}dog",
            5: ".",
        }
    )


def test_words_keep_their_separating_spaces(sp_tokenizer):
    """The regression this class was written for: no word gluing.

    Decoding each token alone would yield "Alargeblackdog"; the windowed decode
    must produce "A large black dog".
    """
    det = IncrementalDetokenizer(sp_tokenizer, batch_size=1)
    streamed = "".join(det.append(0, tid) for tid in (1, 2, 3, 4))
    assert streamed == "A large black dog"


def test_stream_equals_full_decode(sp_tokenizer):
    """Concatenated deltas must equal one decode of every token.

    This is the contract the blocking API and the streaming API share, so any
    dropped or duplicated fragment shows up as a mismatch.
    """
    det = IncrementalDetokenizer(sp_tokenizer, batch_size=1)
    ids = [1, 2, 3, 4, 5]
    streamed = "".join(det.append(0, tid) for tid in ids)
    assert streamed == det.text(0)
    assert streamed == sp_tokenizer.decode(ids)


def test_first_token_has_no_leading_space(sp_tokenizer):
    """A completion must not begin with a space the tokenizer would have stripped."""
    det = IncrementalDetokenizer(sp_tokenizer, batch_size=1)
    assert det.append(0, 1) == "A"


def test_multibyte_character_is_held_back_until_complete():
    """A character split across two tokens emits nothing, then the whole char.

    Emitting the first half would push U+FFFD to the user. The character here is
    U+4F60 ("ni"), whose UTF-8 encoding is three bytes split 2 + 1.
    """
    tokenizer = ByteLevelLike({1: b"\xe4\xbd", 2: b"\xa0", 3: b"!"})
    det = IncrementalDetokenizer(tokenizer, batch_size=1)

    assert det.append(0, 1) == ""  # incomplete: held back
    assert det.append(0, 2) == "\u4f60"  # completed
    assert det.append(0, 3) == "!"


def test_multibyte_stream_equals_full_decode():
    """Even with held-back fragments the totals must agree."""
    tokenizer = ByteLevelLike({1: b"\xe4\xbd", 2: b"\xa0", 3: b"\xe5\xa5", 4: b"\xbd"})
    det = IncrementalDetokenizer(tokenizer, batch_size=1)
    ids = [1, 2, 3, 4]
    streamed = "".join(det.append(0, tid) for tid in ids)
    assert streamed == "\u4f60\u597d"
    assert streamed == det.text(0)


def test_no_replacement_character_ever_escapes():
    """Nothing emitted may contain U+FFFD, however the bytes are chopped."""
    tokenizer = ByteLevelLike({1: b"\xf0\x9f", 2: b"\x98", 3: b"\x80"})  # U+1F600, 4 bytes
    det = IncrementalDetokenizer(tokenizer, batch_size=1)
    emitted = [det.append(0, tid) for tid in (1, 2, 3)]
    assert all("\ufffd" not in piece for piece in emitted)
    assert "".join(emitted) == "\U0001f600"


def test_sequences_are_independent(sp_tokenizer):
    """Per-sequence offsets must not share state across the batch.

    Interleaving two sequences at different lengths is what a single shared
    ``prefix_offset`` would corrupt.
    """
    det = IncrementalDetokenizer(sp_tokenizer, batch_size=2)

    assert det.append(0, 1) == "A"
    assert det.append(1, 3) == "black"
    assert det.append(0, 2) == " large"
    assert det.append(1, 4) == " dog"

    assert det.text(0) == "A large"
    assert det.text(1) == "black dog"
    assert det.tokens(0) == [1, 2]
    assert det.tokens(1) == [3, 4]


def test_tokens_accumulate_in_order(sp_tokenizer):
    det = IncrementalDetokenizer(sp_tokenizer, batch_size=1)
    for tid in (4, 2, 1):
        det.append(0, tid)
    assert det.tokens(0) == [4, 2, 1]


def test_text_of_untouched_sequence_is_empty(sp_tokenizer):
    """An empty token list must decode to "" and not call the tokenizer."""
    det = IncrementalDetokenizer(sp_tokenizer, batch_size=2)
    det.append(0, 1)
    assert det.text(1) == ""
    assert det.tokens(1) == []
