"""Tests for :class:`StopCriteria`.

The class exists to track stop state *without* per-step device-to-host syncs:
membership is a boolean lookup table indexed by the sampled token, and the
finished/reason state stays on device. Reading it back costs a sync, so
``all_finished`` and ``reasons`` are the only methods allowed to do that.

Everything here runs on CPU tensors -- the logic is device-agnostic, and testing
it on CPU keeps it in the fast tier where it will actually be run.

The subtle contract is ``writable``: during batched prefill some positions belong
to the original prompt, and the token sampled there is discarded rather than
appended. Those positions must not be able to finish a sequence, or a prompt that
happens to contain an EOS id would truncate generation before it started.
"""

from __future__ import annotations

import torch

from lite_llama.engine.stop_criteria import StopCriteria

_VOCAB = 32
_EOS = 7
_EOS2 = 9


def _criteria(batch_size: int, stop_ids=frozenset({_EOS})) -> StopCriteria:
    return StopCriteria(batch_size, set(stop_ids), _VOCAB, device="cpu")


def _all_writable(batch_size: int) -> torch.Tensor:
    return torch.ones(batch_size, dtype=torch.bool)


def test_starts_unfinished():
    sc = _criteria(3)
    assert not sc.finished.any()
    assert not sc.all_finished()
    assert sc.reasons() == ["length"] * 3


def test_stop_token_finishes_only_its_own_sequence():
    sc = _criteria(3)
    sc.update(torch.tensor([_EOS, 1, 2]), _all_writable(3))
    assert sc.finished.tolist() == [True, False, False]
    assert sc.reasons() == ["eos", "length", "length"]


def test_non_stop_token_changes_nothing():
    sc = _criteria(2)
    sc.update(torch.tensor([1, 2]), _all_writable(2))
    assert not sc.finished.any()


def test_multiple_stop_ids_are_all_recognised():
    """Both the tokenizer EOS and generation_config extras must terminate."""
    sc = _criteria(2, stop_ids={_EOS, _EOS2})
    sc.update(torch.tensor([_EOS2, _EOS]), _all_writable(2))
    assert sc.finished.tolist() == [True, True]
    assert sc.all_finished()


def test_empty_stop_set_never_finishes():
    """With no stop ids the only exit is the length cap."""
    sc = _criteria(2, stop_ids=frozenset())
    for tid in range(_VOCAB):
        sc.update(torch.full((2,), tid), _all_writable(2))
    assert not sc.finished.any()


def test_non_writable_position_cannot_finish():
    """A stop id sampled at a prompt position must be ignored.

    During batched prefill the shorter sequences still produce a sampled token at
    positions that belong to the prompt; those are discarded. Letting them finish
    would truncate a sequence whose prompt merely contains the EOS id.
    """
    sc = _criteria(2)
    writable = torch.tensor([False, True])
    sc.update(torch.tensor([_EOS, _EOS]), writable)
    assert sc.finished.tolist() == [False, True]


def test_first_reason_wins():
    """Once finished, a later stop token must not rewrite the reason.

    The reason is reported to the caller as the *cause* of termination, so it has
    to reflect the first event, not the last token the loop happened to sample
    before noticing.
    """
    sc = _criteria(1)
    sc.mark_repeat(0)
    sc.update(torch.tensor([_EOS]), _all_writable(1))
    assert sc.reasons() == ["repeat"]


def test_mark_repeat_sets_reason_and_finishes():
    sc = _criteria(2)
    sc.mark_repeat(1)
    assert sc.finished.tolist() == [False, True]
    assert sc.reasons() == ["length", "repeat"]


def test_all_finished_requires_every_sequence():
    sc = _criteria(2)
    sc.update(torch.tensor([_EOS, 1]), _all_writable(2))
    assert not sc.all_finished()
    sc.update(torch.tensor([1, _EOS]), _all_writable(2))
    assert sc.all_finished()


def test_finished_is_monotonic():
    """A finished sequence never reverts, whatever it samples afterwards.

    The decode loop keeps stepping every sequence until the whole batch stops, so
    tokens keep arriving for already-finished rows and must not reopen them.
    """
    sc = _criteria(1)
    sc.update(torch.tensor([_EOS]), _all_writable(1))
    for tid in (1, 2, 3):
        sc.update(torch.tensor([tid]), _all_writable(1))
        assert sc.finished.item()
