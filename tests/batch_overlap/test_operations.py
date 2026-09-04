"""The batch-overlap executor primitives: stages, state hygiene (CPU)."""

from __future__ import annotations

import pytest

from rapid_llm.batch_overlap.operations import (
    StateDict,
    YieldOperation,
    execute_operations,
    execute_overlapped_operations,
)


def _op(log: list, name: str):
    """An op that only records its name — the schedule is what's under test."""

    def run(state) -> None:
        log.append(name)

    return run


def _grow(value: int):
    """An op that appends to the state's own trace list."""

    def run(state) -> None:
        state.trace.append(value)

    return run


def _stream(values: list[int]) -> list:
    """A three-stage stream, one op per stage."""
    return [_grow(v) for v in values[:1]] + [
        item for v in values[1:] for item in (YieldOperation(), _grow(v))
    ]


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def test_streams_strictly_alternate_at_zero_delta():
    log: list = []
    execute_overlapped_operations(
        [StateDict(), StateDict()],
        [
            [_op(log, "a1"), YieldOperation(), _op(log, "a2")],
            [_op(log, "b1"), YieldOperation(), _op(log, "b2")],
        ],
        delta_stages=[0, 0],
    )
    assert log == ["a1", "b1", "a2", "b2"]


def test_delta_stages_give_the_lead_stream_a_head_start():
    """A stays exactly delta_stages ahead: its head runs early, its tail late."""
    log: list = []
    execute_overlapped_operations(
        [StateDict(), StateDict()],
        [
            [_op(log, "a1"), YieldOperation(), _op(log, "a2"), YieldOperation(), _op(log, "a3")],
            [_op(log, "b1"), YieldOperation(), _op(log, "b2"), YieldOperation(), _op(log, "b3")],
        ],
        delta_stages=[0, 2],
    )
    # A's first two stages run alone, then every A stage pairs with the B stage
    # two behind it, then B drains its last two.
    assert log == ["a1", "a2", "a3", "b1", "b2", "b3"]


def test_ops_within_one_stage_never_interleave():
    """A yield is the only switch point — ops inside a stage stay together."""
    log: list = []
    execute_overlapped_operations(
        [StateDict(), StateDict()],
        [[_op(log, "a1"), _op(log, "a2"), _op(log, "a3")], [_op(log, "b1")]],
        delta_stages=[0, 0],
    )
    assert log == ["a1", "a2", "a3", "b1"]


def test_interleaving_matches_serial_results():
    """The schedule changes, the outcomes must not: each stream's ops only
    touch their own state, so interleaved and serial runs agree."""
    ops_a, ops_b = _stream([1, 2, 3]), _stream([10, 20, 30])
    interleaved = execute_overlapped_operations(
        [StateDict({"trace": []}), StateDict({"trace": []})],
        [ops_a, ops_b],
        delta_stages=[0, 1],
    )
    serial_a = execute_operations(StateDict({"trace": []}), ops_a)
    serial_b = execute_operations(StateDict({"trace": []}), ops_b)
    assert [state.trace for state in interleaved] == [serial_a.trace, serial_b.trace]
    assert [state.trace for state in interleaved] == [[1, 2, 3], [10, 20, 30]]


def test_mismatched_stage_counts_are_rejected():
    """delta_stages only means something when both streams cut the same way."""
    with pytest.raises(AssertionError):
        execute_overlapped_operations(
            [StateDict(), StateDict()],
            [[_op([], "a1"), YieldOperation(), _op([], "a2")], [_op([], "b1")]],
            delta_stages=[0, 1],
        )


def test_the_lead_delta_must_be_zero():
    """One lead parameter, carried by the trailing stream (sglang's convention)."""
    with pytest.raises(ValueError):
        execute_overlapped_operations([StateDict(), StateDict()], [[], []], delta_stages=[1, 0])


def test_negative_delta_is_rejected():
    with pytest.raises(ValueError):
        execute_overlapped_operations([StateDict(), StateDict()], [[], []], delta_stages=[0, -1])


def test_a_leading_or_trailing_yield_is_rejected():
    """An empty stage would make the stage count — and delta — ambiguous."""
    with pytest.raises(ValueError):
        execute_overlapped_operations(
            [StateDict(), StateDict()],
            [[YieldOperation(), _op([], "a1")], [_op([], "b1")]],
            delta_stages=[0, 0],
        )


# --------------------------------------------------------------------------- #
# StateDict hygiene
# --------------------------------------------------------------------------- #
def test_state_writes_once_until_a_key_is_popped():
    state = StateDict({"hidden_states": 1})
    with pytest.raises(AssertionError, match="already exists"):
        state.hidden_states = 2
    assert state.pop("hidden_states") == 1
    state.hidden_states = 2  # the name is free again
    assert state.hidden_states == 2


def test_state_read_of_a_missing_key_names_the_keys_it_has():
    state = StateDict({"residual": None})
    with pytest.raises(AttributeError, match="residual"):
        _ = state.hidden_states


def test_state_clear_proves_every_intermediate_was_released():
    state = StateDict({"hidden_states": 1, "residual": 2})
    with pytest.raises(AssertionError, match="unexpected keys"):
        state.clear(["hidden_states"])
    state.clear(["hidden_states", "residual"])
    with pytest.raises(AttributeError):
        _ = state.hidden_states
