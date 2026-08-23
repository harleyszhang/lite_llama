"""Tests for the continuous-batching admission policy.

The scheduler decides three things per step — who prefills, who decodes, and
which cache slot each running request holds — and it decides them without any
tensors, which is why these run on CPU with no checkpoint. Two invariants matter
more than the rest:

* **Slots are conserved.** A leaked slot silently shrinks the engine's capacity
  until it can admit nothing at all; a double-issued slot is worse, because two
  requests then share KV and produce plausible-looking garbage.
* **No request can starve.** FCFS plus a token budget is easy to write in a way
  that wedges the queue: a prompt too big for the budget would sit at the head
  forever and block everything behind it.
"""

from __future__ import annotations

import pytest

from lite_llama.engine.sampler import SamplingParams
from lite_llama.engine.scheduler import (
    Request,
    RequestStatus,
    Scheduler,
    SchedulerConfig,
)

_MAX_SEQ_LEN = 128


def make_request(request_id: str, prompt_len: int = 4, **params) -> Request:
    """A request whose prompt is ``prompt_len`` throwaway token ids."""
    return Request(
        request_id=request_id,
        prompt="x" * prompt_len,
        prompt_token_ids=list(range(prompt_len)),
        params=SamplingParams(**params),
    )


@pytest.fixture
def scheduler() -> Scheduler:
    """Four slots and a small token budget, so limits are reachable in one step."""
    return Scheduler(
        SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=4, max_num_batched_tokens=64),
        num_slots=4,
    )


# --------------------------------------------------------------------------- #
# Admission
# --------------------------------------------------------------------------- #
def test_generation_cap_is_resolved_against_the_context_window(scheduler):
    """``max_gen_len=None`` means "fill the window", not "unbounded"."""
    request = make_request("a", prompt_len=100)
    scheduler.add_request(request)
    assert request.max_new_tokens == _MAX_SEQ_LEN - 100


def test_generation_cap_honours_an_explicit_request(scheduler):
    request = make_request("a", prompt_len=10, max_gen_len=7)
    scheduler.add_request(request)
    assert request.max_new_tokens == 7


def test_generation_cap_is_clamped_by_the_window(scheduler):
    """A request asking for more than fits gets what fits, not a later overflow."""
    request = make_request("a", prompt_len=120, max_gen_len=1000)
    scheduler.add_request(request)
    assert request.max_new_tokens == _MAX_SEQ_LEN - 120


@pytest.mark.parametrize(
    ("prompt_len", "why"),
    [(0, "empty prompt"), (_MAX_SEQ_LEN, "no room left to generate")],
)
def test_unservable_prompts_are_refused_at_submission(scheduler, prompt_len, why):
    """Refusing here is what lets the engine assume every admitted request fits."""
    with pytest.raises(ValueError):
        scheduler.add_request(make_request("a", prompt_len=prompt_len))
    assert scheduler.num_waiting == 0, why


# --------------------------------------------------------------------------- #
# Prefill scheduling
# --------------------------------------------------------------------------- #
def test_prefill_takes_priority_over_decode(scheduler):
    """A queued arrival gets prefilled alongside running decode (chunked prefill)."""
    scheduler.add_request(make_request("a"))
    scheduler.schedule()  # admits and prefills `a`

    scheduler.add_request(make_request("b"))
    output = scheduler.schedule()

    assert [r.request_id for r in output.prefill] == ["b"]
    # With chunked prefill, decode runs alongside prefill (not exclusive)
    assert any(r.request_id == "a" for r in output.decode)


def test_prefill_admits_in_arrival_order(scheduler):
    for name in "abc":
        scheduler.add_request(make_request(name))
    assert [r.request_id for r in scheduler.schedule().prefill] == ["a", "b", "c"]


def test_concurrency_is_capped_and_the_rest_stay_queued(scheduler):
    for index in range(6):
        scheduler.add_request(make_request(f"r{index}"))

    admitted = scheduler.schedule().prefill

    assert len(admitted) == 4  # max_num_seqs
    assert scheduler.num_waiting == 2
    assert all(r.status is RequestStatus.WAITING for r in scheduler.waiting)


def test_token_budget_bounds_the_padded_prefill_grid(scheduler):
    """Prefill pads to the longest prompt, so the budget is measured padded.

    Two 40-token prompts cost 80 padded token-slots against a 64 budget, so the
    second waits even though three slots are still free.
    """
    scheduler.add_request(make_request("long-a", prompt_len=40))
    scheduler.add_request(make_request("long-b", prompt_len=40))

    assert [r.request_id for r in scheduler.schedule().prefill] == ["long-a"]
    assert scheduler.num_waiting == 1


def test_a_prompt_bigger_than_the_budget_still_runs(scheduler):
    """Otherwise it wedges the FCFS queue permanently."""
    scheduler.add_request(make_request("huge", prompt_len=100))
    scheduler.add_request(make_request("small", prompt_len=4))

    assert [r.request_id for r in scheduler.schedule().prefill] == ["huge"]
    assert scheduler.num_waiting == 1


def test_max_num_seqs_cannot_exceed_the_available_slots():
    """Concurrency is bounded by cache slots, whatever the caller asked for."""
    sched = Scheduler(SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=64), num_slots=3)
    assert sched.max_num_seqs == 3


def test_zero_slots_is_rejected():
    with pytest.raises(ValueError):
        Scheduler(SchedulerConfig(max_seq_len=_MAX_SEQ_LEN), num_slots=0)


# --------------------------------------------------------------------------- #
# Decode scheduling
# --------------------------------------------------------------------------- #
def test_decode_covers_every_running_request(scheduler):
    for name in "abc":
        scheduler.add_request(make_request(name))
    scheduler.schedule()  # prefill all three

    output = scheduler.schedule()

    assert output.prefill == []
    assert [r.request_id for r in output.decode] == ["a", "b", "c"]


def test_an_idle_scheduler_schedules_nothing(scheduler):
    assert scheduler.schedule().is_empty
    assert not scheduler.has_unfinished_requests()


# --------------------------------------------------------------------------- #
# Slot accounting
# --------------------------------------------------------------------------- #
def test_running_requests_hold_distinct_slots(scheduler):
    for index in range(4):
        scheduler.add_request(make_request(f"r{index}"))
    admitted = scheduler.schedule().prefill

    slots = [r.slot for r in admitted]
    assert len(set(slots)) == 4, "two requests sharing a slot would share KV"
    assert all(0 <= slot < 4 for slot in slots)
    assert scheduler.num_free_slots == 0


def test_finishing_returns_the_slot_and_retires_the_request(scheduler):
    request = make_request("a")
    scheduler.add_request(request)
    scheduler.schedule()
    slot = request.slot

    scheduler.finish(request, "eos")

    assert request.status is RequestStatus.FINISHED
    assert request.finish_reason == "eos"
    assert request.slot is None
    assert request.finish_time is not None
    assert scheduler.num_running == 0
    assert scheduler.num_free_slots == 4, f"slot {slot} leaked"


def test_finishing_twice_does_not_double_free_the_slot(scheduler):
    """A double free would hand one slot to two later requests."""
    request = make_request("a")
    scheduler.add_request(request)
    scheduler.schedule()

    scheduler.finish(request, "eos")
    scheduler.finish(request, "length")

    assert scheduler.num_free_slots == 4
    assert request.finish_reason == "eos", "the first reason is the true one"


def test_a_freed_slot_is_reused_by_a_queued_request(scheduler):
    for index in range(5):
        scheduler.add_request(make_request(f"r{index}"))
    admitted = scheduler.schedule().prefill
    freed = admitted[1].slot

    scheduler.finish(admitted[1], "eos")
    next_group = scheduler.schedule().prefill

    assert [r.request_id for r in next_group] == ["r4"]
    assert next_group[0].slot == freed


def test_many_finish_and_admit_rounds_conserve_slots(scheduler):
    """The leak this catches only surfaces as a stuck engine much later."""
    for round_index in range(20):
        request = make_request(f"r{round_index}")
        scheduler.add_request(request)
        scheduler.schedule()
        scheduler.finish(request, "eos")

    assert scheduler.num_free_slots == 4
    assert scheduler.num_running == 0
    assert not scheduler.has_unfinished_requests()


def test_running_requests_are_never_preempted(scheduler):
    """Slot capacity equals the context window, so nothing can be evicted.

    Admitting six requests into four slots must leave the four that are running
    untouched rather than recycling one of their slots.
    """
    for index in range(6):
        scheduler.add_request(make_request(f"r{index}"))
    admitted = scheduler.schedule().prefill
    held = {r.request_id: r.slot for r in admitted}

    for _ in range(5):
        scheduler.schedule()

    assert {r.request_id: r.slot for r in scheduler.running} == held


# --------------------------------------------------------------------------- #
# Abort
# --------------------------------------------------------------------------- #
def test_aborting_a_running_request_frees_its_slot(scheduler):
    request = make_request("a")
    scheduler.add_request(request)
    scheduler.schedule()

    assert scheduler.abort("a") is request
    assert request.finish_reason == "abort"
    assert scheduler.num_free_slots == 4


def test_aborting_a_queued_request_dequeues_it(scheduler):
    scheduler.add_request(make_request("a"))
    scheduler.add_request(make_request("b"))

    aborted = scheduler.abort("b")

    assert aborted is not None and aborted.finish_reason == "abort"
    assert scheduler.num_waiting == 1
    assert [r.request_id for r in scheduler.schedule().prefill] == ["a"]


def test_aborting_an_unknown_id_is_a_no_op(scheduler):
    assert scheduler.abort("nope") is None


# --------------------------------------------------------------------------- #
# Request bookkeeping
# --------------------------------------------------------------------------- #
def test_seq_len_counts_prompt_plus_generated():
    request = make_request("a", prompt_len=5)
    request.output_token_ids.extend([1, 2, 3])
    assert request.seq_len == 8


def test_has_room_tracks_the_generation_cap(scheduler):
    request = make_request("a", max_gen_len=2)
    scheduler.add_request(request)

    assert request.has_room
    request.output_token_ids.append(1)
    assert request.has_room
    request.output_token_ids.append(2)
    assert not request.has_room
