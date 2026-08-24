"""Tests for the continuous-batching admission policy.

Structure follows vLLM's ``v1/core/test_scheduler.py`` (admission order,
budget gating, preemption lifecycle, prefix-cache interaction, SchedulerOutput
field integrity) and SGLang's scheduler tests (chunked prefill interleaving,
fairness under oversubscription), adapted to lite_llama's API.

Ten test classes, each owning one concern:

    TestRequestAdmission      — generation cap, unservable refusal, slot ceiling
    TestPrefillScheduling     — FCFS order, token budget, padded grid gating
    TestChunkedPrefill        — chunk splitting, advance_chunks, decode interleave
    TestDecodeScheduling      — full decode coverage, idle no-op
    TestSlotAccounting        — distinct slots, finish/free, leak detection
    TestPreemption            — opt-in oversubscription, progress quantum, fairness
    TestSchedulerOutput       — chunk_lens, preempted, prefill+decode coexist
    TestAbort                 — running/queued/unknown
    TestRequestBookkeeping    — seq_len, has_room, finish_reason
    TestPrefixCacheIntegration— admission hit, finish release, preempt reset
"""

from __future__ import annotations

import pytest

from lite_llama.engine.sampler import SamplingParams
from lite_llama.engine.scheduler import (
    Request,
    RequestStatus,
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
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


def make_request_with_tokens(
    request_id: str, token_ids: list[int], **params
) -> Request:
    """A request with explicit token ids (for prefix-cache tests)."""
    return Request(
        request_id=request_id,
        prompt="x" * len(token_ids),
        prompt_token_ids=token_ids,
        params=SamplingParams(**params),
    )


def _decode_once(output: SchedulerOutput) -> None:
    """Give every decoding request one token, then advance prefill chunks."""
    for r in output.decode:
        r.output_token_ids.append(999)


@pytest.fixture
def scheduler() -> Scheduler:
    """Four slots and a small token budget, so limits are reachable in one step."""
    return Scheduler(
        SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=4, max_num_batched_tokens=64),
        num_slots=4,
    )


# --------------------------------------------------------------------------- #
# 1. Request admission
# --------------------------------------------------------------------------- #
class TestRequestAdmission:
    """Generation cap resolution and unservable prompt refusal."""

    def test_generation_cap_is_resolved_against_the_context_window(self, scheduler):
        """``max_gen_len=None`` means "fill the window", not "unbounded"."""
        request = make_request("a", prompt_len=100)
        scheduler.add_request(request)
        assert request.max_new_tokens == _MAX_SEQ_LEN - 100

    def test_generation_cap_honours_an_explicit_request(self, scheduler):
        request = make_request("a", prompt_len=10, max_gen_len=7)
        scheduler.add_request(request)
        assert request.max_new_tokens == 7

    def test_generation_cap_is_clamped_by_the_window(self, scheduler):
        """A request asking for more than fits gets what fits, not a later overflow."""
        request = make_request("a", prompt_len=120, max_gen_len=1000)
        scheduler.add_request(request)
        assert request.max_new_tokens == _MAX_SEQ_LEN - 120

    @pytest.mark.parametrize(
        ("prompt_len", "why"),
        [(0, "empty prompt"), (_MAX_SEQ_LEN, "no room left to generate")],
    )
    def test_unservable_prompts_are_refused_at_submission(
        self, scheduler, prompt_len, why
    ):
        """Refusing here is what lets the engine assume every admitted request fits."""
        with pytest.raises(ValueError):
            scheduler.add_request(make_request("a", prompt_len=prompt_len))
        assert scheduler.num_waiting == 0, why

    def test_max_num_seqs_cannot_exceed_the_available_slots(self):
        """Concurrency is bounded by cache slots, whatever the caller asked for."""
        sched = Scheduler(
            SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=64),
            num_slots=3,
        )
        assert sched.max_num_seqs == 3

    def test_max_num_seqs_can_exceed_slots_with_preemption(self):
        """enable_preemption lets max_num_seqs go beyond slot count."""
        sched = Scheduler(
            SchedulerConfig(
                max_seq_len=_MAX_SEQ_LEN,
                max_num_seqs=10,
                enable_preemption=True,
            ),
            num_slots=3,
        )
        assert sched.max_num_seqs == 10

    def test_zero_slots_is_rejected(self):
        with pytest.raises(ValueError):
            Scheduler(SchedulerConfig(max_seq_len=_MAX_SEQ_LEN), num_slots=0)


# --------------------------------------------------------------------------- #
# 2. Prefill scheduling
# --------------------------------------------------------------------------- #
class TestPrefillScheduling:
    """FCFS order, token budget gating, and prefill+decode coexistence."""

    def test_prefill_admits_in_arrival_order(self, scheduler):
        for name in "abc":
            scheduler.add_request(make_request(name))
        assert [r.request_id for r in scheduler.schedule().prefill] == ["a", "b", "c"]

    def test_concurrency_is_capped_and_the_rest_stay_queued(self, scheduler):
        for index in range(6):
            scheduler.add_request(make_request(f"r{index}"))
        admitted = scheduler.schedule().prefill
        assert len(admitted) == 4  # max_num_seqs
        assert scheduler.num_waiting == 2
        assert all(r.status is RequestStatus.WAITING for r in scheduler.waiting)

    def test_token_budget_bounds_the_padded_prefill_grid(self, scheduler):
        """Prefill pads to the longest prompt, so the budget is measured padded.

        Two 40-token prompts cost 80 padded token-slots against a 64 budget, so the
        second waits even though three slots are still free.
        """
        scheduler.add_request(make_request("long-a", prompt_len=40))
        scheduler.add_request(make_request("long-b", prompt_len=40))
        assert [r.request_id for r in scheduler.schedule().prefill] == ["long-a"]
        assert scheduler.num_waiting == 1

    def test_a_prompt_bigger_than_the_budget_still_runs(self, scheduler):
        """Otherwise it wedges the FCFS queue permanently."""
        scheduler.add_request(make_request("huge", prompt_len=100))
        scheduler.add_request(make_request("small", prompt_len=4))
        assert [r.request_id for r in scheduler.schedule().prefill] == ["huge"]
        assert scheduler.num_waiting == 1

    def test_prefill_takes_priority_over_decode(self, scheduler):
        """A queued arrival gets prefilled alongside running decode (chunked prefill)."""
        scheduler.add_request(make_request("a"))
        scheduler.schedule()  # admits and prefills `a`
        scheduler.add_request(make_request("b"))
        output = scheduler.schedule()
        assert [r.request_id for r in output.prefill] == ["b"]
        assert any(r.request_id == "a" for r in output.decode)  # coexists


# --------------------------------------------------------------------------- #
# 3. Chunked prefill
# --------------------------------------------------------------------------- #
class TestChunkedPrefill:
    """Long prompts split into chunks; decode interleaves with prefill chunks."""

    @pytest.fixture
    def chunked_scheduler(self) -> Scheduler:
        return Scheduler(
            SchedulerConfig(
                max_seq_len=_MAX_SEQ_LEN,
                max_num_seqs=4,
                max_num_batched_tokens=1 << 20,
                max_chunk_size=32,
            ),
            num_slots=4,
        )

    def test_long_prompt_is_split_into_chunks(self, chunked_scheduler):
        """A 100-token prompt with max_chunk_size=32 takes 4 chunks."""
        sched = chunked_scheduler
        sched.add_request(make_request("long", prompt_len=100))
        out = sched.schedule()
        assert out.prefill_chunk_lens == [32]
        assert sched.num_running == 1  # admitted but still chunking

    def test_chunk_lens_are_correct_per_step(self, chunked_scheduler):
        """Each step processes at most max_chunk_size tokens of the remaining prompt."""
        sched = chunked_scheduler
        sched.add_request(make_request("long", prompt_len=100))
        expected = [32, 32, 32, 4]  # 100 = 32*3 + 4
        actual = []
        for _ in range(4):
            out = sched.schedule()
            actual += out.prefill_chunk_lens
            sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
        assert actual == expected

    def test_advance_chunks_moves_request_to_decode(self, chunked_scheduler):
        """After all chunks are processed, the request leaves the chunking list."""
        sched = chunked_scheduler
        sched.add_request(make_request("long", prompt_len=100))
        for _ in range(4):
            out = sched.schedule()
            sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
        # After 4 chunks (32*3+4=100), the request is no longer chunking.
        out = sched.schedule()
        assert not any(r.request_id == "long" for r in out.prefill)
        assert any(r.request_id == "long" for r in out.decode)

    def test_decode_runs_alongside_prefill_chunks(self, chunked_scheduler):
        """Chunked prefill does not block already-decoding requests."""
        sched = chunked_scheduler
        sched.add_request(make_request("short-a", prompt_len=10))
        sched.schedule()  # prefill short-a completely
        sched.advance_chunks([], [])
        # short-a is now decoding; long-b starts chunking.
        sched.add_request(make_request("long-b", prompt_len=100))
        out = sched.schedule()
        assert any(r.request_id == "short-a" for r in out.decode)
        assert any(r.request_id == "long-b" for r in out.prefill)

    def test_max_chunk_size_zero_disables_chunking(self):
        """max_chunk_size=0 means no chunking — the full prompt is one prefill."""
        sched = Scheduler(
            SchedulerConfig(
                max_seq_len=_MAX_SEQ_LEN,
                max_num_seqs=4,
                max_num_batched_tokens=1 << 20,
                max_chunk_size=0,
            ),
            num_slots=4,
        )
        sched.add_request(make_request("long", prompt_len=100))
        out = sched.schedule()
        assert out.prefill_chunk_lens == [100]


# --------------------------------------------------------------------------- #
# 4. Decode scheduling
# --------------------------------------------------------------------------- #
class TestDecodeScheduling:
    """Full decode coverage and idle no-op."""

    def test_decode_covers_every_running_request(self, scheduler):
        for name in "abc":
            scheduler.add_request(make_request(name))
        scheduler.schedule()  # prefill all three
        output = scheduler.schedule()
        assert output.prefill == []
        assert [r.request_id for r in output.decode] == ["a", "b", "c"]

    def test_an_idle_scheduler_schedules_nothing(self, scheduler):
        assert scheduler.schedule().is_empty
        assert not scheduler.has_unfinished_requests()


# --------------------------------------------------------------------------- #
# 5. Slot accounting
# --------------------------------------------------------------------------- #
class TestSlotAccounting:
    """Slot conservation: no leaks, no double-issues."""

    def test_running_requests_hold_distinct_slots(self, scheduler):
        for index in range(4):
            scheduler.add_request(make_request(f"r{index}"))
        admitted = scheduler.schedule().prefill
        slots = [r.slot for r in admitted]
        assert len(set(slots)) == 4, "two requests sharing a slot would share KV"
        assert all(0 <= slot < 4 for slot in slots)
        assert scheduler.num_free_slots == 0

    def test_finishing_returns_the_slot_and_retires_the_request(self, scheduler):
        request = make_request("a")
        scheduler.add_request(request)
        scheduler.schedule()
        slot = request.slot
        scheduler.finish(request, "eos")
        assert request.status is RequestStatus.FINISHED
        assert request.finish_reason == "eos"
        assert request.slot is None
        assert scheduler.num_running == 0
        assert scheduler.num_free_slots == 4, f"slot {slot} leaked"

    def test_finishing_twice_does_not_double_free_the_slot(self, scheduler):
        """A double free would hand one slot to two later requests."""
        request = make_request("a")
        scheduler.add_request(request)
        scheduler.schedule()
        scheduler.finish(request, "eos")
        scheduler.finish(request, "length")
        assert scheduler.num_free_slots == 4
        assert request.finish_reason == "eos", "the first reason is the true one"

    def test_a_freed_slot_is_reused_by_a_queued_request(self, scheduler):
        for index in range(5):
            scheduler.add_request(make_request(f"r{index}"))
        admitted = scheduler.schedule().prefill
        freed = admitted[1].slot
        scheduler.finish(admitted[1], "eos")
        next_group = scheduler.schedule().prefill
        assert [r.request_id for r in next_group] == ["r4"]
        assert next_group[0].slot == freed

    def test_many_finish_and_admit_rounds_conserve_slots(self, scheduler):
        """The leak this catches only surfaces as a stuck engine much later."""
        for round_index in range(20):
            request = make_request(f"r{round_index}")
            scheduler.add_request(request)
            scheduler.schedule()
            scheduler.finish(request, "eos")
        assert scheduler.num_free_slots == 4
        assert scheduler.num_running == 0
        assert not scheduler.has_unfinished_requests()

    def test_running_requests_are_never_preempted_without_enable(self, scheduler):
        """Without enable_preemption, admitted requests keep their slots."""
        for index in range(6):
            scheduler.add_request(make_request(f"r{index}"))
        admitted = scheduler.schedule().prefill
        held = {r.request_id: r.slot for r in admitted}
        for _ in range(5):
            scheduler.schedule()
        assert {r.request_id: r.slot for r in scheduler.running} == held


# --------------------------------------------------------------------------- #
# 6. Preemption (recompute, opt-in oversubscription)
# --------------------------------------------------------------------------- #
def _oversubscribed(num_slots: int = 2, max_num_seqs: int = 3) -> Scheduler:
    return Scheduler(
        SchedulerConfig(
            max_seq_len=_MAX_SEQ_LEN,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=1 << 20,
            max_chunk_size=0,
            enable_preemption=True,
        ),
        num_slots=num_slots,
    )


class TestPreemption:
    """Opt-in oversubscription via recompute preemption."""

    def test_preemption_disabled_by_default(self):
        """Without enable_preemption the batch stays slot-capped and nothing evicts."""
        sched = Scheduler(
            SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=8),
            num_slots=2,
        )
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        for _ in range(4):
            out = sched.schedule()
            _decode_once(out)
            sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
        assert sched.num_preemptions == 0

    def test_oversubscription_admits_beyond_slots(self):
        """max_num_seqs may exceed slots; the extra request runs by preempting."""
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        seen: set[str] = set()
        for _ in range(6):
            out = sched.schedule()
            for r in out.decode:
                seen.add(r.request_id)
            _decode_once(out)
            sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
        # All three requests get decode turns despite only two slots.
        assert seen == {"r0", "r1", "r2"}
        assert sched.num_preemptions >= 1

    def test_preempted_request_is_reported_in_output(self):
        """SchedulerOutput.preempted names who was evicted this step."""
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        preempted_ids: list[str] = []
        for _ in range(5):
            out = sched.schedule()
            preempted_ids += [r.request_id for r in out.preempted]
            _decode_once(out)
            sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
        assert preempted_ids  # at least one preemption was surfaced

    def test_progress_quantum_prevents_livelock(self):
        """A just-recomputed request is protected until it decodes once.

        Without the quantum, two requests could preempt each other every step and
        neither would ever emit a token.
        """
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        total_tokens = 0
        for _ in range(9):
            out = sched.schedule()
            total_tokens += len(out.decode)
            _decode_once(out)
            sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
        # Real forward progress: many tokens produced, not a stalled ping-pong.
        assert total_tokens >= 6

    def test_preemption_fairness_all_requests_progress(self):
        """Under sustained oversubscription, no request is permanently starved."""
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        decode_counts: dict[str, int] = {"r0": 0, "r1": 0, "r2": 0}
        for _ in range(12):
            out = sched.schedule()
            for r in out.decode:
                decode_counts[r.request_id] += 1
            _decode_once(out)
            sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
        # Every request got at least one decode turn.
        assert all(count > 0 for count in decode_counts.values())


# --------------------------------------------------------------------------- #
# 7. SchedulerOutput field integrity
# --------------------------------------------------------------------------- #
class TestSchedulerOutput:
    """prefill_chunk_lens, preempted, and prefill+decode coexistence."""

    def test_chunk_lens_match_remaining_tokens(self):
        sched = Scheduler(
            SchedulerConfig(
                max_seq_len=_MAX_SEQ_LEN,
                max_num_seqs=4,
                max_num_batched_tokens=1 << 20,
                max_chunk_size=50,
            ),
            num_slots=4,
        )
        sched.add_request(make_request("a", prompt_len=120))
        out = sched.schedule()
        assert out.prefill_chunk_lens == [50]  # first chunk = 50 of 120
        sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
        out = sched.schedule()
        assert out.prefill_chunk_lens == [50]  # second chunk
        sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
        out = sched.schedule()
        assert out.prefill_chunk_lens == [20]  # last chunk = 20

    def test_is_empty_when_idle(self, scheduler):
        assert scheduler.schedule().is_empty

    def test_prefill_and_decode_coexist(self, scheduler):
        """v0.7: prefill and decode can be non-empty in the same step."""
        scheduler.add_request(make_request("a"))
        scheduler.schedule()  # a is now running (decoding)
        scheduler.add_request(make_request("b"))
        out = scheduler.schedule()
        assert out.prefill and out.decode  # both non-empty

    def test_preempted_field_is_populated_on_eviction(self):
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        found_preemption = False
        for _ in range(6):
            out = sched.schedule()
            if out.preempted:
                found_preemption = True
                assert all(r.status is RequestStatus.WAITING for r in out.preempted)
                break
            _decode_once(out)
            sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
        assert found_preemption, "expected at least one preemption step"


# --------------------------------------------------------------------------- #
# 8. Abort
# --------------------------------------------------------------------------- #
class TestAbort:
    def test_aborting_a_running_request_frees_its_slot(self, scheduler):
        request = make_request("a")
        scheduler.add_request(request)
        scheduler.schedule()
        assert scheduler.abort("a") is request
        assert request.finish_reason == "abort"
        assert scheduler.num_free_slots == 4

    def test_aborting_a_queued_request_dequeues_it(self, scheduler):
        scheduler.add_request(make_request("a"))
        scheduler.add_request(make_request("b"))
        aborted = scheduler.abort("b")
        assert aborted is not None and aborted.finish_reason == "abort"
        assert scheduler.num_waiting == 1
        assert [r.request_id for r in scheduler.schedule().prefill] == ["a"]

    def test_aborting_an_unknown_id_is_a_no_op(self, scheduler):
        assert scheduler.abort("nope") is None


# --------------------------------------------------------------------------- #
# 9. Request bookkeeping
# --------------------------------------------------------------------------- #
class TestRequestBookkeeping:
    def test_seq_len_counts_prompt_plus_generated(self):
        request = make_request("a", prompt_len=5)
        request.output_token_ids.extend([1, 2, 3])
        assert request.seq_len == 8

    def test_has_room_tracks_the_generation_cap(self, scheduler):
        request = make_request("a", max_gen_len=2)
        scheduler.add_request(request)
        assert request.has_room
        request.output_token_ids.append(1)
        assert request.has_room
        request.output_token_ids.append(2)
        assert not request.has_room

    def test_is_finished_flag(self):
        request = make_request("a")
        assert not request.is_finished
        request.status = RequestStatus.FINISHED
        assert request.is_finished


# --------------------------------------------------------------------------- #
# 10. Prefix cache integration
# --------------------------------------------------------------------------- #
class TestPrefixCacheIntegration:
    """Prefix cache wired into the scheduler's admission/finish/preempt path."""

    def _sched(
        self,
        *,
        enable_prefix_cache: bool = True,
        enable_preemption: bool = False,
        max_chunk_size: int = 0,
        max_num_seqs: int = 8,
        num_slots: int = 8,
    ) -> Scheduler:
        config = SchedulerConfig(
            max_seq_len=4096,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=65536,
            max_chunk_size=max_chunk_size,
            enable_prefix_cache=enable_prefix_cache,
            enable_preemption=enable_preemption,
        )
        return Scheduler(config, num_slots=num_slots)

    def test_disabled_cache_means_zero_cached_tokens(self):
        sched = self._sched(enable_prefix_cache=False)
        shared = list(range(64))
        sched.add_request(make_request_with_tokens("a", shared))
        sched.schedule()
        sched.add_request(make_request_with_tokens("b", shared))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.num_cached_tokens == 0

    def test_second_request_hits_shared_prefix(self):
        sched = self._sched()
        shared = list(range(64))  # 4 blocks of 16
        sched.add_request(make_request_with_tokens("a", shared))
        out_a = sched.schedule()
        a = next(r for r in out_a.prefill if r.request_id == "a")
        assert a.num_cached_tokens == 0  # first request: cold

        sched.add_request(make_request_with_tokens("b", shared))
        out_b = sched.schedule()
        b = next(r for r in out_b.prefill if r.request_id == "b")
        assert b.num_cached_tokens >= 48  # at least 3 of 4 blocks
        assert b.num_cached_tokens < b.prompt_len  # never skips the whole prompt

    def test_divergent_prompts_get_zero_cached(self):
        sched = self._sched()
        sched.add_request(make_request_with_tokens("a", list(range(64))))
        sched.schedule()
        sched.add_request(make_request_with_tokens("b", list(range(1000, 1064))))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.num_cached_tokens == 0

    def test_hit_rate_grows_with_shared_requests(self):
        sched = self._sched()
        shared = list(range(64))
        rates: list[float] = []
        for name in ("a", "b", "c"):
            sched.add_request(make_request_with_tokens(name, shared))
            sched.schedule()
            rates.append(sched.prefix_cache_hit_rate)
        assert rates[0] == 0.0          # first request: cold
        assert rates[1] > 0.0          # second: hit
        assert rates[2] >= rates[1]    # third: monotonically non-decreasing

    def test_finish_releases_prefix_but_it_stays_cached(self):
        """After the first request finishes, its prefix survives (LRU persistence)."""
        sched = self._sched()
        shared = list(range(64))
        sched.add_request(make_request_with_tokens("a", shared))
        out = sched.schedule()
        a = out.prefill[0]
        sched.finish(a, "eos")
        # req b arrives after a finished — should still hit the shared prefix.
        sched.add_request(make_request_with_tokens("b", shared))
        out_b = sched.schedule()
        b = next(r for r in out_b.prefill if r.request_id == "b")
        assert b.num_cached_tokens >= 48

    def test_preempted_request_resets_cached_tokens(self):
        """Preemption clears num_cached_tokens; recompute starts from scratch."""
        sched = self._sched(
            enable_preemption=True,
            max_num_seqs=3,
            num_slots=2,
        )
        shared = list(range(64))
        for i in range(3):
            sched.add_request(make_request_with_tokens(f"r{i}", shared))
        for _ in range(6):
            out = sched.schedule()
            for r in out.decode:
                r.output_token_ids.append(999)
            sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
            if out.preempted:
                for p in out.preempted:
                    assert p.num_cached_tokens == 0
                return
        pytest.fail("expected at least one preemption within 6 steps")
