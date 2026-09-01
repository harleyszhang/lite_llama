"""Step-level tests for the launch/harvest pipeline loop (O2), without a model.

A scripted executor — one that also answers ``readback_async`` the way the
StreamPool-backed worker does — drives the pipelined step loop on CPU. What
these tests pin down is the mode's contract:

* ``step()`` reports the *previous* step's tokens: one step of extra latency
  is the price of overlapping host work with compute, and the token stream
  itself must not change — same ids, same finish reason, one step later.
* The request ledger is optimistic between launch and harvest
  (``pending_tokens``) and drains to exactly zero when the request retires,
  counting the discarded extra pass a late stop rides.
* The decode plan carries placeholder token ids and optimistic lengths,
  because the real input token is still on the device.

Usage:
    pytest tests/engine/test_pipeline_engine.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lite_llama.engine.continuous_engine import ContinuousBatchingEngine, _decode_work
from lite_llama.engine.sampler import SamplingParams
from lite_llama.engine.scheduler import SchedulerConfig
from lite_llama.executor.worker import PIPELINE_ENV, PassKind

_EOS = 2
_WORD = 100  # any token id the stop set does not contain


class _FakeTokenizer:
    """The two methods the engine calls: encode for prompts, decode for deltas."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [10, 11, 12]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "x" * len(token_ids)


class _ScriptedExecutor:
    """Returns scripted token rows; every execute is followed by a readback.

    The readback hands back an independent host-side copy — like the pinned
    buffer the real pool returns — so a later pass's tokens cannot overwrite
    a buffer still awaiting its harvest.
    """

    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows
        self._calls = 0
        self.num_slots = 4
        self.readbacks = 0
        self.plans: list = []

    def execute(self, plan) -> tuple:
        self.plans.append(plan)
        row = self._rows[min(self._calls, len(self._rows) - 1)]
        self._calls += 1
        width = len(plan.sampling)
        import torch

        return torch.tensor((row * width)[:width]), None

    def readback_async(self, tokens) -> tuple:
        self.readbacks += 1
        return tokens.detach().clone(), None

    def shutdown(self) -> None:
        pass


def _build_engine(
    rows: list[list[int]], *, pipeline: bool | None = True, **scheduler_kwargs
) -> tuple[ContinuousBatchingEngine, _ScriptedExecutor]:
    """A pipelined engine over a fake LLMEngine and scripted passes."""
    fake = SimpleNamespace(
        model_runner=SimpleNamespace(spec=SimpleNamespace(is_multimodal=False)),
        device="cpu",
        tokenizer=_FakeTokenizer(),
        stop_token_ids={_EOS},
        max_seq_len=64,
    )
    executor = _ScriptedExecutor(rows)
    engine = ContinuousBatchingEngine(
        fake,
        SchedulerConfig(max_seq_len=64, max_num_seqs=4, **scheduler_kwargs),
        executor=executor,
        pipeline=pipeline,
    )
    return engine, executor


def _drain(engine: ContinuousBatchingEngine) -> list[list]:
    """The async front end's loop: step until nothing is queued or in flight."""
    steps = []
    while engine.has_unfinished_requests():
        steps.append(engine.step())
    return steps


# --------------------------------------------------------------------------- #
# The one-step-late token stream
# --------------------------------------------------------------------------- #
def test_pipeline_reports_the_same_tokens_one_step_late():
    """The pipelined loop emits the same ids as the synchronous one, shifted.

    Step 1 launches the prefill and harvests nothing; from then on every
    step harvests the *previous* pass while launching the next. The stream
    must not lose or gain a token for it — the eos finish arrives one step
    later with the same reason, and the extra pass the late stop rides is
    discarded, not appended.
    """
    engine, executor = _build_engine([[_WORD], [_WORD], [_EOS]])
    request = engine.add_request("hi")

    steps = _drain(engine)
    engine.shutdown()

    # First step reports nothing: the prefill is launched, not harvested.
    assert steps[0] == []
    reported = [step for step in steps if step]
    assert all(step[0] is request for step in reported)

    last = reported[-1]
    assert request.finish_reason == "eos"
    assert [r.request_id for r in last] == [request.request_id]
    assert request.delta == ""  # the stop token is punctuation, not output
    # The discarded extra pass never lands in the output.
    assert request.output_token_ids == [_WORD, _WORD]
    # The ledger closed: one decrement per launch, even the discarded one.
    assert request.pending_tokens == 0
    # Every launched pass asked for exactly one readback.
    assert executor.readbacks == len(executor.plans)


def test_pipeline_matches_the_synchronous_token_stream():
    """Same script, both loops: the output ids and the finish must agree."""
    def run(pipeline: bool) -> ContinuousBatchingEngine:
        engine, _ = _build_engine([[_WORD], [_WORD], [_EOS]], pipeline=pipeline)
        request = engine.add_request("hi")
        _drain(engine)
        engine.shutdown()
        return request

    pipelined, synchronous = run(True), run(False)
    assert pipelined.output_token_ids == synchronous.output_token_ids
    assert pipelined.finish_reason == synchronous.finish_reason


# --------------------------------------------------------------------------- #
# The optimistic ledger
# --------------------------------------------------------------------------- #
def test_pipeline_pending_tokens_rises_at_launch_and_drains_to_zero():
    engine, _ = _build_engine([[_WORD], [_WORD], [_WORD], [_WORD]])
    request = engine.add_request("hi")

    engine.step()  # prefill launched, nothing harvested yet
    assert request.pending_tokens == 1
    engine.step()  # decode launched (+1), prefill harvested (-1)
    assert request.pending_tokens == 1

    _drain(engine)
    engine.shutdown()
    assert request.pending_tokens == 0


def test_decode_work_from_device_plans_the_optimistic_ledger():
    """The device-fed decode plan: placeholder token, lengths one ahead."""
    request = SimpleNamespace(
        slot=3,
        seq_len=7,  # prompt (5) plus the two tokens the host has harvested
        output_token_ids=[5, 6],
        pending_tokens=1,  # the device has sampled one more
        params=None,
    )
    plan = _decode_work([request], from_device=True).plan
    assert plan.tokens == (-1,)  # replaced by the worker's device-side gather
    assert plan.seq_lens == (8,)
    assert plan.seq_starts == (7,)  # the row its unharvested token lands at
    assert plan.gen_counts == (3,)


def test_pipeline_decode_plans_carry_placeholders():
    """Every decode pass the pipeline launches feeds ids from the device."""
    engine, executor = _build_engine([[_WORD], [_WORD], [_WORD], [_WORD]])
    engine.add_request("hi")
    _drain(engine)
    engine.shutdown()

    decode_plans = [p for p in executor.plans if p.kind is PassKind.DECODE]
    assert decode_plans, "the drain must have launched decode passes"
    assert all(p.tokens == tuple(-1 for _ in p.tokens) for p in decode_plans)


# --------------------------------------------------------------------------- #
# Late stops and guards
# --------------------------------------------------------------------------- #
def test_pipeline_length_stop_pays_the_late_pass():
    """A length stop retires one step later, at the same output length."""
    engine, _ = _build_engine([[_WORD], [_WORD], [_WORD]])
    request = engine.add_request("hi", sampling_params=SamplingParams(max_gen_len=2))

    steps = _drain(engine)
    engine.shutdown()

    reported = [step for step in steps if step]
    assert request.finish_reason == "length"
    assert request.output_token_ids == [_WORD, _WORD]
    assert request.pending_tokens == 0
    # The finish is reported, never stranded: the drain heard it on the step
    # that harvested the second token.
    assert reported[-1][0] is request


def test_pipeline_rejects_preemption():
    """Planning one token ahead cannot coexist with recompute preemption."""
    with pytest.raises(ValueError, match="preemption"):
        _build_engine([[_WORD]], enable_preemption=True)


def test_pipeline_env_flag_selects_the_mode(monkeypatch):
    monkeypatch.setenv(PIPELINE_ENV, "1")
    engine, _ = _build_engine([[_WORD]], pipeline=None)
    assert engine._pipeline
    engine.shutdown()

    monkeypatch.setenv(PIPELINE_ENV, "0")
    engine, _ = _build_engine([[_WORD]], pipeline=None)
    assert not engine._pipeline
    engine.shutdown()


# --------------------------------------------------------------------------- #
# Mixed steps
# --------------------------------------------------------------------------- #
def test_pipeline_harvests_both_passes_of_a_mixed_step():
    """A step launching a prefill beside a decode harvests both next step.

    One launch list, one harvest list: the request joining mid-flight rides
    the same one-step-late reporting as the one already decoding.
    """
    engine, _ = _build_engine([[11], [22], [33], [44], [55]])
    first = engine.add_request("hi")

    engine.step()  # prefill for the first request only
    second = engine.add_request("lo")

    # This step launches the newcomer's prefill alongside the veteran's
    # decode, while harvesting the first prefill.
    stepped = engine.step()
    assert [r.request_id for r in stepped] == [first.request_id]
    assert first.output_token_ids == [11]

    _drain(engine)
    engine.shutdown()

    # Both requests heard their own tokens; the scripted rows never crossed.
    assert first.output_token_ids[0] == 11
    assert second.output_token_ids[0] == 22
    assert first.pending_tokens == 0
    assert second.pending_tokens == 0
