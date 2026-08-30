"""Step-level tests for the continuous-batching engine, without a model.

The engine's step loop is plan -> execute -> harvest, and the executor boundary
is deliberately thin (pure data in, sampled tokens out), which lets a scripted
executor drive the whole loop on CPU. What that buys is coverage of the
harvest layer — stop handling, finish accounting, the step() return contract —
without a checkpoint: a bug there hangs an async stream or loses a finish
reason, and neither needs a GPU to reproduce.

The contract under test: a request that stops this step still appears in
step()'s return, carrying its finish_reason and an empty delta. The async
front end learns a request ended only from what this list hands back, so a
request missing from it strands its stream on a final chunk that never comes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from lite_llama.engine.async_engine import AsyncLLMEngine
from lite_llama.engine.continuous_engine import ContinuousBatchingEngine
from lite_llama.engine.scheduler import SchedulerConfig

_EOS = 2
_WORD = 100  # any token id the stop set does not contain
_TIMEOUT = 20.0


class _FakeTokenizer:
    """The two methods the engine calls: encode for prompts, decode for deltas."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [10, 11, 12]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "x" * len(token_ids)


class _ScriptedExecutor:
    """Returns scripted token rows, one row per ``execute`` call.

    The step loop zips what a pass returns with the requests that pass named,
    in order, so a script row is simply the token each request receives that
    pass. Rows run out and then the last one repeats, keeping a drain loop
    simple to write.
    """

    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows
        self._calls = 0
        self.num_slots = 4

    def execute(self, plan) -> torch.Tensor:
        row = self._rows[min(self._calls, len(self._rows) - 1)]
        self._calls += 1
        width = len(plan.sampling)
        return torch.tensor((row * width)[:width])

    def shutdown(self) -> None:
        pass


def _build_engine(rows: list[list[int]]) -> ContinuousBatchingEngine:
    """A real ContinuousBatchingEngine over a fake LLMEngine and scripted passes."""
    fake = SimpleNamespace(
        model_runner=SimpleNamespace(spec=SimpleNamespace(is_multimodal=False)),
        device="cpu",
        tokenizer=_FakeTokenizer(),
        stop_token_ids={_EOS},
        max_seq_len=64,
    )
    return ContinuousBatchingEngine(
        fake,
        SchedulerConfig(max_seq_len=64, max_num_seqs=4),
        executor=_ScriptedExecutor(rows),
    )


async def _collect(engine: AsyncLLMEngine, prompt: str) -> list:
    return [chunk async for chunk in engine.generate(prompt)]


def test_a_request_stopping_on_eos_is_returned_by_step():
    """Regression: an eos finish used to be dropped from step()'s return.

    The request finishes inside harvest, which used to skip past it — so a
    caller draining step() never saw the finish reason. It must come back with
    the reason set and an empty delta: the stop token is punctuation, not
    output.
    """
    engine = _build_engine([[_WORD], [_EOS]])
    request = engine.add_request("hi")

    first = engine.step()  # prefill: one ordinary token
    assert [r.request_id for r in first] == [request.request_id]
    assert request.delta
    assert request.finish_reason is None

    last = engine.step()  # decode: the stop token
    assert [r.request_id for r in last] == [request.request_id]
    assert request.finish_reason == "eos"
    assert request.delta == ""
    assert request.is_finished
    assert not engine.has_unfinished_requests()
    engine.shutdown()


def test_the_eos_stop_token_is_not_counted_as_output():
    """The request ends with only its prefill token in output_token_ids."""
    engine = _build_engine([[_WORD], [_EOS]])
    request = engine.add_request("hi")

    engine.step()
    engine.step()

    assert request.output_token_ids == [_WORD]
    engine.shutdown()


def test_duplicate_live_request_ids_are_rejected_without_losing_state():
    engine = _build_engine([[_WORD], [_EOS]])
    first = engine.add_request("first", request_id="same")

    with pytest.raises(ValueError, match="already active"):
        engine.add_request("second", request_id="same")

    assert engine.scheduler.waiting == [first]
    engine.shutdown()


def test_generated_request_ids_skip_user_supplied_ids():
    engine = _build_engine([[_WORD], [_EOS]])
    explicit = engine.add_request("first", request_id="req-0")
    generated = engine.add_request("second")

    assert explicit.request_id == "req-0"
    assert generated.request_id == "req-1"
    engine.shutdown()


async def test_an_eos_request_does_not_strand_its_stream():
    """Same regression, through the async front end: the stream must hear it.

    The worker publishes only what step() returns, so a finish missing from
    that list leaves the awaiting coroutine blocked on a final chunk that
    never arrives — a hang, not an error, invisible until a request just
    stops responding. The wait_for turns that hang into a test failure.
    """
    engine = _build_engine([[_WORD], [_EOS]])
    async with AsyncLLMEngine(engine) as async_engine:
        chunks = await asyncio.wait_for(_collect(async_engine, "hi"), _TIMEOUT)

    assert chunks[-1].finish_reason == "eos"
    assert chunks[-1].is_finished
    assert all(chunk.finish_reason is None for chunk in chunks[:-1])
    assert chunks[-1].text  # the prefill token still reached the caller
