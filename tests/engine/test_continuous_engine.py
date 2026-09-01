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

import torch

from lite_llama.engine.async_engine import AsyncLLMEngine
from lite_llama.engine.continuous_engine import ContinuousBatchingEngine
from lite_llama.engine.sampler import PositionLogprobs, SamplingParams
from lite_llama.engine.scheduler import SchedulerConfig
from lite_llama.executor.worker import PassLogprobs

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

    def execute(self, plan) -> tuple[torch.Tensor, None]:
        row = self._rows[min(self._calls, len(self._rows) - 1)]
        self._calls += 1
        width = len(plan.sampling)
        return torch.tensor((row * width)[:width]), None

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


def _record(token_id: int, logprob: float = -0.1) -> PositionLogprobs:
    return PositionLogprobs(
        token_id=token_id,
        logprob=logprob,
        top_token_ids=(token_id,),
        top_logprobs=(logprob,),
    )


class _LogprobsExecutor(_ScriptedExecutor):
    """Scripted tokens plus the logprob records the plan asks for.

    Builds records the way the real worker does: one sampled record per sampled
    row, and per chunk the prompt rows the position contract calls for — every
    row of a partial chunk, all but the sampling row of a final one. Each
    record's ``token_id`` is the row's target, so a test can verify records
    landed on the *right* position, not just the right count.
    """

    def execute(self, plan):
        row = self._rows[min(self._calls, len(self._rows) - 1)]
        self._calls += 1
        width = len(plan.sampling)
        tokens = torch.tensor((row * width)[:width])
        sampled = None
        if any(params.logprobs is not None for params in plan.sampling):
            sampled = tuple(_record(int(token)) for token in tokens)
        prompt: list = [None] * len(plan.slots)
        if plan.prompt_logprobs:
            sampled_set = set(plan.sampled)
            offset = 0
            for index in range(len(plan.slots)):
                chunk = plan.seq_lens[index] - plan.seq_starts[index]
                offset += chunk
                if plan.prompt_logprobs[index] is None:
                    continue
                rows = chunk - 1 if index in sampled_set else chunk
                targets = plan.prompt_targets[offset - chunk : offset - chunk + rows]
                prompt[index] = tuple(_record(target) for target in targets)
        return tokens, PassLogprobs(
            sampled=sampled or (),
            prompt=tuple(prompt),
        )


def _build_logprobs_engine(rows, **config) -> ContinuousBatchingEngine:
    fake = SimpleNamespace(
        model_runner=SimpleNamespace(spec=SimpleNamespace(is_multimodal=False)),
        device="cpu",
        tokenizer=_FakeTokenizer(),
        stop_token_ids={_EOS},
        max_seq_len=64,
    )
    return ContinuousBatchingEngine(
        fake,
        SchedulerConfig(max_seq_len=64, max_num_seqs=4, **config),
        executor=_LogprobsExecutor(rows),
    )


def test_logprobs_ride_the_step_to_the_request():
    """A sampled record lands on delta_logprobs for the step and output_logprobs for good."""
    engine = _build_logprobs_engine([[_WORD], [_EOS]])
    request = engine.add_request("hi", SamplingParams(logprobs=1))

    engine.step()  # prefill: one ordinary token
    assert request.delta_logprobs is not None
    assert request.delta_logprobs.token_id == _WORD
    assert len(request.output_logprobs) == 1

    engine.step()  # decode: the eos token, whose record must be dropped
    assert request.finish_reason == "eos"
    assert len(request.output_logprobs) == 1
    assert request.output_logprobs[0].token_id == _WORD
    engine.shutdown()


def test_requests_that_never_ask_carry_no_records():
    engine = _build_logprobs_engine([[_WORD], [_EOS]])
    request = engine.add_request("hi")  # no logprobs requested

    engine.step()
    assert request.delta_logprobs is None
    assert request.output_logprobs is None
    engine.shutdown()


def test_prompt_logprobs_are_attributed_across_chunks():
    """A 3-token prompt with a 2-token chunk budget scores positions 1 and 2.

    The first chunk [0, 2) is partial and scores both its rows (positions 1, 2);
    the final chunk [2, 3) has only its sampling row, so it scores nothing.
    Position 0 has no predictor and stays ``None``. The record's token id is the
    target the fake scored, which pins each record to its position.
    """
    engine = _build_logprobs_engine([[_WORD], [_EOS]], max_num_batched_tokens=2)
    request = engine.add_request("hi", SamplingParams(prompt_logprobs=1))
    assert request.prompt_len == 3  # the fake tokenizer encodes [10, 11, 12]

    engine.step()  # partial chunk [0, 2)
    assert request.prompt_logprobs[0] is None
    assert request.prompt_logprobs[1].token_id == 11
    assert request.prompt_logprobs[2].token_id == 12

    engine.step()  # final chunk + first sampled token; nothing new is attributed
    assert [r is None for r in request.prompt_logprobs] == [True, False, False]
    engine.shutdown()


def test_generate_returns_logprobs_in_the_request_output():
    engine = _build_logprobs_engine([[_WORD], [_EOS]])
    outputs = engine.generate(["hi"], SamplingParams(logprobs=1, prompt_logprobs=1))

    (output,) = outputs
    assert output.prompt_logprobs[0] is None
    assert output.prompt_logprobs[1].token_id == 11
    completion = output.outputs[0]
    assert [r.token_id for r in completion.logprobs] == [_WORD]
    engine.shutdown()


# --------------------------------------------------------------------------- #
# Observability (A7): the engine reports its own numbers
# --------------------------------------------------------------------------- #
def test_a_finished_request_lands_in_the_metrics():
    """One engine run answers for counters, histograms and the gauges."""
    engine = _build_engine([[_WORD], [_EOS]])
    engine.add_request("hi")

    engine.step()
    assert engine.metrics.prompt_tokens_total._values.get((), 0) == 0  # not finished yet
    engine.step()

    text = engine.metrics.render_prometheus()
    assert 'lite_llama:request_success_total{finish_reason="eos"} 1' in text
    assert "lite_llama:prompt_tokens_total 3" in text  # the fake prompt is 3 tokens
    assert "lite_llama:generation_tokens_total 1" in text  # the eos token is not output
    assert "lite_llama:request_queue_time_seconds_count 1" in text
    assert "lite_llama:time_to_first_token_seconds_count 1" in text
    assert "lite_llama:num_requests_running 0" in text  # drained by the end
    engine.shutdown()


def test_an_aborted_request_is_counted_without_finishing():
    engine = _build_engine([[_WORD]])
    request = engine.add_request("hi")

    engine.abort(request.request_id)

    text = engine.metrics.render_prometheus()
    assert 'lite_llama:request_success_total{finish_reason="abort"} 1' in text
    assert not engine.has_unfinished_requests()
    engine.shutdown()


def test_metrics_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LITE_LLAMA_METRICS", "0")
    engine = _build_engine([[_WORD], [_EOS]])
    engine.add_request("hi")
    engine.step()
    engine.step()

    assert not engine.metrics.enabled
    assert engine.metrics.render_prometheus() == "\n"
    engine.shutdown()
