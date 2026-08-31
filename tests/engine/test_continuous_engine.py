"""CPU-side tests for the continuous engine's step loop, with a scripted model.

test_continuous_batching.py holds the GPU/weights end-to-end suite: real
checkpoint, real kernels, co-tenancy assertions. A whole class of bugs cannot
reach it, though — the engine talking to its scheduler, slot batch and sampler
is pure host-side Python, and a wiring break there needs no GPU to reproduce.
These tests drive :class:`ContinuousBatchingEngine` with a fake model runner
that emits scripted tokens, so the step loop's behaviour is pinned exactly.

The regression this file exists for: a prompt longer than ``max_chunk_size``
was re-prefilled in full every step, re-sampling the same last-position logits
each time (greedy output degenerated to one repeated token until the length
cap), because ``step()`` never advanced the scheduler's chunk tracking. The
engine now clamps ``max_chunk_size`` to 0 until the prefill attention kernel
can read a previous chunk's KV.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest import mock

import torch

from lite_llama.engine import continuous_engine
from lite_llama.engine.continuous_engine import ContinuousBatchingEngine
from lite_llama.engine.sampler import Sampler, SamplingParams
from lite_llama.engine.scheduler import SchedulerConfig

_VOCAB = 1000
_EOS = 999
_MAX_SEQ_LEN = 64
_GREEDY = SamplingParams(
    temperature=0.0, max_gen_len=16, repetition_penalty=1.0, stop_on_repeat=False
)


class _FakeTokenizer:
    """Renders token ids as ``<id>`` so per-step deltas are exact and checkable."""

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(f"<{token}>" for token in token_ids)


class _FakeSlotBatch:
    """Records the attention metadata each step set; pads nothing (no CUDA graphs)."""

    def __init__(self, num_slots: int) -> None:
        self.num_slots = num_slots
        self.prefill_calls: list[tuple[list[int], list[int]]] = []
        self.decode_calls: list[tuple[list[int], list[int]]] = []
        self._seq_lens = torch.zeros(0, dtype=torch.long)

    def begin_prefill(self, slots, prompt_lens) -> None:
        self.prefill_calls.append((list(slots), list(prompt_lens)))

    def begin_decode(self, slots, seq_lens) -> int:
        self.decode_calls.append((list(slots), list(seq_lens)))
        self._seq_lens = torch.tensor(list(seq_lens), dtype=torch.long)
        return len(slots)

    @property
    def seq_lens(self) -> torch.Tensor:
        return self._seq_lens

    def reset(self) -> None:
        pass


class _FakeModelRunner:
    """Serves scripted token sequences instead of running a model.

    A script's first token is emitted by the prefill, the rest one per decode;
    scripts must end in ``_EOS`` unless the test wants a length-cap finish.
    A *repeat* prefill of the same prompt re-emits the script's first token,
    which is what a real model does when the same prompt is prefilled again:
    the last-position logits — and so the sampled token — are identical.
    """

    def __init__(self, slot_batch: _FakeSlotBatch) -> None:
        self.spec = SimpleNamespace(is_multimodal=False)
        self._slot_batch = slot_batch
        self._pending: deque[list[int]] = deque()
        self._by_prompt: dict[tuple[int, ...], list[int]] = {}
        self._slot_script: dict[int, deque[int]] = {}
        self.prefill_forwards = 0
        self.decode_forwards = 0

    def enable_slot_kv_cache(self) -> _FakeSlotBatch:
        return self._slot_batch

    def queue_script(self, tokens: list[int]) -> None:
        """Bind the next admitted request's tokens, in admission order."""
        self._pending.append(list(tokens))

    def forward(self, input_ids, positions, _kv_buffer, logits_positions=None):
        is_prefill = logits_positions is not None
        calls = self._slot_batch.prefill_calls if is_prefill else self._slot_batch.decode_calls
        slots = calls[-1][0]
        logits = torch.full((len(slots), _VOCAB), -1e9)
        for row, slot in enumerate(slots):
            if is_prefill:
                key = tuple(input_ids[row].tolist())
                script = self._by_prompt.get(key)
                if script is None:
                    script = self._pending.popleft()
                    self._by_prompt[key] = script
                    self._slot_script[slot] = deque(script[1:])
                token = script[0]
            else:
                token = self._slot_script[slot].popleft()
            logits[row, token] = 0.0
        if is_prefill:
            self.prefill_forwards += 1
            return logits  # the model gathers last positions: [batch, vocab]
        self.decode_forwards += 1
        return logits.unsqueeze(1)  # [batch, seq=1, vocab]


class _FakeEngine:
    """The slice of :class:`LLMEngine` the continuous engine actually touches."""

    def __init__(self, runner: _FakeModelRunner) -> None:
        self.device = "cpu"
        self.max_seq_len = _MAX_SEQ_LEN
        self.pad_id = 0
        self.tokenizer = _FakeTokenizer()
        self.stop_token_ids = {_EOS}
        self.model_runner = runner
        self.sampler = Sampler()


def _build_engine(config: SchedulerConfig | None = None, num_slots: int = 4):
    slot_batch = _FakeSlotBatch(num_slots)
    runner = _FakeModelRunner(slot_batch)
    config = config or SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=num_slots)
    engine = ContinuousBatchingEngine(_FakeEngine(runner), config)
    return engine, runner, slot_batch


def _drain(engine: ContinuousBatchingEngine) -> None:
    while engine.has_unfinished_requests():
        engine.step()


# --------------------------------------------------------------------------- #
# Chunked prefill
# --------------------------------------------------------------------------- #
def test_a_prompt_longer_than_the_chunk_size_is_prefilled_once_in_full():
    """Regression: long prompts must not loop through prefill.

    Before the clamp, the scheduler tracked such a request as mid-chunk
    forever, so every step re-ran the full prefill and the harvest appended
    whatever token those identical logits produced — greedy output was one
    token repeated until the length cap.
    """
    engine, runner, slot_batch = _build_engine(
        SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=4, max_chunk_size=8)
    )
    runner.queue_script([10, 20, 30, _EOS])
    request = engine.add_request("p", _GREEDY, prompt_token_ids=[1] * 40)

    _drain(engine)

    assert request.finish_reason == "eos"
    assert request.output_token_ids == [10, 20, 30]
    assert runner.prefill_forwards == 1
    assert runner.decode_forwards == 3
    # The one prefill carried the whole prompt, not an 8-token chunk.
    assert slot_batch.prefill_calls == [([0], [40])]


def test_chunked_prefill_is_clamped_off_with_a_warning(monkeypatch):
    logger = mock.Mock()
    monkeypatch.setattr(continuous_engine, "logger", logger)
    engine, _, _ = _build_engine(
        SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=4, max_chunk_size=512)
    )
    assert engine.config.max_chunk_size == 0
    assert engine.scheduler.config.max_chunk_size == 0
    logger.warning.assert_called_once()
    assert "max_chunk_size" in logger.warning.call_args.args[0]


def test_the_chunk_clamp_stays_silent_when_chunking_is_already_off(monkeypatch):
    logger = mock.Mock()
    monkeypatch.setattr(continuous_engine, "logger", logger)
    engine, _, _ = _build_engine(
        SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=4, max_chunk_size=0)
    )
    assert engine.config.max_chunk_size == 0
    logger.warning.assert_not_called()


# --------------------------------------------------------------------------- #
# Preemption
# --------------------------------------------------------------------------- #
def test_preemption_is_clamped_off_with_a_warning(monkeypatch):
    logger = mock.Mock()
    monkeypatch.setattr(continuous_engine, "logger", logger)
    engine, _, _ = _build_engine(
        SchedulerConfig(
            max_seq_len=_MAX_SEQ_LEN,
            max_num_seqs=4,
            max_chunk_size=0,
            enable_preemption=True,
        )
    )
    assert engine.config.enable_preemption is False
    assert engine.scheduler.config.enable_preemption is False
    logger.warning.assert_called_once()
    assert "enable_preemption" in logger.warning.call_args.args[0]


def test_a_clamped_preemption_config_falls_back_to_queueing(monkeypatch):
    """Oversubscription with preemption requested still finishes every request.

    Before the clamp the scheduler evicted and recomputed the two youngest
    requests every step and neither ever reached a second decode token (the
    recompute's first token re-arms eviction eligibility immediately). With
    preemption clamped off, excess requests simply queue for a slot.
    """
    monkeypatch.setattr(continuous_engine, "logger", mock.Mock())  # silence the clamp
    engine, runner, _ = _build_engine(
        SchedulerConfig(
            max_seq_len=_MAX_SEQ_LEN,
            max_num_seqs=4,
            max_chunk_size=0,
            enable_preemption=True,
        ),
        num_slots=2,
    )
    for script in ([10, _EOS], [20, _EOS], [30, _EOS]):
        runner.queue_script(script)
    requests = [engine.add_request(f"p{i}", _GREEDY, prompt_token_ids=[i + 1]) for i in range(3)]

    _drain(engine)

    assert [r.finish_reason for r in requests] == ["eos"] * 3
    assert [r.output_token_ids for r in requests] == [[10], [20], [30]]
    assert engine.scheduler.num_preemptions == 0


# --------------------------------------------------------------------------- #
# Batch reshaping
# --------------------------------------------------------------------------- #
def test_requests_finish_independently_and_the_batch_shrinks():
    engine, runner, slot_batch = _build_engine()
    runner.queue_script([10, 11, _EOS])
    runner.queue_script([20, 21, 22, _EOS])
    first = engine.add_request("a", _GREEDY, prompt_token_ids=[1, 2, 3])
    second = engine.add_request("b", _GREEDY, prompt_token_ids=[4, 5])

    _drain(engine)

    assert first.finish_reason == "eos"
    assert first.output_token_ids == [10, 11]
    assert first.text == "<10><11>"
    assert second.finish_reason == "eos"
    assert second.output_token_ids == [20, 21, 22]
    assert second.text == "<20><21><22>"
    assert runner.prefill_forwards == 1
    assert runner.decode_forwards == 3
    # The last decode ran with the survivor alone.
    assert slot_batch.decode_calls == [([0, 1], [4, 3]), ([0, 1], [5, 4]), ([1], [5])]


def test_a_request_without_eos_stops_at_the_length_cap():
    engine, runner, _ = _build_engine()
    runner.queue_script([10, 20, 30])  # no EOS; the cap must stop it
    params = SamplingParams(
        temperature=0.0, max_gen_len=2, repetition_penalty=1.0, stop_on_repeat=False
    )
    request = engine.add_request("p", params, prompt_token_ids=[1, 2])

    _drain(engine)

    assert request.finish_reason == "length"
    assert request.output_token_ids == [10, 20]
    assert runner.prefill_forwards == 1
    assert runner.decode_forwards == 1


def test_requests_beyond_the_slot_count_are_admitted_in_waves():
    engine, runner, _ = _build_engine(SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=2))
    for script in ([10, _EOS], [20, _EOS], [30, _EOS]):
        runner.queue_script(script)
    requests = [engine.add_request(f"p{i}", _GREEDY, prompt_token_ids=[i + 1]) for i in range(3)]

    engine.step()
    assert engine.scheduler.num_running == 2
    assert engine.scheduler.num_waiting == 1
    _drain(engine)

    assert [r.finish_reason for r in requests] == ["eos"] * 3
    assert [r.output_token_ids for r in requests] == [[10], [20], [30]]
    # The third request prefilled alone, after a slot freed up.
    assert runner.prefill_forwards == 2
    assert runner.decode_forwards == 2
    assert engine.scheduler.num_free_slots == engine.scheduler.num_slots


def test_a_late_arrival_prefills_while_the_running_batch_waits_a_step():
    """One model pass per step: an admission step runs prefill, not decode."""
    engine, runner, slot_batch = _build_engine()
    runner.queue_script([10, 11, 12, _EOS])
    runner.queue_script([20, _EOS])
    first = engine.add_request("a", _GREEDY, prompt_token_ids=[1])
    engine.step()  # prefill first
    engine.step()  # decode first -> 11

    second = engine.add_request("b", _GREEDY, prompt_token_ids=[2])
    engine.step()  # prefill second; the running request idles this step

    assert runner.prefill_forwards == 2
    assert runner.decode_forwards == 1  # unchanged by the admission step
    assert first.output_token_ids == [10, 11]

    _drain(engine)

    assert first.output_token_ids == [10, 11, 12]
    assert first.finish_reason == "eos"
    assert second.output_token_ids == [20]
    assert second.finish_reason == "eos"
    assert slot_batch.decode_calls == [([0], [2]), ([0, 1], [3, 2]), ([0], [4])]
