"""Tests for :mod:`lite_llama.engine.data_parallel`.

The coordinator's whole job is bookkeeping — route each prompt to a replica, then put
the completions back where the caller expects them — so that is what is asserted:

* **routing** buckets prompts per replica via the load balancer, total and disjoint
  (``_route`` is exercised through a lightweight stub, so this tier needs no GPU);
* **reassembly** returns completions in the caller's order, the failure that would be
  hardest to notice: every answer is individually plausible, just attached to the
  wrong prompt;
* **parity with a single GPU** holds *per replica-batch*. It deliberately does not
  compare against one run over the full prompt list: a replica batches a subset where
  one GPU batched everything, and a different batch shape changes GEMM reduction order,
  which can flip an fp16 greedy tie. Replaying the same sub-batches keeps the batch
  composition identical, so any difference is a real routing bug, not float noise.

The end-to-end tier needs two GPUs; it skips rather than fails on one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lite_llama import LLM, DataParallelEngine, RequestOutput, SamplingParams

# Enough KV for the short generations here, small enough that two replicas plus the
# single-GPU reference engine coexist on one card.
_KV_TOKENS = 4096

_PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "The sun rises in the",
    "Water boils at",
    "Machine learning is",
    "The largest planet is",
]

#: Greedy and with every early-exit heuristic off, so the only thing that can make two
#: runs differ is the arithmetic itself.
_GREEDY = SamplingParams(
    temperature=0.0, max_gen_len=24, repetition_penalty=1.0, stop_on_repeat=False
)

requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason=f"data parallelism needs 2 GPUs, found {torch.cuda.device_count()}",
)


# --------------------------------------------------------------------------- #
# Construction contract (no GPU work)
# --------------------------------------------------------------------------- #
def test_device_kwarg_is_rejected():
    """A caller-supplied device would silently put every replica on one GPU."""
    with pytest.raises(ValueError, match="device is derived"):
        DataParallelEngine(model="unused", data_parallel_size=2, device="cuda:0")


def test_non_positive_replica_count_is_rejected():
    with pytest.raises(ValueError, match="data_parallel_size must be >= 1"):
        DataParallelEngine(model="unused", data_parallel_size=0)


def test_unknown_load_balancer_is_rejected():
    with pytest.raises(ValueError, match="unknown load_balancer"):
        DataParallelEngine(model="unused", data_parallel_size=1, load_balancer="magic")


def test_requesting_more_gpus_than_exist_is_rejected(monkeypatch: pytest.MonkeyPatch):
    """The count is checked up front, not discovered by a worker failing to start."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match=r"needs 4 GPUs, but only 2 are visible"):
        DataParallelEngine(model="unused", data_parallel_size=4)

    # The same check covers the composed grid: 2 replicas x TP 2 also wants 4.
    with pytest.raises(ValueError, match=r"needs 4 GPUs, but only 2 are visible"):
        DataParallelEngine(model="unused", data_parallel_size=2, tensor_parallel_size=2)


# --------------------------------------------------------------------------- #
# Routing (no GPU work): drive _route on a bare instance
# --------------------------------------------------------------------------- #
class _RouteHarness:
    """Minimal object carrying just what ``DataParallelEngine._route`` reads.

    ``_route`` only touches ``data_parallel_size`` and ``_balancer``, so a full engine
    (and its GPUs) is unnecessary to test how prompts are bucketed.
    """

    def __init__(self, dp_size: int, policy: str = "round_robin"):
        from lite_llama.engine.dp_load_balancer import make_load_balancer

        self.data_parallel_size = dp_size
        self._balancer = make_load_balancer(policy, dp_size)

    route = DataParallelEngine._route


def test_route_buckets_round_robin():
    buckets = _RouteHarness(dp_size=2).route(["a", "b", "c", "d", "e"])
    assert buckets == [[0, 2, 4], [1, 3]]


def test_route_covers_every_prompt_exactly_once():
    buckets = _RouteHarness(dp_size=3).route([str(i) for i in range(10)])
    assigned = sorted(i for bucket in buckets for i in bucket)
    assert assigned == list(range(10))


def test_route_leaves_idle_replicas_empty():
    """Fewer prompts than replicas is legal: the extra replicas get nothing."""
    buckets = _RouteHarness(dp_size=4).route(["a", "b"])
    assert buckets == [[0], [1], [], []]


# --------------------------------------------------------------------------- #
# End to end across two GPUs
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.weights
@requires_two_gpus
class TestTwoReplicas:
    """One engine, shared by every case here: startup is seconds, generation is not."""

    @pytest.fixture(scope="class")
    def engine(self, model_dir: Path):
        with DataParallelEngine(
            model=str(model_dir),
            data_parallel_size=2,
            max_seq_len=512,
            max_gpu_num_blocks=_KV_TOKENS,
        ) as engine:
            yield engine

    def test_generate_returns_one_output_per_prompt_in_order(self, engine: DataParallelEngine):
        outputs = engine.generate(_PROMPTS, _GREEDY)

        assert len(outputs) == len(_PROMPTS)
        for out, prompt in zip(outputs, _PROMPTS, strict=True):
            assert isinstance(out, RequestOutput)
            # The prompt echo is what proves reassembly did not rotate the results.
            assert out.prompt == prompt
            assert out.text
            assert out.outputs[0].finish_reason in ("eos", "length", "repeat")

    def test_generate_accepts_a_single_string(self, engine: DataParallelEngine):
        """One prompt leaves every replica but one idle, and must still work."""
        outputs = engine.generate("The capital of France is", _GREEDY)

        assert len(outputs) == 1
        assert outputs[0].prompt == "The capital of France is"
        assert outputs[0].text

    def test_generate_handles_an_empty_batch(self, engine: DataParallelEngine):
        assert engine.generate([]) == []

    def test_generate_is_repeatable(self, engine: DataParallelEngine):
        """Same prompts, same engine, greedy: the replicas must not drift apart."""
        first = [out.text for out in engine.generate(_PROMPTS, _GREEDY)]
        second = [out.text for out in engine.generate(_PROMPTS, _GREEDY)]
        assert first == second

    def test_matches_a_single_gpu_per_replica_batch(
        self, engine: DataParallelEngine, model_dir: Path
    ):
        """Each replica's batch must produce exactly what one GPU produces for it.

        The reference engine replays the *same* buckets, so batch shapes match and the
        comparison is byte-exact — see this module's docstring for why comparing
        against one run over all six prompts would not be.
        """
        dp_texts = [out.text for out in engine.generate(_PROMPTS, _GREEDY)]

        reference = LLM(model=str(model_dir), max_seq_len=512, max_gpu_num_blocks=_KV_TOKENS)
        try:
            for bucket in engine._route(_PROMPTS):
                expected = reference.generate([_PROMPTS[i] for i in bucket], _GREEDY)
                for index, out in zip(bucket, expected, strict=True):
                    assert dp_texts[index] == out.text
        finally:
            del reference
            torch.cuda.empty_cache()

    def test_tokenizer_is_available_for_token_accounting(self, engine: DataParallelEngine):
        """Benchmarks and evals count tokens through the engine, not through a worker."""
        assert engine.tokenizer.encode("hello world")


@pytest.mark.gpu
@pytest.mark.weights
@requires_two_gpus
def test_least_loaded_engine_also_covers_every_prompt(model_dir: Path):
    """The least-loaded policy must route just as completely as round-robin."""
    with DataParallelEngine(
        model=str(model_dir),
        data_parallel_size=2,
        load_balancer="least_loaded",
        max_seq_len=512,
        max_gpu_num_blocks=_KV_TOKENS,
    ) as engine:
        outputs = engine.generate(_PROMPTS, _GREEDY)

    assert [out.prompt for out in outputs] == _PROMPTS
    assert all(out.text for out in outputs)


@pytest.mark.gpu
@pytest.mark.weights
@requires_two_gpus
def test_generate_after_shutdown_is_an_error(model_dir: Path):
    engine = DataParallelEngine(
        model=str(model_dir),
        data_parallel_size=2,
        max_seq_len=512,
        max_gpu_num_blocks=_KV_TOKENS,
    )
    engine.shutdown()
    engine.shutdown()  # idempotent

    with pytest.raises(RuntimeError, match="has been shut down"):
        engine.generate(["hi"])
