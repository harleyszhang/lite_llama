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

import queue
from pathlib import Path
from typing import ClassVar

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
    (and its GPUs) is unnecessary to test how prompts are bucketed. It is handed token
    *estimates*, one per prompt, which is what the router computes up front.
    """

    def __init__(self, dp_size: int, policy: str = "round_robin"):
        from lite_llama.engine.dp_load_balancer import make_load_balancer

        self.data_parallel_size = dp_size
        self._balancer = make_load_balancer(policy, dp_size)

    route = DataParallelEngine._route


def test_route_buckets_round_robin():
    buckets = _RouteHarness(dp_size=2).route([1, 1, 1, 1, 1])
    assert buckets == [[0, 2, 4], [1, 3]]


def test_route_covers_every_prompt_exactly_once():
    buckets = _RouteHarness(dp_size=3).route([1] * 10)
    assigned = sorted(i for bucket in buckets for i in bucket)
    assert assigned == list(range(10))


def test_route_leaves_idle_replicas_empty():
    """Fewer prompts than replicas is legal: the extra replicas get nothing."""
    buckets = _RouteHarness(dp_size=4).route([1, 1])
    assert buckets == [[0], [1], [], []]


def test_route_sends_a_long_prompt_and_the_short_ones_apart():
    """With ``total_tokens``, prompt length — not arrival order — decides the split.

    This is the routing-level consequence of the balancer reading the estimate: the
    three short prompts must not be striped onto the replica already holding 1k tokens.
    """
    buckets = _RouteHarness(dp_size=2, policy="total_tokens").route([1000, 10, 10, 10])
    assert buckets == [[0], [1, 2, 3]]


# --------------------------------------------------------------------------- #
# The process grid (no GPU work)
# --------------------------------------------------------------------------- #
def test_token_estimates_are_skipped_when_no_policy_reads_them():
    """``round_robin`` must never load a tokenizer — the estimate would be dead cost.

    Reaching ``self.tokenizer`` on the harness would raise ``AttributeError``, so this
    asserts the short circuit rather than the number.
    """
    harness = _RouteHarness(dp_size=2)
    estimates = DataParallelEngine._estimate_tokens(harness, ["a", "bb", "ccc"])
    assert estimates == [0, 0, 0]


def test_replica_queues_hand_the_request_to_every_rank_of_the_replica():
    """A TP replica is several processes, and all of them must run the same forward.

    Sending only to the leader is the shape of the DP x TP hang: the followers sit on an
    empty queue while the leader blocks in a collective waiting for them.
    """

    class _Grid:
        tensor_parallel_size = 2
        _request_queues: ClassVar[list[str]] = ["r0t0", "r0t1", "r1t0", "r1t1"]
        _replica_queues = DataParallelEngine._replica_queues

    grid = _Grid()
    assert grid._replica_queues(0) == ["r0t0", "r0t1"]
    assert grid._replica_queues(1) == ["r1t0", "r1t1"]


class _FakeQueue:
    """In-process stand-in for ``mp.Queue`` — enough for the coordinator's handshake."""

    def __init__(self) -> None:
        self.items: list = []

    def put(self, item) -> None:
        self.items.append(item)

    def get(self, timeout: float | None = None):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)


class _FakeProcess:
    """Records the arguments a worker *would* have been spawned with, then reports ready."""

    spawned: ClassVar[list[tuple]] = []

    def __init__(self, target, args, daemon) -> None:
        self.args = args
        self._alive = True

    def start(self) -> None:
        _FakeProcess.spawned.append(self.args)
        self.args[-1].put(("ready", self.args[0], None))  # result_queue

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self._alive = False

    def terminate(self) -> None:
        self._alive = False


def test_dp_times_tp_spawns_one_process_per_grid_cell(monkeypatch: pytest.MonkeyPatch):
    """A 2x2 grid must be four processes, ranked ``dp_rank * tp_size + tp_rank``.

    This is the deadlock in assertion form. ``init_parallel(tp_size=2, dp_size=2)``
    rendezvouses a world of four, so spawning two processes hangs forever on ranks that
    were never started — a failure no timeout in the coordinator can explain. Asserting
    the grid needs no GPU at all, which is the point: the bug reproduces on a laptop.
    """
    from lite_llama.engine import data_parallel as dp_module

    _FakeProcess.spawned = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        dp_module.mp,
        "get_context",
        lambda method: type("_Ctx", (), {"Queue": _FakeQueue, "Process": _FakeProcess})(),
    )

    engine = DataParallelEngine(model="unused", data_parallel_size=2, tensor_parallel_size=2)

    assert engine.world_size == 4
    assert len(_FakeProcess.spawned) == 4
    # (global_rank, dp_rank, tp_rank) per cell, in rank order.
    assert [args[:3] for args in _FakeProcess.spawned] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 1, 0),
        (3, 1, 1),
    ]
    # Each cell owns its own request queue; sharing one would race the mirrors.
    assert len({id(args[6]) for args in _FakeProcess.spawned}) == 4

    engine.shutdown()


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
            for bucket in engine._route(engine._estimate_tokens(_PROMPTS)):
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
def test_total_tokens_engine_also_covers_every_prompt(model_dir: Path):
    """The token-aware policy must route just as completely as round-robin.

    It is the only policy that tokenises in the coordinator process, so this is also the
    end-to-end check that the estimate path works against a real tokenizer.
    """
    with DataParallelEngine(
        model=str(model_dir),
        data_parallel_size=2,
        load_balancer="total_tokens",
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
