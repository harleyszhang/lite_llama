"""Performance guard for data parallelism: two replicas must actually serve twice.

Two GPU replicas run the same fixed workload once singly and once in
parallel; throughput must scale and every prompt must be answered — a
lower bound, not a microbenchmark.

Usage:
    pytest tests/distributed/test_dp_perf.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lite_llama import DataParallelEngine, SamplingParams
from tests.distributed.tp_harness import needs_gpus

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

_MAX_SEQ_LEN = 512
_KV_TOKENS = 4096
_MAX_GEN = 64

#: Enough requests that per-dispatch overhead is amortised and both replicas stay
#: busy for the whole measurement; short enough to keep the test under a minute.
_PROMPTS = [
    "Explain what a GPU does.",
    "Write a short poem about rain.",
    "List four prime numbers.",
    "What is the capital of Japan?",
    "Describe the colour blue.",
    "Name three programming languages.",
    "Summarise the water cycle.",
    "Why is the sky blue?",
    "Give one use of a hash table.",
    "What does a compiler do?",
    "Name a famous bridge.",
    "Explain recursion briefly.",
]

#: Two cards on an idle machine measure ~1.9x. Set low enough that a busy GPU or a
#: noisy neighbour cannot fail the build, high enough that a serialised
#: coordinator (~1.0x) cannot pass it.
_MIN_SCALING = 1.4

_GREEDY = SamplingParams(temperature=0.0, max_gen_len=_MAX_GEN, repetition_penalty=1.0)


def _serve(model_dir: Path, replicas: int) -> tuple[float, int]:
    """Serve the prompt set on ``replicas`` replicas; return seconds and answers.

    A short warm-up dispatch first: the first generation on a fresh replica pays
    for lazily compiled Triton kernels, and charging that to the two-replica run
    (which pays it twice, in parallel) would measure the compiler.
    """
    engine = DataParallelEngine(
        model=str(model_dir),
        data_parallel_size=replicas,
        max_seq_len=_MAX_SEQ_LEN,
        max_gpu_num_blocks=_KV_TOKENS,
        max_num_seqs=len(_PROMPTS),
        use_cuda_graph=False,
    )
    try:
        engine.generate(_PROMPTS[:replicas], SamplingParams(temperature=0.0, max_gen_len=4))

        started = time.perf_counter()
        outputs = engine.generate(_PROMPTS, _GREEDY)
        elapsed = time.perf_counter() - started
        return elapsed, sum(1 for output in outputs if output.outputs[0].text.strip())
    finally:
        engine.shutdown()


@pytest.fixture(scope="module")
def scaling(model_dir: Path) -> tuple[tuple[float, int], tuple[float, int]]:
    """One measurement per width, run one after the other so neither is starved."""
    one = _serve(model_dir, replicas=1)
    two = _serve(model_dir, replicas=2)
    return one, two


@needs_gpus(2)
def test_a_second_replica_adds_its_own_throughput(scaling):
    """The wall clock for a fixed prompt set must drop close to proportionally."""
    (one_s, _), (two_s, _) = scaling
    ratio = one_s / two_s
    print(f"\ndp scaling: {one_s:.2f}s -> {two_s:.2f}s ({ratio:.2f}x)")
    assert ratio > _MIN_SCALING, (
        f"two replicas served the set in {two_s:.2f}s against {one_s:.2f}s on one "
        f"({ratio:.2f}x, wanted >{_MIN_SCALING}x): the replicas are not running "
        f"concurrently"
    )


@needs_gpus(2)
def test_both_widths_answered_every_prompt(scaling):
    """Guards the ratio: half the work done twice as fast is not a speed-up."""
    for width, (_elapsed, answered) in zip((1, 2), scaling, strict=True):
        assert answered == len(_PROMPTS), f"dp={width} answered {answered}/{len(_PROMPTS)}"
