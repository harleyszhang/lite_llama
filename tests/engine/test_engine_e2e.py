"""End-to-end generation tests over a real converted checkpoint.

These exercise the full stack — registry -> loader -> executor -> engine ->
sampler -> detokenizer — which is the only place where an interaction bug
between two correct-looking layers shows up. The checkpoint is supplied by the
``model_dir`` fixture in ``tests/conftest.py``; the ``gpu``/``weights`` marks
below let it skip cleanly when either is unavailable.

The invariants asserted here are all *relational* (same input twice, batched vs
individual, streamed vs blocking) rather than golden strings, so they hold for
any checkpoint the fixture points at.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lite_llama import SamplingParams, TextGenerator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]


@pytest.fixture(scope="module")
def generator(model_dir: Path) -> TextGenerator:
    """One generator for the module: reloading weights per test dominates runtime."""
    return TextGenerator(checkpoints_dir=str(model_dir), max_seq_len=512, device="cuda")


def test_greedy_generation_is_deterministic(generator: TextGenerator):
    params = SamplingParams(temperature=0.0, top_p=0.9, max_gen_len=8)
    first = generator.generate(["Q: 2 + 2 = ?\nA:"], params)
    second = generator.generate(["Q: 2 + 2 = ?\nA:"], params)
    assert first[0] == second[0]
    assert len(first[0]) > 0


def test_batched_generation_matches_per_sequence(generator: TextGenerator):
    """Batching must not change results: it only changes how work is grouped.

    A mismatch here means cross-sequence contamination — a shared buffer, a
    wrong KV row, or a mask that spans the batch boundary.
    """
    prompts = ["The capital of France is", "The largest ocean is"]
    params = SamplingParams(temperature=0.0, max_gen_len=6)
    batched = generator.generate(prompts, params)
    individual = [generator.generate([p], params)[0] for p in prompts]
    assert batched == individual


def test_streaming_and_blocking_agree(generator: TextGenerator):
    """Stream deltas must concatenate to exactly what generate() returns.

    Guards the incremental detokenizer's window arithmetic: dropping or
    double-emitting a fragment shows up as a diff against the blocking path.
    """
    params = SamplingParams(temperature=0.0, max_gen_len=6)
    prompt = "Hello world"
    blocking = generator.generate([prompt], params)[0]

    streamed = "".join(step[0] for step in generator.stream([prompt], params))
    assert streamed == blocking


def test_multiple_calls_reuse_kv_cache_without_leaks(generator: TextGenerator):
    """Ten small requests must all fit the profiled cache; a leak would OOM by run 10."""
    params = SamplingParams(temperature=0.0, max_gen_len=4)
    for _ in range(10):
        assert generator.generate(["Once upon a time"], params)[0]


def test_mixed_length_batch_matches_individual(generator: TextGenerator):
    """Regression: prefill flattens the padded ``[batch, max_len]`` grid row-major.

    A packed (sum-of-lengths) tokens table pointed every sequence after the
    first at the previous sequence's tail rows, silently corrupting mixed-length
    batches. The length gap here is wide enough that the two layouts diverge.
    """
    prompts = ["Hi", "The history of the Roman Empire spans many centuries, and"]
    params = SamplingParams(temperature=0.0, max_gen_len=6)
    batched = generator.generate(prompts, params)
    individual = [generator.generate([p], params)[0] for p in prompts]
    assert batched == individual


def test_longer_batch_keeps_sequences_independent(generator: TextGenerator):
    """Four distinct prompts at once: catches off-by-one row mapping that a
    2-sequence batch is too small to expose."""
    prompts = [
        "The capital of France is",
        "Water boils at",
        "The largest planet is",
        "Two plus two equals",
    ]
    params = SamplingParams(temperature=0.0, max_gen_len=5)
    batched = generator.generate(prompts, params)
    individual = [generator.generate([p], params)[0] for p in prompts]
    assert batched == individual
