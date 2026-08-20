"""End-to-end GPU tests using the converted Qwen2.5-0.5B checkpoint.

These tests exercise the full stack: registry -> executor -> engine -> sampler.
They are marked ``gpu`` and ``weights`` so both CI and CPU-only developers skip
them automatically. Set ``LITE_LLAMA_TEST_MODEL_DIR`` to point at any converted
lite_llama checkpoint (must contain ``config.json`` + ``*.pth``) to run locally.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from lite_llama import SamplingParams, TextGenerator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]


_DEFAULT_MODEL_DIR = "my_weight/Qwen2.5-0.5B"


def _resolve_model_dir() -> Path:
    candidate = Path(os.environ.get("LITE_LLAMA_TEST_MODEL_DIR", _DEFAULT_MODEL_DIR))
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[1] / candidate
    if not (candidate / "config.json").is_file():
        pytest.skip(f"no lite_llama-format checkpoint at {candidate}")
    if not any(candidate.glob("*.pth")):
        pytest.skip(f"no *.pth checkpoint at {candidate}")
    return candidate


@pytest.fixture(scope="module")
def generator() -> TextGenerator:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    model_dir = _resolve_model_dir()
    return TextGenerator(
        checkpoints_dir=str(model_dir),
        max_seq_len=512,
        device="cuda",
    )


def test_greedy_generation_is_deterministic(generator: TextGenerator):
    params = SamplingParams(temperature=0.0, top_p=0.9, max_gen_len=8)
    first = generator.generate(["Q: 2 + 2 = ?\nA:"], params)
    second = generator.generate(["Q: 2 + 2 = ?\nA:"], params)
    assert first[0] == second[0]
    assert len(first[0]) > 0


def test_batched_generation_matches_per_sequence(generator: TextGenerator):
    """Running two prompts in one batch must yield the same tokens as running them apart."""
    prompts = ["The capital of France is", "The largest ocean is"]
    params = SamplingParams(temperature=0.0, max_gen_len=6)
    batched = generator.generate(prompts, params)
    individual = [generator.generate([p], params)[0] for p in prompts]
    assert batched == individual


def test_streaming_and_blocking_agree(generator: TextGenerator):
    """Stream deltas must concatenate to the same string generate() returns."""
    params = SamplingParams(temperature=0.0, max_gen_len=6)
    prompt = "Hello world"
    blocking = generator.generate([prompt], params)[0]

    streamed = ""
    for step in generator.stream([prompt], params):
        streamed += step[0]
    assert streamed == blocking


def test_multiple_calls_reuse_kv_cache_without_leaks(generator: TextGenerator):
    """Ten small requests must all fit in the profiled cache; leaks would OOM by run 10."""
    params = SamplingParams(temperature=0.0, max_gen_len=4)
    for _ in range(10):
        generator.generate(["Once upon a time"], params)


def test_mixed_length_batch_matches_individual(generator: TextGenerator):
    """Regression: prefill flattens the padded [batch, max_len] grid row-major.

    A packed (sum-of-lengths) tokens table pointed every sequence after the
    first at the previous sequence's tail rows, silently corrupting mixed-length
    batches. The token-length gap here must be large enough that the two layouts
    diverge.
    """
    prompts = ["Hi", "The history of the Roman Empire spans many centuries, and"]
    params = SamplingParams(temperature=0.0, max_gen_len=6)
    batched = generator.generate(prompts, params)
    individual = [generator.generate([p], params)[0] for p in prompts]
    assert batched == individual
