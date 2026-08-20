"""Tests for the vLLM-style :class:`LLM` entry point.

Covers the contract that ``examples/basic.py`` relies on:

* ``LLM(model=...)`` builds a working engine (and disables CUDA graphs for
  multimodal checkpoints automatically),
* ``generate`` accepts a single string or a batch and returns one
  :class:`RequestOutput` per prompt with prompt echo and a finish reason,
* passing ``images`` to a text-only model is a clear error,
* the legacy ``TextGenerator`` wrapper keeps its old return type.

Marked ``gpu``+``weights``; set ``LITE_LLAMA_TEST_MODEL_DIR`` to override the
checkpoint location.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from lite_llama import LLM, RequestOutput, SamplingParams, TextGenerator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

_DEFAULT_MODEL_DIR = "my_weight/Qwen2.5-0.5B"
_KV_TOKENS = 2048

def _resolve_model_dir() -> Path:
    candidate = Path(os.environ.get("LITE_LLAMA_TEST_MODEL_DIR", _DEFAULT_MODEL_DIR))
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[1] / candidate
    if not (candidate / "config.json").is_file() or not any(candidate.glob("*.pth")):
        pytest.skip(f"no lite_llama-format checkpoint at {candidate}")
    return candidate


@pytest.fixture(scope="module")
def model_dir() -> Path:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return _resolve_model_dir()


@pytest.fixture(scope="module")
def llm(model_dir: Path) -> LLM:
    return LLM(model=str(model_dir), max_seq_len=512, max_gpu_num_blocks=_KV_TOKENS)


def test_generate_batch_returns_request_outputs(llm: LLM):
    prompts = ["The capital of France is", "One plus one equals"]
    params = SamplingParams(temperature=0.0, max_gen_len=8)

    outputs = llm.generate(prompts, params)

    assert len(outputs) == len(prompts)
    for out, prompt in zip(outputs, prompts, strict=True):
        assert isinstance(out, RequestOutput)
        assert out.prompt == prompt
        assert out.text  # non-empty completion
        assert out.outputs[0].finish_reason in ("eos", "length", "repeat")


def test_generate_accepts_a_single_string(llm: LLM):
    outputs = llm.generate("The capital of France is", SamplingParams(temperature=0.0, max_gen_len=4))
    assert len(outputs) == 1
    assert outputs[0].prompt == "The capital of France is"


def test_generate_is_deterministic_greedy(llm: LLM):
    params = SamplingParams(temperature=0.0, max_gen_len=12)
    a = llm.generate(["The capital of France is"], params)[0].text
    b = llm.generate(["The capital of France is"], params)[0].text
    assert a == b


def test_images_rejected_on_text_model(llm: LLM):
    with pytest.raises(ValueError, match="text-only"):
        llm.generate(["hi"], SamplingParams(), images=[object()])  # type: ignore[list-item]


def test_parallel_size_placeholders(llm: LLM, model_dir: Path):
    with pytest.raises(NotImplementedError, match="tensor_parallel"):
        LLM(model=str(model_dir), tensor_parallel_size=2)
    with pytest.raises(NotImplementedError, match="data_parallel"):
        LLM(model=str(model_dir), data_parallel_size=2)


def test_legacy_text_generator_delegates(model_dir: Path):
    """TextGenerator keeps its old ``list[str]`` return shape over the new LLM."""
    gen = TextGenerator(
        checkpoints_dir=str(model_dir), max_seq_len=512, max_gpu_num_blocks=_KV_TOKENS, device="cuda"
    )
    out = gen.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_gen_len=8))
    assert isinstance(out, list) and isinstance(out[0], str) and out[0]
    del gen  # release its KV reservation for later tests
    torch.cuda.empty_cache()
