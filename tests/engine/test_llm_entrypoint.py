"""Tests for the vLLM-style :class:`LLM` entry point.

Covers the contract ``examples/basic.py`` relies on:

* ``LLM(model=...)`` builds a working engine (disabling CUDA graphs for
  multimodal checkpoints automatically),
* ``generate`` takes a single string or a batch and returns one
  :class:`RequestOutput` per prompt, with the prompt echoed and a finish reason,
* passing ``images`` to a text-only model is a clear error rather than a crash
  deep in the vision path,
* the legacy ``TextGenerator`` wrapper keeps its old ``list[str]`` return type.

The checkpoint comes from the ``model_dir`` fixture in ``tests/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lite_llama import LLM, RequestOutput, SamplingParams, TextGenerator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

# Small KV reservation so several generators can coexist within one GPU during
# the module's lifetime.
_KV_TOKENS = 2048


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
    """A bare string must not be iterated into one request per character."""
    prompt = "The capital of France is"
    outputs = llm.generate(prompt, SamplingParams(temperature=0.0, max_gen_len=4))
    assert len(outputs) == 1
    assert outputs[0].prompt == prompt


def test_generate_is_deterministic_greedy(llm: LLM):
    params = SamplingParams(temperature=0.0, max_gen_len=12)
    a = llm.generate(["The capital of France is"], params)[0].text
    b = llm.generate(["The capital of France is"], params)[0].text
    assert a == b


def test_finish_reason_is_length_when_capped(llm: LLM):
    """A 4-token cap on a continuation prompt stops on length, not EOS."""
    out = llm.generate(
        ["Count upward: one two three"], SamplingParams(temperature=0.0, max_gen_len=4)
    )[0]
    assert out.outputs[0].finish_reason == "length"


def test_images_rejected_on_text_model(llm: LLM):
    with pytest.raises(ValueError, match="text-only"):
        llm.generate(["hi"], SamplingParams(), images=[object()])  # type: ignore[list-item]


def test_parallel_size_placeholders(model_dir: Path):
    """TP is implemented; DP is not. DP must still fail loudly."""
    # tensor_parallel_size=2 should not raise NotImplementedError (TP is supported).
    # On a single-GPU machine it may fail for other reasons (NCCL), so we only
    # assert it does not raise NotImplementedError.
    try:
        llm = LLM(model=str(model_dir), tensor_parallel_size=2, max_seq_len=512, max_gpu_num_blocks=_KV_TOKENS)
        del llm
        torch.cuda.empty_cache()
    except NotImplementedError:
        pytest.fail("tensor_parallel_size=2 should not raise NotImplementedError")
    except Exception:
        pass  # NCCL/network errors are acceptable in test environments
    with pytest.raises(NotImplementedError, match="data_parallel"):
        LLM(model=str(model_dir), data_parallel_size=2)


def test_legacy_text_generator_delegates(model_dir: Path):
    """TextGenerator keeps its ``list[str]`` shape on top of the new LLM."""
    gen = TextGenerator(
        checkpoints_dir=str(model_dir),
        max_seq_len=512,
        max_gpu_num_blocks=_KV_TOKENS,
        device="cuda",
    )
    out = gen.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_gen_len=8))
    assert isinstance(out, list)
    assert isinstance(out[0], str)
    assert out[0]
    del gen  # release its KV reservation for later tests
    torch.cuda.empty_cache()
