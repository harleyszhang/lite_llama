"""CUDA Graph regression tests.

Guarantee that the fix for the "graph baked stale tensor pointers" bug stays in
place: greedy generation with the graph enabled must produce byte-identical text
to eager execution on the same checkpoint.

Marked ``gpu``+``weights`` so it is skipped in CI and on CPU-only developer
machines; set ``LITE_LLAMA_TEST_MODEL_DIR`` to override the checkpoint location.
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
    if not (candidate / "config.json").is_file() or not any(candidate.glob("*.pth")):
        pytest.skip(f"no lite_llama-format checkpoint at {candidate}")
    return candidate


@pytest.fixture(scope="module")
def model_dir() -> Path:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return _resolve_model_dir()


def test_graph_matches_eager_on_single_prompt(model_dir: Path):
    """The graph must reproduce eager output — otherwise its pointers are stale."""
    params = SamplingParams(temperature=0.0, max_gen_len=12)
    prompt = ["The capital of France is"]

    eager_gen = TextGenerator(checkpoints_dir=str(model_dir), max_seq_len=512, device="cuda")
    eager_out = eager_gen.generate(prompt, params)
    del eager_gen
    torch.cuda.empty_cache()

    graph_gen = TextGenerator(
        checkpoints_dir=str(model_dir), max_seq_len=512, device="cuda", use_cuda_graph=True
    )
    graph_out = graph_gen.generate(prompt, params)

    assert eager_out == graph_out


def test_graph_matches_eager_on_uneven_batch(model_dir: Path):
    """Batched decoding with a padded shorter prompt must also stay bit-exact."""
    params = SamplingParams(temperature=0.0, max_gen_len=8)
    prompts = ["The capital of France is", "The largest ocean is"]

    eager_gen = TextGenerator(checkpoints_dir=str(model_dir), max_seq_len=512, device="cuda")
    eager_out = eager_gen.generate(prompts, params)
    del eager_gen
    torch.cuda.empty_cache()

    graph_gen = TextGenerator(
        checkpoints_dir=str(model_dir), max_seq_len=512, device="cuda", use_cuda_graph=True
    )
    graph_out = graph_gen.generate(prompts, params)

    assert eager_out == graph_out


def test_graph_survives_repeat_calls(model_dir: Path):
    """The stale-pointer bug produced *different* garbage on each subsequent call."""
    gen = TextGenerator(
        checkpoints_dir=str(model_dir), max_seq_len=512, device="cuda", use_cuda_graph=True
    )
    params = SamplingParams(temperature=0.0, max_gen_len=6)
    first = gen.generate(["Once upon a time"], params)
    second = gen.generate(["Once upon a time"], params)
    third = gen.generate(["Once upon a time"], params)
    assert first == second == third


def test_capture_clamps_batch_sizes_to_request_table(model_dir: Path):
    """Batch sizes above ``max_request_num`` index past ``b_req_tokens_table``.

    Capturing them read out-of-bounds memory and killed the CUDA context with
    delayed CUBLAS errors. With a tiny KV pool the manager must skip those
    batch sizes instead of crashing.
    """
    gen = TextGenerator(
        checkpoints_dir=str(model_dir),
        max_seq_len=512,
        device="cuda",
        use_cuda_graph=True,
        max_gpu_num_blocks=1024,  # max_request_num = 1024 // 512 = 2
    )
    manager = gen.engine.executor._graph_manager
    assert manager is not None, "expected at least the small batch sizes to capture"
    assert all(key.batch_size <= 2 for key in manager._runners)

    out = gen.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_gen_len=6))
    assert out[0]
