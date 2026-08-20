"""CUDA Graph regression tests.

A captured graph replays a fixed sequence of kernels against fixed memory
addresses. That makes two failure modes possible that eager execution simply
cannot have:

* **Stale pointers.** If capture bakes in a tensor that is later reallocated,
  replay reads whatever now occupies that address. The symptom is output that is
  plausible-looking but wrong, and *different* on each call.
* **Out-of-bounds capture.** Capturing a batch size larger than the request
  table indexes past ``b_req_tokens_table``, corrupting the CUDA context with
  delayed CUBLAS errors far from the real cause.

Every test therefore pins graph output against eager output on the same
checkpoint: eager is the oracle, and byte-identical greedy text is the contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lite_llama import SamplingParams, TextGenerator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]


def _generate(model_dir: Path, prompts: list[str], params: SamplingParams, *, graph: bool):
    """Run ``prompts`` on a fresh generator, then release its KV reservation.

    Each generator profiles and reserves its own KV pool, so leaving one alive
    would starve the next; the explicit teardown keeps the tests independent of
    execution order.
    """
    gen = TextGenerator(
        checkpoints_dir=str(model_dir),
        max_seq_len=512,
        device="cuda",
        use_cuda_graph=graph,
    )
    try:
        return gen.generate(prompts, params)
    finally:
        del gen
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "prompts",
    [
        pytest.param(["The capital of France is"], id="single"),
        pytest.param(["The capital of France is", "The largest ocean is"], id="even-batch"),
        pytest.param(
            ["Hi", "The history of the Roman Empire spans many centuries"], id="uneven-batch"
        ),
    ],
)
def test_graph_matches_eager(model_dir: Path, prompts: list[str]):
    """Graph output must equal eager output; anything else means stale pointers.

    The uneven batch matters separately: the shorter prompt is padded, so a
    graph that captured the padded extent rather than the real length would
    diverge only here.
    """
    params = SamplingParams(temperature=0.0, max_gen_len=8)
    eager = _generate(model_dir, prompts, params, graph=False)
    graphed = _generate(model_dir, prompts, params, graph=True)
    assert eager == graphed


def test_graph_survives_repeat_calls(model_dir: Path):
    """The stale-pointer bug produced *different* garbage on each later call, so
    replaying the same graph three times is what exposes it."""
    gen = TextGenerator(
        checkpoints_dir=str(model_dir), max_seq_len=512, device="cuda", use_cuda_graph=True
    )
    try:
        params = SamplingParams(temperature=0.0, max_gen_len=6)
        runs = [gen.generate(["Once upon a time"], params) for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]
    finally:
        del gen
        torch.cuda.empty_cache()


def test_capture_clamps_batch_sizes_to_request_table(model_dir: Path):
    """Batch sizes above ``max_request_num`` index past ``b_req_tokens_table``.

    With a 1024-token pool and 512-token sequences the table holds 2 requests,
    so the manager must skip larger capture batch sizes instead of crashing —
    and must still capture, and correctly replay, the small ones.
    """
    gen = TextGenerator(
        checkpoints_dir=str(model_dir),
        max_seq_len=512,
        device="cuda",
        use_cuda_graph=True,
        max_gpu_num_blocks=1024,  # max_request_num = 1024 // 512 = 2
    )
    try:
        manager = gen.engine.executor._graph_manager
        assert manager is not None, "expected at least the small batch sizes to capture"
        assert all(key.batch_size <= 2 for key in manager._runners)

        out = gen.generate(
            ["The capital of France is"], SamplingParams(temperature=0.0, max_gen_len=6)
        )
        assert out[0]
    finally:
        del gen
        torch.cuda.empty_cache()
