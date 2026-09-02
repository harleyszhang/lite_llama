"""CUDA Graph regression tests.

Three guarantees: graph replay matches eager outputs, repeat replays
stay stable, and capture clamps batch sizes to the request table so a
graph never launches with unshaped rows.

Usage:
    pytest tests/compile/test_cuda_graph.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lite_llama import SamplingParams, TextGenerator
from lite_llama.executor.cuda_graph import _GraphKey

pytestmark = [pytest.mark.gpu, pytest.mark.weights]


def _generate(
    model_dir: Path,
    prompts: list[str],
    params: SamplingParams,
    *,
    graph: bool,
    lazy: bool = False,
):
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
        cuda_graph_lazy=lazy,
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


@pytest.mark.parametrize(
    "prompts",
    [
        pytest.param(["The capital of France is"], id="single"),
        pytest.param(["The capital of France is", "The largest ocean is"], id="even-batch"),
    ],
)
def test_lazy_graph_matches_eager(model_dir: Path, prompts: list[str]):
    """Lazy-captured graphs replay the same decode steps eager would run.

    An on-demand capture borrows the live step's write targets for its warmup,
    so a lazy graph diverging from eager here would mean the borrowed warmup
    corrupted state the eager path would have read.
    """
    params = SamplingParams(temperature=0.0, max_gen_len=8)
    eager = _generate(model_dir, prompts, params, graph=False)
    lazy = _generate(model_dir, prompts, params, graph=True, lazy=True)
    assert eager == lazy


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
        manager = gen.engine.model_runner._graph_manager
        assert manager is not None, "expected at least the small batch sizes to capture"
        assert all(key.batch_size <= 2 for key in manager._runners)

        out = gen.generate(
            ["The capital of France is"], SamplingParams(temperature=0.0, max_gen_len=6)
        )
        assert out[0]
    finally:
        del gen
        torch.cuda.empty_cache()


def test_lazy_captures_on_demand(model_dir: Path):
    """Startup holds only the seed pair; new shapes capture on first use.

    The seed pair is min-batch×min-bucket and max-batch×max-bucket. A single
    short prompt lands exactly on the small seed (no new capture), then a
    batch-2 generate forces (2, 256) — off-seed — to be captured by the first
    decode step that needs it, not at startup.
    """
    gen = TextGenerator(
        checkpoints_dir=str(model_dir),
        max_seq_len=512,
        device="cuda",
        use_cuda_graph=True,
        cuda_graph_lazy=True,
    )
    try:
        manager = gen.engine.model_runner._graph_manager
        assert manager is not None, "lazy mode must at least capture the seed pair"
        buckets = manager.seq_len_buckets  # (256, 512) after the max_seq_len filter
        assert set(manager._runners) == {
            _GraphKey(manager.batch_sizes[0], buckets[0]),
            _GraphKey(manager.batch_sizes[-1], buckets[-1]),
        }

        out = gen.generate(
            ["The capital of France is"], SamplingParams(temperature=0.0, max_gen_len=6)
        )
        assert out[0]
        assert len(manager._runners) == 2  # the seed served every step

        out = gen.generate(
            ["The capital of France is", "The largest ocean is"],
            SamplingParams(temperature=0.0, max_gen_len=6),
        )
        assert out[0] and out[1]
        assert _GraphKey(2, 256) in manager._runners  # captured on demand
        assert len(manager._runners) == 3
    finally:
        del gen
        torch.cuda.empty_cache()
