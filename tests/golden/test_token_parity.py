"""Golden-output regression: optimisations must not move a single token.

Eager generation is diffed against graph replay, against a re-run in
the same process, and against the committed JSON goldens — any single
token change fails the suite.

Usage:
    pytest tests/golden/test_token_parity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lite_llama import SamplingParams, TextGenerator
from tests.golden.cases import (
    CASES,
    MAX_GPU_NUM_BLOCKS,
    MAX_SEQ_LEN,
    PENALTIES,
    case_key,
)

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

_DATA_DIR = Path(__file__).parent / "data"


def _collect(model_dir: Path, *, use_cuda_graph: bool) -> dict[str, list[str]]:
    """Generate every case/penalty combination on one generator."""
    gen = TextGenerator(
        checkpoints_dir=str(model_dir),
        max_seq_len=MAX_SEQ_LEN,
        max_gpu_num_blocks=MAX_GPU_NUM_BLOCKS,
        use_cuda_graph=use_cuda_graph,
        device="cuda",
    )
    try:
        results = {}
        for name, prompts, max_gen_len in CASES:
            for penalty in PENALTIES:
                params = SamplingParams(
                    temperature=0.0, max_gen_len=max_gen_len, repetition_penalty=penalty
                )
                results[case_key(name, penalty)] = gen.generate(prompts, params)
        return results
    finally:
        del gen
        torch.cuda.empty_cache()


def _report(expected: dict[str, list[str]], actual: dict[str, list[str]]) -> str:
    """First differing sequence, trimmed -- enough to identify the regression."""
    for key, want in expected.items():
        got = actual.get(key)
        if got == want:
            continue
        if got is None:
            return f"case {key!r} missing from output"
        for i, (a, b) in enumerate(zip(want, got, strict=False)):
            if a != b:
                return f"case {key!r} sequence {i}:\n  expected {a[:160]!r}\n  actual   {b[:160]!r}"
        return f"case {key!r} differs in sequence count: {len(want)} vs {len(got)}"
    return ""


@pytest.fixture(scope="module")
def eager_outputs(model_dir: Path) -> dict[str, list[str]]:
    return _collect(model_dir, use_cuda_graph=False)


def test_eager_matches_graph(model_dir: Path, eager_outputs):
    """CUDA-graph replay must reproduce eager output exactly, on every layout.

    Needs no stored baseline, so it stays valid for any checkpoint and cannot
    rot. This is the check that actually guards graph capture.
    """
    graph_outputs = _collect(model_dir, use_cuda_graph=True)
    assert not _report(eager_outputs, graph_outputs), _report(eager_outputs, graph_outputs)


def test_generation_is_reproducible_within_a_process(eager_outputs, model_dir: Path):
    """Re-running the same cases must give the same text.

    Rules out state leaking between ``generate()`` calls -- an unreset KV cache
    or a stale allocator cursor would show up as run-to-run drift.
    """
    again = _collect(model_dir, use_cuda_graph=False)
    assert not _report(eager_outputs, again), _report(eager_outputs, again)


def test_matches_committed_golden(model_dir: Path, eager_outputs):
    """Compare against the recorded baseline for this checkpoint, if one exists."""
    golden_path = _DATA_DIR / f"{model_dir.name}.json"
    if not golden_path.is_file():
        pytest.skip(
            f"no golden baseline for {model_dir.name!r}; record one with:\n"
            f"  .venv/bin/python scripts/golden_tokens.py --save {golden_path}"
            f" --model-dir {model_dir}"
        )

    expected = json.loads(golden_path.read_text())
    diff = _report(expected, eager_outputs)
    assert not diff, (
        f"generated text drifted from {golden_path.name}.\n{diff}\n\n"
        "If the change is intentional, re-record with:\n"
        f"  .venv/bin/python scripts/golden_tokens.py --save {golden_path}"
        f" --model-dir {model_dir}"
    )


def test_golden_baseline_covers_every_case(model_dir: Path):
    """A stale baseline missing new cases would silently check less than it claims."""
    golden_path = _DATA_DIR / f"{model_dir.name}.json"
    if not golden_path.is_file():
        pytest.skip(f"no golden baseline for {model_dir.name!r}")

    expected = json.loads(golden_path.read_text())
    wanted = {case_key(name, p) for name, _, _ in CASES for p in PENALTIES}
    assert wanted <= set(expected), f"baseline is missing: {sorted(wanted - set(expected))}"
