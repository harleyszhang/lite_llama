"""Shared fixtures and collection policy for the lite_llama test suite.

Three things live here so that no individual test file has to re-derive them:

* **Checkpoint discovery.** Integration tests need a HuggingFace checkpoint
  directory. Resolving its path per-file previously went wrong: the helper was
  copy-pasted into three files as ``Path(__file__).parents[1] / "my_weight/..."``,
  which pointed at ``tests/my_weight`` once the files moved into subdirectories.
  Every weights-gated test then skipped even on a machine that had the weights,
  so the whole integration tier was silently dead. It is computed once here,
  relative to the repository root.

* **Automatic marking.** Anything under ``tests/kernels/`` calls a Triton kernel
  and therefore needs a GPU; the ``gpu`` mark is applied by directory instead of
  by a ``pytestmark`` line that a new file can forget.

* **Skip decisions.** ``gpu`` skips without CUDA, ``weights`` skips without a
  checkpoint. The skip reason names the thing that was missing, so a skipped
  run is diagnosable rather than merely green.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# tests/conftest.py -> tests/ -> repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: HuggingFace checkpoint used by the integration tier. Override with
#: ``LITE_LLAMA_TEST_MODEL_DIR`` to point at any other checkpoint.
DEFAULT_MODEL_DIR = "my_weight/Qwen2.5-0.5B"

#: Directories whose tests always need a CUDA device.
_GPU_ONLY_DIRS = ("kernels",)


def _resolve_model_dir() -> Path:
    """Absolute path of the checkpoint under test, without validating it."""
    candidate = Path(os.environ.get("LITE_LLAMA_TEST_MODEL_DIR", DEFAULT_MODEL_DIR))
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _checkpoint_problem(path: Path) -> str | None:
    """Describe why ``path`` is unusable as a checkpoint, or ``None`` if it is."""
    if not path.is_dir():
        return f"no such directory: {path}"
    if not (path / "config.json").is_file():
        return f"no config.json in {path}"
    if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
        return f"no *.safetensors or *.bin weights in {path}"
    return None


# --------------------------------------------------------------------------- #
# Collection policy
# --------------------------------------------------------------------------- #
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply directory-based marks, then skip what the machine cannot run."""
    model_dir = _resolve_model_dir()
    # Evaluated once: probing the filesystem per test item is pointless and the
    # answer cannot change mid-run.
    checkpoint_problem = _checkpoint_problem(model_dir)
    cuda_missing = not torch.cuda.is_available()

    skip_gpu = pytest.mark.skip(reason="needs a CUDA device")
    skip_weights = pytest.mark.skip(reason=f"needs a checkpoint: {checkpoint_problem}")

    for item in items:
        # Triton kernels cannot run on CPU; mark by location so new kernel
        # tests inherit the requirement automatically.
        if any(
            f"tests/{d}/" in item.nodeid or f"tests\\{d}\\" in item.nodeid for d in _GPU_ONLY_DIRS
        ):
            item.add_marker(pytest.mark.gpu)

        if cuda_missing and "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if checkpoint_problem and "weights" in item.keywords:
            item.add_marker(skip_weights)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_torch_state():
    """Seed every test identically and drop cached blocks afterwards.

    Kernel tests compare against a reference on random inputs, so an unseeded
    run would be non-reproducible: a failure could not be re-triggered.
    """
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Validated checkpoint directory; skips the test when it is unusable."""
    path = _resolve_model_dir()
    problem = _checkpoint_problem(path)
    if problem:
        pytest.skip(f"needs a checkpoint: {problem}")
    return path


@pytest.fixture
def cuda_available() -> bool:
    """Skip unless CUDA is present. Prefer the ``gpu`` mark in new tests."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return True
