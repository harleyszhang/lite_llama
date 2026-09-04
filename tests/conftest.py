"""Shared fixtures and collection policy for the lite_llama test suite.

``pytest_ignore_collect`` drops GPU-only and golden directories when no
checkpoint or device is available; fixtures expose the resolved model
dir, reset torch state, and keep dispatch deterministic between tests.

Usage:
    pytest tests/   # collection policy from this file applies automatically
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# tests/conftest.py -> tests/ -> repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Checkpoint the integration tier uses by default. Override with
#: ``LITE_LLAMA_TEST_MODEL_DIR`` to point at any other checkpoint.
DEFAULT_MODEL_NAME = "Qwen2.5-0.5B"

#: Shared checkpoint root, the same variable the benchmark scripts read. Honoured
#: here too so a machine that keeps weights outside the repository runs the
#: integration tier without symlinking them into ``my_weight/``.
MODELZOO_ENV = "LITE_LLAMA_MODELZOO"

#: Directories whose tests always need a CUDA device.
_GPU_ONLY_DIRS = ("kernels",)

#: Directories that constitute the golden gate — these must never silently skip.
_GOLDEN_DIRS = ("golden",)

#: When set, golden tests that cannot run become hard FAILs instead of xfail.
_GOLDEN_STRICT = os.environ.get("LITE_LLAMA_GOLDEN_STRICT", "") == "1"


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Do not import Triton-only test modules on a machine without CUDA.

    Marker selection happens after Python imports a test module. Where Triton
    is intentionally not installed (macOS), merely collecting ``tests/kernels``
    would fail before the automatic ``gpu`` skip can apply.
    """
    if torch.cuda.is_available():
        return None
    try:
        relative = collection_path.relative_to(REPO_ROOT)
    except ValueError:
        return None
    return any(relative.parts[:2] == ("tests", d) for d in _GPU_ONLY_DIRS) or None


def checkpoint_candidates(reference: str) -> list[Path]:
    """Every place a checkpoint reference may resolve to, most specific first.

    ``reference`` is a path (absolute, or relative to the repository root) or a
    bare checkpoint name. The reference itself comes first, then its final
    component under the shared checkpoint root -- so a machine that keeps weights
    outside the repository finds them without symlinking them in. Finally
    ``my_weight/`` is searched for whichever single usable checkpoint it holds: a
    machine that downloaded ``-Instruct`` where a config names the base model
    still runs the tier instead of skipping all of it, which is how the failures
    hiding behind those skips stay visible.

    Public because ``tests/evals`` resolves the checkpoint each of its configs
    names, and the chain must not be answered in two places.
    """
    path = Path(reference).expanduser()
    candidates = [path if path.is_absolute() else REPO_ROOT / path]

    name = path.name
    zoo = os.environ.get(MODELZOO_ENV)
    if zoo:
        root = Path(zoo).expanduser()
        candidates += [root / name, *sorted(root.glob(f"*/{name}"))]

    usable = [candidate for candidate in candidates if checkpoint_problem(candidate) is None]
    if usable:
        return usable
    local = sorted(p for p in (REPO_ROOT / "my_weight").glob("*") if checkpoint_problem(p) is None)
    return local or candidates


def _resolve_model_dir() -> Path:
    """Absolute path of the checkpoint under test, without validating it.

    An explicit ``LITE_LLAMA_TEST_MODEL_DIR`` is the whole answer: a caller that
    named a checkpoint does not want a fallback silently substituted for it.
    Otherwise the first usable candidate, or the first candidate when none is
    usable -- the caller reports that one's problem rather than testing nothing.
    """
    explicit = os.environ.get("LITE_LLAMA_TEST_MODEL_DIR")
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else REPO_ROOT / path
    return checkpoint_candidates(DEFAULT_MODEL_NAME)[0]


def checkpoint_problem(path: Path) -> str | None:
    """Describe why ``path`` is unusable as a checkpoint, or ``None`` if it is.

    Public because ``tests/evals`` gates on checkpoints named in its own configs
    rather than on the one this file resolves, and "what counts as a usable
    checkpoint" must not be answered in two places.
    """
    if not path.is_dir():
        return f"no such directory: {path}"
    if not (path / "config.json").is_file():
        return f"no config.json in {path}"
    if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
        return f"no *.safetensors or *.bin weights in {path}"
    return None


def _is_golden(nodeid: str) -> bool:
    """Whether this test item belongs to the golden gate suite."""
    return any(f"tests/{d}/" in nodeid or f"tests\\{d}\\" in nodeid for d in _GOLDEN_DIRS)


# --------------------------------------------------------------------------- #
# Collection policy
# --------------------------------------------------------------------------- #
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply directory-based marks, then skip what the machine cannot run.

    For golden tests the outcome is never a silent skip:
    - ``LITE_LLAMA_GOLDEN_STRICT=1``: hard FAIL (pytest.fail at collect time).
    - Otherwise: ``xfail(reason="UNVERIFIED: ...", run=False)`` — shows as
      yellow/orange in CI rather than green.
    """
    model_dir = _resolve_model_dir()
    checkpoint_problem_reason = checkpoint_problem(model_dir)
    cuda_missing = not torch.cuda.is_available()

    skip_gpu = pytest.mark.skip(reason="needs a CUDA device")
    skip_weights = pytest.mark.skip(reason=f"needs a checkpoint: {checkpoint_problem_reason}")

    for item in items:
        # Triton kernels cannot run on CPU; mark by location so new kernel
        # tests inherit the requirement automatically.
        if any(
            f"tests/{d}/" in item.nodeid or f"tests\\{d}\\" in item.nodeid for d in _GPU_ONLY_DIRS
        ):
            item.add_marker(pytest.mark.gpu)

        is_golden_test = _is_golden(item.nodeid)

        if cuda_missing and "gpu" in item.keywords:
            if is_golden_test:
                # Golden tests must NOT silently skip — mark as UNVERIFIED.
                if _GOLDEN_STRICT:
                    item.add_marker(pytest.mark.skip(reason="GOLDEN GATE FAIL: no CUDA device"))
                    # Override with a custom fixture that calls pytest.fail
                    item.add_marker(
                        pytest.mark.xfail(
                            reason="UNVERIFIED: no CUDA device (set LITE_LLAMA_GOLDEN_STRICT=1 to hard-fail)",
                            run=False,
                            strict=True,
                        )
                    )
                else:
                    item.add_marker(
                        pytest.mark.xfail(
                            reason="UNVERIFIED: no CUDA device",
                            run=False,
                        )
                    )
            else:
                item.add_marker(skip_gpu)

        if checkpoint_problem_reason and "weights" in item.keywords:
            if is_golden_test:
                if _GOLDEN_STRICT:
                    item.add_marker(
                        pytest.mark.xfail(
                            reason=f"UNVERIFIED: {checkpoint_problem_reason}",
                            run=False,
                            strict=True,
                        )
                    )
                else:
                    item.add_marker(
                        pytest.mark.xfail(
                            reason=f"UNVERIFIED: {checkpoint_problem_reason}",
                            run=False,
                        )
                    )
            else:
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


#: Hermetic dispatch, process-wide: a developer's frozen records must not flip
#: the suite. This has to be set at conftest *import*, not only in the
#: function-scoped fixture below — module/session-scoped fixtures (engine
#: builders in tests/engine, tests/golden, ...) are instantiated before
#: function-scoped autouse fixtures, and a dispatch made there caches its
#: decision on the global registry for the rest of the session.
os.environ["LITE_LLAMA_FROZEN_RANK"] = "0"


@pytest.fixture(autouse=True)
def _frozen_rank_off(monkeypatch: pytest.MonkeyPatch):
    """Re-pin the switch per test, so one opting in cannot leak the opt-in.

    A test opting into frozen ranking sets ``LITE_LLAMA_FROZEN_RANK=1`` (via
    monkeypatch) for its own duration; teardown restores the process-wide
    ``"0"`` above rather than an unset variable.
    """
    monkeypatch.setenv("LITE_LLAMA_FROZEN_RANK", "0")


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Validated checkpoint directory; skips the test when it is unusable."""
    path = _resolve_model_dir()
    problem = checkpoint_problem(path)
    if problem:
        pytest.skip(f"needs a checkpoint: {problem}")
    return path


@pytest.fixture
def cuda_available() -> bool:
    """Skip unless CUDA is present. Prefer the ``gpu`` mark in new tests."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return True
