"""Tests for the ``no-hardcoded-weight-paths`` pre-commit hook.

Rejected forms — absolute paths and bare directory prefixes — produce a
report naming file, line and path, while allowed forms pass. The hook
script is loaded and run against tmp files.

Usage:
    pytest tests/tools/test_check_hardcoded_paths.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

_HOOK = REPO_ROOT / "tools" / "pre_commit" / "check_hardcoded_paths.py"


def _load_hook():
    """Import the hook by path; it is a standalone script, not an installed module."""
    spec = importlib.util.spec_from_file_location("check_hardcoded_paths", _HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hook():
    return _load_hook()


def _check(hook, tmp_path: Path, source: str, name: str = "sample.py") -> list[str]:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return hook.check_file(path)


# --------------------------------------------------------------------------- #
# Must fire
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "source",
    [
        pytest.param('CKPT = "/home/foo/my_weight/Qwen2.5-0.5B"', id="home"),
        pytest.param('CKPT = "/Users/foo/weights"', id="macos-users"),
        pytest.param('CKPT = "/gemini/code/Llama-3.2-1B-Instruct/"', id="the-original-sin"),
        pytest.param('CKPT = "/mnt/nvme/ckpt"', id="mnt"),
        pytest.param('CKPT = "/data/shared/llm_weights/Qwen3-0.6B"', id="shared-weight-store"),
        pytest.param("CKPT = Path('/root/ckpt')", id="wrapped-in-Path"),
        pytest.param('CKPT = f"/home/{USER}/w"', id="f-string"),
        pytest.param('CKPT = "/home/" + user + "/w"', id="concatenated"),
    ],
)
def test_absolute_paths_are_rejected(hook, tmp_path: Path, source: str):
    assert _check(hook, tmp_path, source), f"should have been flagged: {source}"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('CKPT = os.path.join("/root", "ckpt")', id="os.path.join"),
        pytest.param('CKPT = "/data"', id="bare-data"),
        pytest.param('CKPT = "/home"', id="bare-home"),
    ],
)
def test_bare_prefix_without_a_trailing_segment_is_rejected(hook, tmp_path: Path, source: str):
    """The gap the original pattern had: it required a ``/`` after the prefix.

    ``os.path.join("/root", "ckpt")`` reaches the same hard-coded location by a
    different syntax, so letting it through defeated the hook for anyone who
    happened to build the path rather than write it.
    """
    assert _check(hook, tmp_path, source), f"should have been flagged: {source}"


def test_a_path_in_a_docstring_is_rejected(hook, tmp_path: Path):
    """Docstrings are not exempt, whatever an old comment in the hook claimed."""
    source = '"""Usage: gen = LLM(model="/home/foo/weights")"""\n'
    assert _check(hook, tmp_path, source)


def test_the_report_names_the_file_line_and_path(hook, tmp_path: Path):
    problems = _check(hook, tmp_path, '\n\nCKPT = "/home/foo/w"\n')
    assert len(problems) == 1
    assert "sample.py:3" in problems[0]
    assert "/home/foo/w" in problems[0]


# --------------------------------------------------------------------------- #
# Must not fire
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "source",
    [
        pytest.param('CKPT = "my_weight/Qwen2.5-0.5B"', id="relative-path"),
        pytest.param('CKPT = os.environ["RAPID_LLM_MODEL_DIR"]', id="from-env"),
        pytest.param('CKPT = "/usr/share/data"', id="system-prefix-not-personal"),
        pytest.param('CKPT = "/opt/models"', id="opt"),
        pytest.param('    # CKPT = "/home/foo/w"', id="commented-out"),
        pytest.param('NAME = "/homework/notes"', id="word-starting-with-a-prefix"),
        pytest.param('NAME = "/datasets/imagenet"', id="datasets-is-not-data"),
    ],
)
def test_acceptable_sources_pass(hook, tmp_path: Path, source: str):
    assert _check(hook, tmp_path, source) == [], f"false positive on: {source}"


@pytest.mark.parametrize("directory", ["docs", "tools/pre_commit"])
def test_exempt_directories_are_skipped(hook, tmp_path: Path, directory: str):
    """``docs/`` is illustrative, and the hook's own docstring quotes a bad path."""
    target = tmp_path / directory
    target.mkdir(parents=True)
    path = target / "sample.py"
    path.write_text('CKPT = "/home/foo/w"', encoding="utf-8")
    assert hook.check_file(path) == []


def test_the_hook_does_not_flag_itself(hook):
    """Its module docstring quotes ``/gemini/...`` on purpose, as the motivating example."""
    assert hook.check_file(_HOOK) == []


# --------------------------------------------------------------------------- #
# Exit codes -- what pre-commit actually reads
# --------------------------------------------------------------------------- #
def test_main_exits_zero_when_clean(hook, tmp_path: Path, monkeypatch):
    clean = tmp_path / "clean.py"
    clean.write_text('CKPT = "my_weight/Qwen2.5-0.5B"', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["check_hardcoded_paths.py", str(clean)])
    assert hook.main() == 0


def test_main_exits_nonzero_and_explains_the_fix(hook, tmp_path: Path, monkeypatch, capsys):
    dirty = tmp_path / "dirty.py"
    dirty.write_text('CKPT = "/home/foo/w"', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["check_hardcoded_paths.py", str(dirty)])

    assert hook.main() == 1
    stderr = capsys.readouterr().err
    assert "/home/foo/w" in stderr
    # A failing hook has to say what to do instead, or it just blocks the commit.
    assert "RAPID_LLM_MODEL_DIR" in stderr


def test_the_whole_tracked_tree_passes(hook):
    """The hook must be green on the repository as it stands, or it is noise."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    offenders = [p for name in tracked for p in hook.check_file(REPO_ROOT / name)]
    assert offenders == [], f"tracked sources now violate the hook: {offenders[:5]}"
