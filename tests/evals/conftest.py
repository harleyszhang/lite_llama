"""Turns ``configs/*.yaml`` into one test case each.

``pytest_generate_tests`` parametrises the correctness test over the
resolved config list, so adding a model to the suite means adding one
YAML file.

Usage:
    pytest tests/evals/
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).parent / "configs"

#: Runs when ``--config-list-file`` is not given: the cheapest config that still
#: exercises the whole path, so ``make test`` stays usable.
DEFAULT_CONFIG_LIST = "models-small.txt"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--config-list-file",
        default=DEFAULT_CONFIG_LIST,
        help=(
            "file listing the eval configs to run, one name per line; "
            f"resolved against {CONFIG_DIR} when relative"
        ),
    )


def _resolve_list_file(name: str) -> Path:
    """Find the list file next to the configs, in the CWD, or as given."""
    candidate = Path(name)
    if candidate.is_absolute():
        return candidate
    for base in (CONFIG_DIR, Path.cwd()):
        if (base / candidate).is_file():
            return base / candidate
    return candidate


def read_config_list(name: str) -> list[Path]:
    """Config paths named by the list file, skipping blanks and ``#`` comments.

    Raises:
        FileNotFoundError: The list file, or a config it names, does not exist.
            Silently dropping either would turn a typo into a green run that
            evaluated nothing.
    """
    list_file = _resolve_list_file(name)
    if not list_file.is_file():
        raise FileNotFoundError(f"config list not found: {list_file}")

    configs = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        path = list_file.parent / entry
        if not path.is_file():
            raise FileNotFoundError(f"{list_file.name} names a missing config: {path}")
        configs.append(path)
    return configs


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "config_filename" not in metafunc.fixturenames:
        return
    configs = read_config_list(metafunc.config.getoption("--config-list-file"))
    metafunc.parametrize("config_filename", configs, ids=[c.stem for c in configs])
