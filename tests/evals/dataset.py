"""Benchmark data acquisition: download once, cache on disk, read as JSONL.

Kept separate from the benchmarks themselves so the scoring logic stays a set of
pure functions over ``list[dict]`` and can be unit-tested without a network.

The cache lives outside the repository (``~/.cache/lite_llama/evals`` by
default) so that a checkout stays clean and several worktrees share one copy.
Both the location and the download host are overridable:

* ``LITE_LLAMA_EVAL_DATA_DIR`` — cache directory,
* ``LITE_LLAMA_EVAL_BASE_URL`` — base URL to fetch from, for mirrors or for an
  air-gapped machine serving the files locally.

A machine with no route to the host raises :class:`DatasetUnavailable`, which
the pytest layer turns into a skip rather than a failure: an offline CI box
should report "no dataset" and not "the model regressed".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Canonical GSM8K release (the ``grade-school-math`` repository OpenAI published
#: with the paper). The files are plain JSONL with ``question`` / ``answer``
#: keys, ``answer`` ending in ``#### <number>``.
DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
)

#: Files each benchmark needs, relative to the base URL.
GSM8K_FILES = {"train": "train.jsonl", "test": "test.jsonl"}

_DOWNLOAD_TIMEOUT_S = 60


class DatasetUnavailable(RuntimeError):
    """The benchmark data is neither cached nor reachable."""


def cache_dir() -> Path:
    """Directory holding downloaded benchmark files."""
    override = os.environ.get("LITE_LLAMA_EVAL_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "lite_llama" / "evals"


def _base_url() -> str:
    return os.environ.get("LITE_LLAMA_EVAL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def fetch(name: str, subdir: str) -> Path:
    """Return the local path of ``name``, downloading it into ``subdir`` if needed.

    Args:
        name: File name, appended to both the base URL and the cache directory.
        subdir: Per-benchmark folder inside the cache, so two benchmarks may ship
            files of the same name.

    Raises:
        DatasetUnavailable: The file is not cached and the download failed.
    """
    target = cache_dir() / subdir / name
    if target.is_file() and target.stat().st_size > 0:
        return target

    import requests

    url = f"{_base_url()}/{name}"
    try:
        response = requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_S)
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling first: an interrupted download must not leave a
        # truncated file that later runs would happily accept from the cache.
        staging = target.with_suffix(target.suffix + ".part")
        with open(staging, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        staging.replace(target)
    except Exception as exc:  # network, DNS, HTTP status, disk
        raise DatasetUnavailable(
            f"cannot obtain {name!r} from {url}: {type(exc).__name__}: {exc}\n"
            f"Pre-seed the cache at {target} or point LITE_LLAMA_EVAL_BASE_URL "
            f"at a reachable mirror."
        ) from exc

    return target


def read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file, ignoring ``#`` comment lines."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


def load_gsm8k() -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` GSM8K splits — 7473 and 1319 records."""
    return (
        read_jsonl(fetch(GSM8K_FILES["train"], "gsm8k")),
        read_jsonl(fetch(GSM8K_FILES["test"], "gsm8k")),
    )
