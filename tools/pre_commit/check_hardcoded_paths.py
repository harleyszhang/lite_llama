#!/usr/bin/env python
"""Reject hard-coded absolute model/weight paths in tracked Python sources.

lite_llama historically shipped entry points with the author's own checkpoint
directory baked in — ``model_path = "/gemini/code/Llama-3.2-1B-Instruct/"`` and
friends — which makes the CLI unusable for everybody else. Checkpoint locations
must come from a CLI argument or an environment variable instead.

Only ``#`` comments are exempt, plus the directories in :data:`_EXEMPT_DIRS`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Absolute POSIX paths that look like a personal or machine-specific location.
# The trailing segment is optional so that a bare prefix is caught too —
# ``os.path.join("/root", "ckpt")`` is the same bug as ``"/root/ckpt"``. Requiring
# either a ``/`` or the closing quote right after the prefix keeps innocent words
# like ``"/homework"`` from matching.
_FORBIDDEN = re.compile(
    r"""["'](/(?:home|Users|root|gemini|mnt|data)(?:/[^"'\n]*)?)["']""",
)

# Directories whose contents are illustrative rather than executable library code.
_EXEMPT_DIRS = ("docs/", "tools/pre_commit/", "tests/tools/")


def _is_exempt(path: Path) -> bool:
    posix = path.as_posix()
    return any(posix.startswith(prefix) or f"/{prefix}" in posix for prefix in _EXEMPT_DIRS)


def check_file(path: Path) -> list[str]:
    if _is_exempt(path):
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []

    problems = []
    for lineno, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        # A ``#`` comment may show a sample path. A docstring deliberately may not:
        # "usage example" text is the most common route by which someone's personal
        # path gets copied back into working code.
        if stripped.startswith("#"):
            continue
        match = _FORBIDDEN.search(line)
        if match:
            problems.append(f"{path}:{lineno}: hard-coded absolute path {match.group(1)!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()

    problems: list[str] = []
    for file in args.files:
        problems.extend(check_file(file))

    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(
            "\nPass the checkpoint directory via a CLI flag or the LITE_LLAMA_MODEL_DIR "
            "environment variable instead of hard-coding it.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
