"""Persistent JSON config store for autotune results.

:class:`ConfigStore` reads and writes one JSON document per cache dir,
merging measured configs under their :class:`TuneKey` — simple enough
to inspect by hand, stable enough to commit.

Usage:
    store = ConfigStore(cache_dir)
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ....utils.env_compat import getenv
from .config_key import TuneKey

#: Schema version; bump when the JSON structure changes incompatibly.
_SCHEMA_VERSION = 1


def _default_cache_dir() -> Path:
    """Resolve the autotune cache dir, honouring the pre-rename location.

    An explicit ``RAPID_LLM_AUTOTUNE_DIR`` wins (legacy ``LITE_LLAMA_``
    spelling accepted); otherwise the new default is used unless only the
    legacy ``~/.cache/lite_llama/autotune`` exists, so measured configs
    survive the rename without a migration step.
    """
    explicit = getenv("RAPID_LLM_AUTOTUNE_DIR")
    if explicit:
        return Path(explicit)
    new = Path.home() / ".cache" / "rapid_llm" / "autotune"
    legacy = Path.home() / ".cache" / "lite_llama" / "autotune"
    if not new.exists() and legacy.exists():
        return legacy
    return new


class ConfigStore:
    """Read/write autotune configurations to disk as JSON.

    Args:
        cache_dir: Directory for the JSON files. Created on first write.
            Defaults to ``~/.cache/rapid_llm/autotune/`` or the path in
            ``RAPID_LLM_AUTOTUNE_DIR`` (falls back to the legacy
            ``~/.cache/lite_llama/autotune/`` when that is what exists).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self._loaded: dict[str, dict[TuneKey, dict]] = {}
        self._mutex = threading.RLock()

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: TuneKey) -> dict | None:
        """Return the tile config for *key*, or ``None`` on miss."""
        entry = self.get_entry(key)
        return entry["config"] if entry else None

    def get_entry(self, key: TuneKey) -> dict | None:
        """Return the full entry for *key* — config, latency, timestamp — or None.

        The frozen-rank provider needs the measured latency, not just the
        config payload, so the entry is addressable without going behind the
        store's back.
        """
        return self._ensure_loaded(key.op).get(key)

    def put(self, key: TuneKey, config: dict, latency_us: float) -> None:
        """Insert an entry without losing writes from another tuner process."""
        entry = {
            "gpu": key.gpu,
            "shape_bucket": key.shape_bucket,
            "dtype": key.dtype,
            "config": config,
            "latency_us": latency_us,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with self._write_lock(key.op):
            # A prior lookup may have cached an older snapshot. Reload while
            # holding the process lock, merge this key, then replace atomically.
            entries = self._read(key.op)
            entries[key] = entry
            self._loaded[key.op] = entries
            self._flush(key.op)

    def load_all(self, op: str) -> dict[TuneKey, dict]:
        """Return all entries for *op* (keyed by TuneKey)."""
        return dict(self._ensure_loaded(op))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _json_path(self, op: str) -> Path:
        return self._cache_dir / f"{op}.json"

    def _ensure_loaded(self, op: str) -> dict[TuneKey, dict]:
        """Lazy-load the op's JSON file into memory."""
        if op in self._loaded:
            return self._loaded[op]
        entries = self._read(op)
        self._loaded[op] = entries
        return entries

    def _read(self, op: str) -> dict[TuneKey, dict]:
        """Read one operation's current on-disk entries."""
        entries: dict[TuneKey, dict] = {}
        path = self._json_path(op)
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("entries", []):
                key = TuneKey(
                    gpu=entry["gpu"],
                    op=op,
                    shape_bucket=entry["shape_bucket"],
                    dtype=entry["dtype"],
                )
                entries[key] = entry
        return entries

    @contextmanager
    def _write_lock(self, op: str) -> Iterator[None]:
        """Serialise writers in this process and across tuner processes."""
        with self._mutex:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            lock_path = self._cache_dir / f".{op}.lock"
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _flush(self, op: str) -> None:
        """Serialise the op's entries to JSON (atomic write via rename)."""
        entries = self._loaded.get(op, {})
        data = {
            "version": _SCHEMA_VERSION,
            "entries": list(entries.values()),
        }
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        target = self._json_path(op)
        # Atomic: write to a temp file in the same directory, then rename.
        # fdopen takes ownership of the descriptor, so it is closed on every
        # exit path and a failed replace only has to clean up the temp file.
        fd, tmp = tempfile.mkstemp(dir=self._cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, indent=2, ensure_ascii=False))
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
