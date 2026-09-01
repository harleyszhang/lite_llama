"""Persistent JSON config store for autotune results.

:class:`ConfigStore` reads and writes one JSON document per cache dir,
merging measured configs under their :class:`TuneKey` — simple enough
to inspect by hand, stable enough to commit.

Usage:
    store = ConfigStore(cache_dir)
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .config_key import TuneKey

#: Schema version; bump when the JSON structure changes incompatibly.
_SCHEMA_VERSION = 1

#: Default cache directory (XDG-compatible fallback).
_DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "LITE_LLAMA_AUTOTUNE_DIR",
        Path.home() / ".cache" / "lite_llama" / "autotune",
    )
)


class ConfigStore:
    """Read/write autotune configurations to disk as JSON.

    Args:
        cache_dir: Directory for the JSON files. Created on first write.
            Defaults to ``~/.cache/lite_llama/autotune/`` or the path in
            ``LITE_LLAMA_AUTOTUNE_DIR``.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        # In-memory cache: op -> {TuneKey -> entry_dict}
        self._loaded: dict[str, dict[TuneKey, dict]] = {}

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
        """Insert or overwrite an entry and flush to disk."""
        entries = self._ensure_loaded(key.op)
        entries[key] = {
            "gpu": key.gpu,
            "shape_bucket": key.shape_bucket,
            "dtype": key.dtype,
            "config": config,
            "latency_us": latency_us,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        }
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
        self._loaded[op] = entries
        return entries

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
        fd, tmp = tempfile.mkstemp(dir=self._cache_dir, suffix=".tmp")
        try:
            os.write(fd, json.dumps(data, indent=2, ensure_ascii=False).encode())
            os.close(fd)
            os.replace(tmp, target)
        except BaseException:
            os.close(fd) if not os.get_inheritable(fd) else None
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
