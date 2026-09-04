"""Environment variable lookup with legacy-name compatibility.

The project was renamed from ``lite_llama`` to ``rapid_llm``; every
``RAPID_LLM_*`` variable used to be spelled ``LITE_LLAMA_*``. Reads prefer
the new name and fall back to the legacy one, warning once per legacy
variable so existing deployments keep working while scripts migrate.
"""

from __future__ import annotations

import os
import warnings

_LEGACY_PREFIX = "LITE_LLAMA_"
_NEW_PREFIX = "RAPID_LLM_"

#: Legacy variables already warned about in this process (warn once each).
_warned: set[str] = set()


def getenv(name: str, default: str | None = None) -> str | None:
    """Read *name* from the environment, falling back to its legacy spelling.

    ``RAPID_LLM_FOO`` wins; when unset, the legacy ``LITE_LLAMA_FOO`` is
    honoured and a :class:`DeprecationWarning` is emitted once per process.
    """
    value = os.environ.get(name)
    if value is not None:
        return value
    if name.startswith(_NEW_PREFIX):
        legacy = _LEGACY_PREFIX + name[len(_NEW_PREFIX) :]
        value = os.environ.get(legacy)
        if value is not None and legacy not in _warned:
            _warned.add(legacy)
            warnings.warn(
                f"Environment variable {legacy} is deprecated and will be removed; "
                f"rename it to {name}.",
                DeprecationWarning,
                stacklevel=2,
            )
    return value if value is not None else default
