"""Accuracy tools: whole-model numeric comparison against a reference.

Re-exports :func:`find_first_divergent_layer` — the whole-model counterpart of
the single-layer harness, naming the first decoder layer whose output leaves
the numeric noise band — plus its report types.

Usage:
    from rapid_llm.tools.accuracy import find_first_divergent_layer
"""

from .divergence import (
    DEFAULT_PROMPT,
    DEFAULT_REL_THRESHOLD,
    DivergenceChecker,
    DivergenceReport,
    LayerDiff,
    LogitsDiff,
    SubmoduleDiff,
    find_first_divergent_layer,
)

__all__ = [
    "DEFAULT_PROMPT",
    "DEFAULT_REL_THRESHOLD",
    "DivergenceChecker",
    "DivergenceReport",
    "LayerDiff",
    "LogitsDiff",
    "SubmoduleDiff",
    "find_first_divergent_layer",
]
