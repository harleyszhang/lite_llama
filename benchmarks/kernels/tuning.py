"""Shared pieces of the ``--tune`` path in the quantised-kernel benchmarks.

``bench_quant_gemm.py`` and ``bench_fused_moe.py`` search different config spaces
against different kernels, but they report the same thing about a search: which
config the heuristic picked, which one won, and how many candidates were thrown
out. That record lived in both files under two different field names, so it
lives here once.

The search loop itself stays in each benchmark: what makes a candidate
*rejectable* is kernel-specific knowledge (a tile wider than the scale group
reads one scale for k elements that do not share one), and folding both loops
into one parameterised function would hide that behind injected callables.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lite_llama.kernels.dispatcher.autotune import TuneKey

__all__ = ["TuneResult", "nbytes"]


def nbytes(*tensors: torch.Tensor) -> int:
    """Bytes these tensors occupy, for the scale/zero-point terms of a traffic formula."""
    return sum(t.numel() * t.element_size() for t in tensors)


@dataclass(frozen=True, slots=True)
class TuneResult:
    """The outcome of searching one :class:`TuneKey`, including a search that won nothing.

    Attributes:
        label: Row label — the scheme or case the search ran for.
        rejected: Candidates that failed to compile or disagreed with the output
            the correctness gate had already checked.
    """

    key: TuneKey
    label: str
    tokens: tuple[int, ...]
    baseline_config: dict[str, int]
    baseline_us: float
    best_config: dict[str, int]
    best_us: float
    rejected: int

    @property
    def gain(self) -> float:
        """Fraction of the heuristic's time the winner saves. Negative is a loss."""
        return (self.baseline_us - self.best_us) / self.baseline_us if self.baseline_us else 0.0

    @property
    def changed(self) -> bool:
        return self.best_config != self.baseline_config
