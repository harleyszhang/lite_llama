"""Backend registry: probe availability, rank by priority, select, explain.

Aligned with vLLM's kernel selection pattern where each backend declares
is_supported() + can_implement() and the framework picks the best match.

Environment variable overrides:
    LITE_LLAMA_LINEAR_BACKEND=triton    # force linear backend
    LITE_LLAMA_ATTENTION_BACKEND=triton # force attention backend

Usage:
    backend = select_backend("linear", dtype="fp16")
    print(explain_selection("linear"))
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ...utils.logger import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class Backend:
    """One registered kernel backend.

    Attributes:
        name: Human-readable id (e.g. "triton", "torch", "cutlass").
        op: Operation type ("linear", "attention", "moe").
        priority: Higher wins when multiple are available.
        probe: Returns True if usable on this machine.
        reason: Why this backend needs specific hardware/libs.
    """

    name: str
    op: str
    priority: int
    probe: Callable[[], bool]
    reason: str = ""


# --------------------------------------------------------------------------- #
# Probe functions
# --------------------------------------------------------------------------- #
def _probe_triton() -> bool:
    """Triton available (Linux + CUDA)."""
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


def _probe_torch() -> bool:
    """PyTorch always available as fallback."""
    return True


def _probe_fp8_native() -> bool:
    """Native fp8 tensor cores (sm89+)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 9 or (major == 8 and minor >= 9)


def _probe_cuda_graph() -> bool:
    """CUDA graphs require CUDA."""
    return torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class BackendRegistry:
    """Collects backends and selects the best available one per op."""

    def __init__(self) -> None:
        self._backends: list[Backend] = []
        self._cache: dict[str, Backend | None] = {}
        self._explanations: dict[str, str] = {}

    def register(self, backend: Backend) -> None:
        self._backends.append(backend)
        self._cache.pop(backend.op, None)

    def select(self, op: str, **kwargs: Any) -> Backend | None:
        """Return highest-priority available backend for *op*."""
        if op in self._cache:
            return self._cache[op]

        env_key = f"LITE_LLAMA_{op.upper()}_BACKEND"
        forced = os.environ.get(env_key)

        candidates = sorted(
            [b for b in self._backends if b.op == op],
            key=lambda b: b.priority, reverse=True,
        )

        selected: Backend | None = None
        lines: list[str] = []

        for b in candidates:
            if forced and b.name != forced:
                lines.append(f"  [{b.name}] skipped (env={forced})")
                continue
            ok = b.probe()
            lines.append(f"  [{b.name}] pri={b.priority} {'OK' if ok else 'N/A'} ({b.reason})")
            if ok and selected is None:
                selected = b

        if forced and selected is None:
            for b in candidates:
                if b.probe():
                    selected = b
                    lines.append(f"  WARN: '{forced}' unavailable, fallback -> '{b.name}'")
                    break

        self._cache[op] = selected
        self._explanations[op] = (
            f"Backend '{op}' selection:\n" + "\n".join(lines)
            + f"\n  -> {selected.name if selected else 'NONE'}"
        )
        if selected:
            _log.info("Backend[%s] = %s (pri=%d)", op, selected.name, selected.priority)
        return selected

    def explain(self, op: str) -> str:
        if op not in self._explanations:
            self.select(op)
        return self._explanations.get(op, f"No backends for '{op}'")


# --------------------------------------------------------------------------- #
# Singleton + defaults
# --------------------------------------------------------------------------- #
_REGISTRY: BackendRegistry | None = None


def get_registry() -> BackendRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = BackendRegistry()
        _register_defaults(_REGISTRY)
    return _REGISTRY


def _register_defaults(r: BackendRegistry) -> None:
    """Register built-in backends (mirrors vLLM kernel candidates)."""
    # Linear
    r.register(Backend("triton_quant", "linear", 100, _probe_triton,
                       "Triton w8a16/w4a16/w8a8/fp8 quantised GEMM"))
    r.register(Backend("triton_fp16", "linear", 90, _probe_triton,
                       "Triton fp16 GEMM (for unquantised)"))
    r.register(Backend("torch_linear", "linear", 10, _probe_torch,
                       "F.linear fallback (always available)"))
    r.register(Backend("fp8_native", "linear", 110, _probe_fp8_native,
                       "Native fp8 tensor cores (sm89+)"))

    # Attention
    r.register(Backend("triton_flash_v2", "attention", 100, _probe_triton,
                       "Triton FlashAttention-2 varlen + FlashDecoding"))
    r.register(Backend("torch_sdpa", "attention", 30, _probe_torch,
                       "torch.nn.functional.scaled_dot_product_attention"))

    # Overlap
    r.register(Backend("cuda_stream", "overlap", 100, _probe_cuda_graph,
                       "Multi-stream compute/comm overlap"))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def select_backend(op: str, **kwargs: Any) -> Backend | None:
    """Select best backend for an op type."""
    return get_registry().select(op, **kwargs)


def explain_selection(op: str = "linear") -> str:
    """Human-readable explanation of backend selection."""
    return get_registry().explain(op)
