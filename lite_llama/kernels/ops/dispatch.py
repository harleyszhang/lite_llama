"""Deterministic dispatch: filter, rank, cache — and be able to say why.

``dispatch(op, ...)`` answers "which implementation runs here?" in four fixed
steps (ROADMAP foundation 2, pillar 5):

1. **Filter** — availability probe, capability window, dtype, scheme, hard
   shape constraints, layout tags, and the golden-accuracy gate. Every
   rejection is recorded with its reason, verbatim.
2. **Rank** — frozen perf measurements (``perf_key``, fastest first) beat
   shape preferences, which beat static priority; the spec name is the final
   tie-break so equal inputs always yield equal outputs.
3. **Cache** — the decision for a full :class:`DispatchKey` is computed once.
4. **Report** — ``explain`` renders the decision chain for humans;
   ``LITE_LLAMA_KERNEL_TRACE=1`` emits one JSON line per decision for tools.

An explicit ``backend=`` (or the ``LITE_LLAMA_*_BACKEND`` env vars) narrows the
field to that family and bypasses only the golden gate — the user asked for the
kernel by name, so accuracy evidence is their call, but a missing library or
an unsupported dtype is physics and still excludes it.

Usage:
    sel = dispatch("linear", dtype="bf16", scheme="w8a16_fp8", shape={"k": 4096})
    fn = sel.load()      # imports the implementation lazily
    print(sel.explain()) # why this one won
"""

from __future__ import annotations

import importlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from ...platform.interface import current_platform
from ...platform.spec import PlatformInfo, capabilities_match
from .registry import REGISTRY, OpRegistry, register  # noqa: F401 (re-export)
from .spec import KernelSpec

#: Global override honoured by every dispatch() call when no explicit backend=.
FORCE_BACKEND_ENV = "LITE_LLAMA_FORCE_BACKEND"

#: One JSON line per decision when set (op, backend, dtype, shape...).
TRACE_ENV = "LITE_LLAMA_KERNEL_TRACE"


def op_backend_env(op: str) -> str:
    """Env var name pinning one op's backend: ``attention.decode`` ->
    ``LITE_LLAMA_ATTENTION_DECODE_BACKEND``.

    Per-op is the granularity that matters in practice: a machine may want
    flashinfer for attention while linear stays on the native Triton GEMM, and
    one global switch cannot express that. Inherited from the v0.8 backend
    picker, which had these keys for its two hardcoded op families.
    """
    return f"LITE_LLAMA_{op.upper().replace('.', '_')}_BACKEND"


def _forced_backend(op: str, explicit: str | None) -> str | None:
    """Resolve the backend pin: argument, then per-op env, then global env.

    Narrowest wins — an argument is the call site's own decision, a per-op key
    is the operator's, and the global key is the run's.
    """
    return (
        explicit or os.environ.get(op_backend_env(op)) or os.environ.get(FORCE_BACKEND_ENV) or None
    )


# --------------------------------------------------------------------------- #
# Keys and results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DispatchKey:
    """Everything a decision depends on; equal keys must give equal answers.

    Attributes:
        op: Logical op id.
        dtype: Activation dtype label (``"bf16"``).
        scheme: Quantisation scheme label.
        shape: Symbolic dims as a sorted tuple, so the key is hashable.
        layout: Layout tags the call site can provide.
        platform: Snapshot of the hardware the decision is made against.
        forced_backend: Backend family pinned by ``backend=``/env, if any.
    """

    op: str
    dtype: str
    scheme: str
    shape: tuple[tuple[str, int], ...]
    layout: frozenset[str]
    platform: PlatformInfo
    forced_backend: str | None

    @property
    def shape_dict(self) -> dict[str, int]:
        return dict(self.shape)


@dataclass(frozen=True)
class Selected:
    """The outcome of one :func:`dispatch` call, with its full audit trail.

    Attributes:
        spec: The winning implementation's declaration.
        key: The dispatch key that produced this decision.
        rejections: name -> reason for every filtered-out candidate.
        runners_up: Feasible candidates that lost only on rank, best first —
            shown by ``explain`` so "who else could have run here" stays
            visible (ROADMAP module D: candidates, not just the winner).
    """

    spec: KernelSpec
    key: DispatchKey
    rejections: Mapping[str, str] = field(default_factory=dict)
    runners_up: tuple[KernelSpec, ...] = ()

    def load(self) -> Any:
        """Import and return the implementation (cached by target string)."""
        return resolve_target(self.spec.target)

    def explain(self) -> str:
        """Render the decision chain: who ran, who lost, and why."""
        lines = [
            f"op={self.key.op!r} dtype={self.key.dtype} scheme={self.key.scheme}"
            f" shape={self.key.shape_dict} platform={self.key.platform.device_type}"
        ]
        if self.key.forced_backend:
            lines.append(f"forced backend: {self.key.forced_backend}")
        for name, reason in sorted(self.rejections.items()):
            lines.append(f"  [{name}] excluded: {reason}")
        for loser in self.runners_up:
            lines.append(f"  [{loser.name}] feasible, ranked below ({_rank_text(loser, self.key)})")
        lines.append(f"  [{self.spec.name}] dispatched (rank={_rank_text(self.spec, self.key)})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Lazy resolution (the only place implementations get imported)
# --------------------------------------------------------------------------- #
@cache
def resolve_target(ref: str) -> Any:
    """Import ``"module.path:attr"`` once per process."""
    module_name, _, attr = ref.rpartition(":")
    try:
        return getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"cannot resolve kernel target {ref!r}: {e}") from e


@cache
def _probe_available(ref: str) -> tuple[bool, str]:
    """Run an availability probe once; failures count as "not installed".

    Returns (usable, detail) where detail feeds the explain line either way.
    """
    try:
        probe = resolve_target(ref)
        return bool(probe()), str(probe)
    except Exception as e:
        return False, f"probe raised {type(e).__name__}: {e}"


def dtype_label(dtype: Any) -> str:
    """Normalise a torch dtype object (or an already-labelled str) to its label."""
    if isinstance(dtype, str):
        return dtype
    # Called from model code, torch is always loaded by then; importing here
    # keeps the ops tier itself torch-free.
    import torch

    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.float8_e4m3fn: "fp8_e4m3",
        torch.uint8: "u8",
        torch.int8: "i8",
    }.get(dtype, str(dtype).removeprefix("torch."))


# --------------------------------------------------------------------------- #
# Perf lookup (frozen measurements; wired to backends.json in M3)
# --------------------------------------------------------------------------- #
PerfProvider = Callable[[KernelSpec, DispatchKey], "float | None"]


#: Pluggable so the benchmark tool can inject the frozen store without the
#: dispatch tier depending on any storage format. ``None`` = no measurement.
def _no_measurements(spec: KernelSpec, key: DispatchKey) -> float | None:
    return None


_perf_provider: PerfProvider = _no_measurements


def set_perf_provider(provider: PerfProvider) -> None:
    """Install the frozen-latency lookup used by the ranking step."""
    global _perf_provider
    _perf_provider = provider


def _perf_latency_ms(spec: KernelSpec, key: DispatchKey) -> float | None:
    try:
        return _perf_provider(spec, key)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# The decision itself
# --------------------------------------------------------------------------- #
def invalidate_cache(op: str | None = None) -> None:
    """Drop the global registry's cached decisions (all, or one op's)."""
    if op is None:
        REGISTRY._decisions.clear()
    else:
        REGISTRY.notify_change(op)


def _reject_reason(spec: KernelSpec, key: DispatchKey) -> str | None:
    """Why ``spec`` cannot serve this key, or ``None`` if it can.

    Order mirrors the plan's filter list; each reason is written for explain.
    Correctness gates (capability/dtype/scheme/shape/layout/available) hold
    even under a forced backend; only the golden gate yields to an explicit
    request, because evidence policy is the one thing the user may override.
    """
    if key.forced_backend and spec.backend != key.forced_backend:
        return f"backend {spec.backend!r} != forced {key.forced_backend!r}"
    if spec.available is not None:
        ok, detail = _probe_available(spec.available)
        if not ok:
            return f"library unavailable ({spec.available}; {detail})"
    if not capabilities_match(spec.capability, key.platform):
        return f"capability: needs {list(spec.capability) or 'any'}, platform is {key.platform.device_type}"
    if not spec.dtype_ok(key.dtype):
        return f"dtype {key.dtype!r} not in {list(spec.dtypes) or ['any']}"
    if not spec.scheme_ok(key.scheme):
        return f"scheme {key.scheme!r} not in {list(spec.schemes)}"
    dims = key.shape_dict
    if not spec.shape_ok(dims):
        failed = [str(c) for c in spec.shape.hard if not c.satisfied_by(dims)]
        return f"shape: fails {failed} (got {dims})"
    missing = spec.layout_missing(key.layout)
    if missing:
        return f"layout: call site cannot provide {sorted(missing)}"
    if not spec.golden.verified and not key.forced_backend:
        return "golden: not verified — excluded from default dispatch"
    return None


def _rank_key(spec: KernelSpec, key: DispatchKey) -> tuple:
    """Total order over feasible candidates; identical inputs -> identical order."""
    perf = _perf_latency_ms(spec, key)
    return (
        perf if perf is not None else math.inf,  # measured fastest first
        -spec.shape.preference_score(key.shape_dict),  # then shape fit
        -spec.priority,  # then static priority
        spec.name,  # final tie-break keeps the decision deterministic
    )


def _rank_text(spec: KernelSpec, key: DispatchKey) -> str:
    perf = _perf_latency_ms(spec, key)
    perf_txt = "unmeasured" if perf is None else f"{perf:.3f}ms"
    return (
        f"perf={perf_txt} prefer={spec.shape.preference_score(key.shape_dict)} pri={spec.priority}"
    )


def dispatch(
    op: str,
    *,
    dtype: Any,
    scheme: str = "unquantized",
    shape: Mapping[str, int] | None = None,
    layout: frozenset[str] = frozenset(),
    backend: str | None = None,
    platform_info: PlatformInfo | None = None,
    registry: OpRegistry = REGISTRY,
) -> Selected:
    """Pick the implementation for ``op`` under this key, deterministically.

    Args:
        op: Logical op id (``"linear"``, ``"attention.decode"``).
        dtype: Activation dtype (torch dtype or label string).
        scheme: Quantisation scheme label; the dispatch key's quantisation
            dimension, not a separate registry.
        shape: Symbolic dims the decision may depend on (``{"k": 4096}``).
        layout: Layout tags the call site can provide (``frozenset({"kv:paged"})``).
        backend: Pin the backend family; bypasses the golden gate but never
            the physical ones. ``LITE_LLAMA_<OP>_BACKEND`` pins one op and
            ``LITE_LLAMA_FORCE_BACKEND`` the whole run.
        platform_info: Hardware snapshot; defaults to the detected platform.
            Injectable so tests (and tools) run on imagined machines.
        registry: Table to dispatch over; tests use a private instance.

    Raises:
        LookupError: Unknown op, or nothing survives the filter (including a
            forced backend that is not usable here) — failing loud beats
            running a wrong kernel.
    """
    forced = _forced_backend(op, backend)
    info = platform_info if platform_info is not None else current_platform().detect()
    key = DispatchKey(
        op=op,
        dtype=dtype_label(dtype),
        scheme=scheme,
        shape=tuple(sorted((shape or {}).items())),
        layout=frozenset(layout),
        platform=info,
        forced_backend=forced,
    )
    cached = registry._decisions.get(key)
    if cached is not None:
        return cached

    specs = registry.implementations(op)
    if not specs:
        raise LookupError(f"unknown op {op!r}; registered: {', '.join(registry.ops())}")

    rejections: dict[str, str] = {}
    feasible: list[KernelSpec] = []
    for spec in specs:
        reason = _reject_reason(spec, key)
        if reason is None:
            feasible.append(spec)
        else:
            rejections[spec.name] = reason

    if not feasible:
        head = f"no usable implementation for {op!r} under dtype={key.dtype} scheme={key.scheme}"
        if forced:
            head += f" forced={forced!r}"
        raise LookupError(
            head + "\n" + "\n".join(f"  [{n}] {r}" for n, r in sorted(rejections.items()))
        )

    ranked = sorted(feasible, key=lambda s: _rank_key(s, key))
    chosen = ranked[0]
    decision = Selected(
        spec=chosen,
        key=key,
        rejections=rejections,
        runners_up=tuple(ranked[1:]),
    )
    registry._decisions[key] = decision
    _trace(decision)
    return decision


def explain(op: str, **key_kwargs: Any) -> str:
    """Decision chain for one dispatch() call; runs the dispatch if needed."""
    sel = dispatch(op, **key_kwargs)
    return sel.explain()


def _trace(decision: Selected) -> None:
    """One JSON line per decision when tracing is enabled."""
    if os.environ.get(TRACE_ENV, "") != "1":
        return
    from ...utils.logger import get_logger

    get_logger(__name__).info(
        json.dumps(
            {
                "op": decision.key.op,
                "kernel": decision.spec.name,
                "backend": decision.spec.backend,
                "dtype": decision.key.dtype,
                "scheme": decision.key.scheme,
                "shape": decision.key.shape_dict,
                "platform": decision.key.platform.device_type,
                "forced": decision.key.forced_backend,
            }
        )
    )
