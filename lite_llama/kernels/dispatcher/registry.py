"""Global KernelSpec registry: one table for every implementation of every op.

:class:`OpRegistry` indexes specs by op name; the module-level ``REGISTRY``
is the single instance every ops package registers into via
:func:`register` at import time.

Usage:
    from lite_llama.kernels.dispatcher import register; register(spec)
"""

from __future__ import annotations

import logging

from .spec import KernelSpec

logger = logging.getLogger(__name__)


class OpRegistry:
    """The process-wide table of :class:`KernelSpec` entries.

    Not thread-safe by design: registration happens during import, before any
    engine thread exists. The decision cache lives on the instance so a
    private test registry can never be poisoned by (or poison) the global one.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, KernelSpec] = {}
        self._by_op: dict[str, list[KernelSpec]] = {}
        # Populated by dispatch(); invalidated by notify_change on every
        # registration. Keyed by dispatch.DispatchKey, typed loosely here to
        # keep this module independent of the dispatch import.
        self._decisions: dict = {}

    # ------------------------------------------------------------------ #
    # Table maintenance
    # ------------------------------------------------------------------ #
    def register(self, spec: KernelSpec) -> KernelSpec:
        """Validate and insert ``spec``; identical re-registration is a no-op.

        Raises:
            ValueError: malformed spec (see :meth:`KernelSpec.validate`) or a
                different spec already owns this name.
        """
        spec.validate()
        existing = self._by_name.get(spec.name)
        if existing is not None:
            if existing == spec:
                return existing  # idempotent: module re-import must not be an error
            raise ValueError(f"kernel name {spec.name!r} already registered with a different spec")
        self._by_name[spec.name] = spec
        self._by_op.setdefault(spec.op, []).append(spec)
        logger.debug("registered kernel %s for op %s", spec.name, spec.op)
        self.notify_change(spec.op)
        return spec

    def notify_change(self, op: str) -> None:
        """Drop cached decisions for ``op`` after its candidate set changed."""
        for key in [k for k in self._decisions if k.op == op]:
            del self._decisions[key]

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #
    def implementations(self, op: str) -> tuple[KernelSpec, ...]:
        """All specs registered for ``op``, registration order preserved."""
        return tuple(self._by_op.get(op, ()))

    def spec(self, name: str) -> KernelSpec:
        """Look up one implementation by its unique name."""
        try:
            return self._by_name[name]
        except KeyError:
            raise LookupError(f"no kernel registered as {name!r}") from None

    def ops(self) -> tuple[str, ...]:
        """Logical op ids that have at least one implementation."""
        return tuple(self._by_op)

    def specs(self) -> tuple[KernelSpec, ...]:
        """Every registered spec (``explain``/tooling surface)."""
        return tuple(self._by_name.values())

    def decisions(self) -> tuple:
        """Decisions dispatch() has taken so far, in first-use order.

        The read side of the cache :meth:`notify_change` invalidates. A tool that
        drove one layer's forward pass can then report which implementations that
        layer actually reached for, rather than re-deriving the ranking and hoping
        it asks with the same key — the shape and dtype are part of the key, so
        guessing them is how a report ends up naming a kernel that never ran.
        """
        return tuple(self._decisions.values())

    def native_floor(self, op: str) -> KernelSpec | None:
        """The first native row for ``op`` — the never-failing baseline."""
        return next((s for s in self._by_op.get(op, ()) if s.backend == "native"), None)


#: The singleton every impl module registers into.
REGISTRY = OpRegistry()


def register(spec: KernelSpec) -> KernelSpec:
    """Module-level shorthand for :meth:`OpRegistry.register`."""
    return REGISTRY.register(spec)
