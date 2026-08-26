"""Can this machine run an external backend — and if not, how would it get it?

Every non-native KernelSpec must name an ``available`` probe (enforced by
:meth:`~lite_llama.kernels.ops.spec.KernelSpec.validate`), and
:func:`~lite_llama.kernels.ops.dispatch` calls it before considering the row.
That is what makes a missing wheel a *ranking* event instead of a crash: the
row drops out, ``explain`` says why, and the native Triton floor serves the
call. This module holds both halves of that answer — :func:`library_present`,
the probe those rows point at, and :class:`BackendInstall`, the recipe to
report when a backend is absent but wanted.

Core design: the probe really imports the library rather than asking
``importlib.util.find_spec``. These backends are compiled extensions built
against one CUDA (DeepGEMM, FlashMLA, DeepEP) or JIT toolchains that only fail
on first touch (TileOPs/TileLang), so "the package directory exists" and "the
package loads on this driver" are different questions and dispatch only cares
about the second. The cost is bounded: at most one import per module name per
process, paid on the first dispatch that could have used the backend, and a
box that never dispatches that op never pays it at all.

The install recipes are data, not prose, because two very different situations
hide behind "install the backend": FlashInfer is a wheel, while DeepGEMM,
FlashMLA and DeepEP are recursive clones built against the local CUDA (DeepEP
additionally needs NVSHMEM) and cannot be expressed as a pip requirement at
all. :attr:`BackendInstall.extra` is set only when ``pip install`` can really
do the job, so a report never promises an install path that would fail.

Usage:
    library_present("flashinfer")     # -> False on a box without the wheel
    for install, present in survey():  # doctor line per external backend
        print(install.backend, present, install.how_to_get_it())
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from functools import cache

logger = logging.getLogger(__name__)

#: The external backends of v0.9, in the order their milestones land them.
#: A name here must be a module in this package declaring ``INSTALL`` and
#: ``available()``; :func:`survey` walks exactly this tuple.
EXTERNAL_BACKENDS = ("flashinfer", "deepgemm", "tileops", "flashmla", "deepep")


@cache
def library_present(module: str) -> bool:
    """Whether ``module`` actually imports in this process.

    Args:
        module: Top-level import name of the third-party library
            (``"deep_gemm"``, not the distribution name ``"deepgemm"``).

    Returns:
        True when the import succeeded. Cached, so a machine without the
        library pays the failed import once.
    """
    try:
        importlib.import_module(module)
    except Exception as exc:
        # Deliberately broad. A backend built against the wrong CUDA raises
        # OSError from the dynamic loader, a JIT backend can raise RuntimeError
        # while probing the driver, and an interrupted source install leaves
        # partial modules raising AttributeError. All of them mean the same
        # thing to dispatch — this backend cannot serve a call here — and none
        # of them may propagate into a forward pass.
        logger.debug("backend library %s unavailable: %s", module, exc)
        return False
    return True


@dataclass(frozen=True)
class BackendInstall:
    """How one external backend gets onto a machine, as data.

    Attributes:
        backend: Backend family name, matching the ``backend`` field of its
            KernelSpec rows and this module's file name.
        module: Top-level import name :func:`library_present` probes.
        homepage: Upstream project, for the report line.
        requires: Hardware/toolchain window in one phrase (``"sm90+"``), for
            humans; the machine-checked version is each row's ``capability``.
        extra: This repo's optional-dependency name, set **only** when
            ``pip install lite-llama[<extra>]`` genuinely installs the
            backend, or its pip-resolvable prerequisites when the backend
            itself is a source build (then ``source_recipe`` says so).
        source_recipe: Shell commands for a source install, verbatim from
            upstream, when pip cannot express it.
    """

    backend: str
    module: str
    homepage: str
    requires: str
    extra: str | None = None
    source_recipe: str = ""

    def __post_init__(self) -> None:
        if not self.extra and not self.source_recipe:
            # A backend nobody can install is a row that can never be chosen.
            raise ValueError(f"{self.backend!r}: declare an extra, a source recipe, or both")

    def how_to_get_it(self) -> str:
        """One line for a report: the pip extra, the source recipe, or both."""
        parts = []
        if self.extra:
            parts.append(f"pip install lite-llama[{self.extra}]")
        if self.source_recipe:
            parts.append(self.source_recipe)
        return " ; ".join(parts)


def survey() -> tuple[tuple[BackendInstall, bool], ...]:
    """Probe every external backend, in :data:`EXTERNAL_BACKENDS` order.

    Each backend module is imported for its metadata — cheap, since those
    modules are pure data — and then asked its own ``available()``, which may
    check more than the import (a multi-GPU backend also needs a process
    group). Absence is never an error here: this is the function a doctor
    command calls precisely to report it.

    Returns:
        ``(install, present)`` pairs.
    """
    out = []
    for name in EXTERNAL_BACKENDS:
        module = importlib.import_module(f".{name}", __package__)
        out.append((module.INSTALL, bool(module.available())))
    return tuple(out)
