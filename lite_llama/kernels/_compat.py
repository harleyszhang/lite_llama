"""Optional-Triton compatibility: an import surface that survives a Triton-less install.

Triton only ships wheels for Linux+CUDA, so on macOS or a CPU-only host the
package is absent and ``import lite_llama`` would fail at the first kernel
module. That defeats two project goals: host-side tooling (registry probing,
explain, config, docs) should work anywhere, and a missing backend must never
be a hard failure. This module is the single place that knows whether Triton
exists; every kernel module imports ``triton`` / ``tl`` from here instead of
importing them directly.

* With Triton installed, ``triton`` and ``tl`` are the real modules.
* Without it, they are shims that tolerate module *import* (decorators,
  annotations, dtype tables) and raise a precise error only when a kernel is
  actually *launched* — which needs a GPU anyway.
"""

from __future__ import annotations

#: Whether the real Triton package is importable on this machine.
HAS_TRITON: bool

_LAUNCH_ERROR = (
    "Triton is not installed, so GPU kernels cannot launch. Triton ships "
    "Linux/CUDA wheels only; on macOS or a CPU-only host lite_llama can import "
    "and run its host-side logic but cannot execute a model. Install it with "
    "`pip install triton` on a Linux CUDA machine."
)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

    class _MissingKernel:
        """Stand-in for a ``@triton.jit`` function: any launch attempt raises."""

        def __init__(self, fn) -> None:
            self.__wrapped__ = fn
            self.__name__ = getattr(fn, "__name__", "triton_kernel")
            self.__doc__ = getattr(fn, "__doc__", None)

        def _raise(self) -> None:
            raise RuntimeError(f"kernel '{self.__name__}' cannot launch: {_LAUNCH_ERROR}")

        def __call__(self, *args, **kwargs) -> None:
            self._raise()

        def __getitem__(self, _grid):
            # ``kernel[grid](...)`` is the usual launch spelling.
            return lambda *args, **kwargs: self._raise()

        def run(self, *args, **kwargs) -> None:
            self._raise()

        def warmup(self, *args, **kwargs) -> None:
            self._raise()

    def _jit(fn=None, **_kwargs):
        # Both spellings appear in the tree: @triton.jit and @triton.jit().
        if fn is None:
            return _MissingKernel
        return _MissingKernel(fn)

    class _TLPlaceholder:
        """Benign sentinel for ``tl.<attr>`` used at import time (dtype tables,
        ``constexpr`` annotations). Computing with it means a kernel body is
        running, which the launch shim has already refused."""

        def __init__(self, name: str) -> None:
            self._name = name

        def __call__(self, *args, **kwargs) -> _TLPlaceholder:
            return self

        def __repr__(self) -> str:
            return f"<triton-less tl.{self._name}>"

    class _TLShim:
        # Referenced by kernel-signature annotations, which evaluate at def time.
        constexpr = _TLPlaceholder("constexpr")

        def __getattr__(self, name: str) -> _TLPlaceholder:
            return _TLPlaceholder(name)

    class _TritonShim:
        """The subset of the ``triton`` module surface the kernels touch."""

        language = _TLShim()
        jit = staticmethod(_jit)

        @staticmethod
        def cdiv(a: int, b: int) -> int:
            return -(-int(a) // int(b))

        @staticmethod
        def next_power_of_2(n: int) -> int:
            n = int(n)
            return 1 if n <= 1 else 1 << (n - 1).bit_length()

        class Config:
            """Constructor-compatible stand-in for autotune config lists."""

            def __init__(self, kwargs=None, num_warps=4, num_stages=2, **rest) -> None:
                self.kwargs = dict(kwargs or {}, **rest)
                self.num_warps = num_warps
                self.num_stages = num_stages

        @staticmethod
        def autotune(*_args, **_kwargs):
            return _MissingKernel

        @staticmethod
        def heuristics(*_args, **_kwargs):
            return _MissingKernel

        def __getattr__(self, name: str):
            raise RuntimeError(f"triton.{name} is unavailable: {_LAUNCH_ERROR}")

    triton = _TritonShim()
    tl = triton.language
