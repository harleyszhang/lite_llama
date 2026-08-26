"""TileOPs rows: bf16/fp16 dense GEMM from a TileLang-generated operator library.

TileOPs is the third contender for ``linear``, and the reason the ranking step
of dispatch has to be measured rather than argued: on the same bf16 shape,
native Triton, TileOPs and (in fp8) DeepGEMM are each fastest somewhere, so the
default must come from a frozen benchmark record (``perf_key``, M3.2) instead of
a hand-written priority.

Planned domain (row lands in M2.3): ``linear`` for bf16/fp16 via ``GemmFwdOp``.
Watch its semantics — ``GemmFwdOp()(a, b)`` computes ``a @ b.T``, which happens
to match how ``nn.Linear`` stores weights, so the row declares that layout
rather than transposing per call.

The toolchain window is the narrowest of the five: CUDA Toolkit 13.2, sm90, and
TileLang >=0.1.9,<0.2.0. TileOPs itself has no PyPI release yet, so the extra
carries only TileLang (the prerequisite that *is* on PyPI) and the recipe below
installs the library. When the local CUDA does not satisfy it, the backend stays
mechanically ready and ``available()``/``capability`` simply exclude it — that
fallback path is itself part of what M2.3 verifies.

Usage:
    from lite_llama.kernels.backends import tileops
    tileops.available()   # False until the source install is done
"""

from __future__ import annotations

from .probe import BackendInstall, library_present

INSTALL = BackendInstall(
    backend="tileops",
    module="tileops",
    homepage="https://github.com/tile-ai/TileOPs",
    requires="sm90 (Hopper), CUDA Toolkit 13.2, TileLang >=0.1.9,<0.2.0; ops auto-tune on first use",
    # The extra installs TileLang only; TileOPs is the clone below. Both are
    # named so a report never implies the extra alone is enough.
    extra="tileops",
    source_recipe=(
        "git clone https://github.com/tile-ai/TileOPs && "
        "cd TileOPs && pip install -e '.[dev]' -c constraints.txt"
    ),
)


def available() -> bool:
    """Whether TileOPs can serve a call here."""
    return library_present(INSTALL.module)
