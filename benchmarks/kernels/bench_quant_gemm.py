"""Microbenchmark the ``linear`` op across every registered quantisation scheme.

The projection GEMM is where a quantised checkpoint either pays off or does not.
On this H100 the measured answer is blunter than the theory, so read the headline
first and the regimes second:

**cuBLAS bf16 is fastest in 44 of the 48 cells**, and the four exceptions are the
result. They are all ``qwen3-4b/gate_up`` (N=19456, K=2560 — 100 MB in bf16, the
largest weight here) at ``M <= 128``, where int8 W8A8 runs 1.40x / 1.38x / 1.33x /
1.22x at M=1/8/32/128 and fp8 W8A8 1.35x / 1.33x / 1.29x.

That is not an artefact of one shape, it is the rule: **quantisation wins exactly
where bf16 is genuinely bandwidth-bound, and nowhere else.** At M=1 the bf16 row's
share of peak HBM ranges from 10.4% (``qwen3-30b-a3b/down``, a 3 MB weight) to 60.9%
(``qwen3-4b/gate_up``), and the wins sit at the top of that range. Below roughly 50%
the kernel is not waiting on memory, so removing weight bytes removes nothing and the
dequant is pure addition. Use the bf16 ``%bw`` column to predict whether a new
projection will pay, not the compression ratio.

On a mid-sized projection (``qwen3-4b/qkv``, N=6144, K=2560) the margins at ``m=1``
are 21.7 us for bf16 against 22.2 for int4, 22.7 for int8, 24.0 for fp8 W8A8, 28.2
for fp8 W8A16 and 49.0 for NVFP4; at ``m=2048`` the same column reads 89.1 against
103.8 / 166.1 / 334.8 / 370.0 / 755.3. So for most of this table the honest framing
is *what it costs to shrink the checkpoint* — at decode width, for three of the five
formats, close to nothing.

Where int4 reaches parity at ``m=1``, that is two different limits meeting rather
than one shared one, and reading it as "int4 is as fast as bf16" gets the next shape
wrong: bf16 streams 31 MB at 43% of peak and is bandwidth-bound, int4 streams 7.9 MB
at 11.9% and is unpack-bound, and its unpack happens to take as long as bf16's
stream.

The two regimes then explain how the *gap* moves, not who wins:

``M = 1..32`` (decode)
    One activation row against the whole weight matrix, ~2 FLOP/byte against an
    H100 ridge of ~295. Nothing here is compute-limited in principle, and the
    quantised rows' shortfall is dequant throughput: on ``qkv`` 1.02x for ``w4a16``,
    1.05x for ``w8a8_int8``, 2.26x for ``nvfp4``. This is also the only regime where
    a quantised row wins outright, and only on the widest weight.
``M >= 512`` (prefill)
    Intensity clears the ridge and cuBLAS starts using the tensor cores properly
    (723 TFLOP/s at ``m=2048``, 73% of peak), while the Triton rows stay pinned by
    their unpack loops. Every gap widens: 1.17x for ``w8a8_int8`` up to 8.5x for
    ``nvfp4``, and no quantised row wins any cell. This is the regime where an fp8
    W8A8 row *could* win on compute and does not (39% of peak against 73%).

What each row measures:

``native/linear_torch``
    ``F.linear`` in bf16 — cuBLAS, the floor to beat and the reference every
    quantised row is checked against.
``native/linear_w8a16``
    fp8-e4m3 weight, bf16 activation. Half the weight bytes, dequantised in the
    inner loop; no MMA change.
``native/linear_w8a8_fp8``
    fp8-e4m3 on *both* sides, feeding the sm89+ native fp8 MMA. The row that can
    win on compute — and the one that pays a per-token activation quantise pass
    the work formula deliberately does not credit (see :func:`linear_work`).
``native/linear_w8a8_int8``
    SmoothQuant: int8 weight, per-token int8 activation quantised in-kernel.
``native/linear_w4a16``
    AWQ/GPTQ packed int4, bf16 activation. A quarter of the weight bytes, and
    the most unpacking work per byte — yet at decode width it lands within 2% of
    cuBLAS, which it did not before ``--tune`` ran. See :func:`tune`: the
    heuristic was using ``GROUP_M=1`` below 32 rows, and fixing that took 20-44%
    off all eight decode keys. This is the only row in the table whose kernel
    consults the autotune store, so it is the only one a tuning run can improve.
``native/linear_nvfp4``
    NVFP4: e2m1 weight, an fp8-e4m3 scale every 16 elements, one fp32 scale per
    tensor. 4.5 bits per weight, so 3.56x fewer weight bytes than bf16 — and a
    bf16 MMA regardless, because sm90 has no fp4 tensor core. This row is
    **slower than bf16 in all 48 cells**, including the four where int8 and fp8
    W8A8 win, and that is the finding, not a bug: the e2m1 unpack costs ~10
    integer ops per weight element, which is an order of magnitude more time than
    the bytes it saves. Read it as the price of the footprint, and see
    :mod:`lite_llama.kernels.ops.quantization.nvfp4`.

Usage:
    python benchmarks/kernels/bench_quant_gemm.py
    python benchmarks/kernels/bench_quant_gemm.py --json out.json
    python benchmarks/kernels/bench_quant_gemm.py --model-dir /path/to/Qwen3-4B
    python benchmarks/kernels/bench_quant_gemm.py --tune [--dry-run]

The numbers quoted above were recorded with ``LITE_LLAMA_AUTOTUNE=0``, which is
what a user without a tuning cache gets. With the cache from ``--tune`` the int4
row is a further 13-25% faster at ``m>=512``; nothing else in the table moves,
because nothing else reads the store.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from microbench import Row, Work, bench, device_peaks, metadata, report, verify
from tuning import TuneResult, nbytes

# Importing the facade registers every spec row, so dispatch() below finds them.
import lite_llama.kernels  # registers the spec rows as a side effect
import lite_llama.kernels.dispatcher.autotune as autotune_module
from lite_llama.kernels.dispatcher import dispatch
from lite_llama.kernels.dispatcher.autotune import ConfigStore, TuneKey, bucket_m
from lite_llama.kernels.dispatcher.autotune import reset as autotune_reset
from lite_llama.kernels.ops.gemm.linear import (
    linear_nvfp4,
    linear_torch,
    linear_w4a16,
    linear_w8a8_fp8,
    linear_w8a8_int8,
    linear_w8a16,
)
from lite_llama.kernels.ops.quantization import NVFP4_BLOCK, quantize_nvfp4_blockwise
from lite_llama.kernels.ops.quantization.w4a16 import launch_config as w4a16_launch_config
from lite_llama.modules.quantization.utils import (
    quantize_fp8_per_channel,
    quantize_int4_groupwise,
    quantize_int8_per_channel,
)

#: Token counts spanning the whole serving range: 1 is a single-sequence decode
#: step, 2048 a prefill tile. The regime boundary sits between 32 and 512, which
#: is why both sides are sampled rather than just the ends.
TOKENS: tuple[int, ...] = (1, 8, 32, 128, 512, 2048)

#: Weight-only int4 group size, matching what AWQ/GPTQ checkpoints ship.
INT4_GROUP_SIZE = 128

#: fp8 weight-scale block for the weight-only row. 128x128 is what a DeepSeek-V3
#: style fp8 checkpoint stores and what ``Fp8Config`` defaults to.
FP8_BLOCK = 128

#: The 8 magnitudes an e2m1 nibble can name, indexed by its low 3 bits. Spelled
#: out here so the NVFP4 reference decodes the format independently of the kernel
#: under test. ``.view(torch.float4_e2m1fn_x2)`` is not an option: torch 2.13
#: accepts the view but device-asserts on the widening cast.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# --------------------------------------------------------------------------- #
# Model geometry — the shapes that actually run
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Geometry:
    """The four projection shapes of one transformer block.

    Named after the projections rather than by index because their aspect ratios
    differ by 4x, and a kernel that tiles well for ``gate_up`` (tall, N >> K) can
    tile badly for ``down`` (wide, K >> N).
    """

    label: str
    hidden: int
    intermediate: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    def projections(self) -> list[tuple[str, int, int]]:
        """``(name, N, K)`` for qkv, o, gate_up and down."""
        q = self.num_heads * self.head_dim
        kv = self.num_kv_heads * self.head_dim
        return [
            ("qkv", q + 2 * kv, self.hidden),
            ("o", self.hidden, q),
            ("gate_up", 2 * self.intermediate, self.hidden),
            ("down", self.hidden, self.intermediate),
        ]


#: Geometries used when no ``--model-dir`` is given. Transcribed from the two
#: checkpoints this round validates end to end, so the kernel table and the e2e
#: matrix are measuring the same shapes. The MoE entry uses
#: ``moe_intermediate_size`` for its expert projections: that is what
#: ``fused_moe`` runs per expert, and it is 8x narrower than the dense FFN,
#: which moves those rows into a different tiling regime.
BUILTIN_GEOMETRIES: tuple[Geometry, ...] = (
    Geometry(
        "qwen3-4b", hidden=2560, intermediate=9728, num_heads=32, num_kv_heads=8, head_dim=128
    ),
    Geometry(
        "qwen3-30b-a3b", hidden=2048, intermediate=768, num_heads=32, num_kv_heads=4, head_dim=128
    ),
)


def geometry_from_config(model_dir: str) -> Geometry:
    """Read one :class:`Geometry` out of a HF ``config.json``.

    Prefers ``moe_intermediate_size`` when present: on a MoE checkpoint the
    per-expert width is the shape the quantised GEMM actually sees.
    """
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    return Geometry(
        label=Path(model_dir).name,
        hidden=hidden,
        intermediate=cfg.get("moe_intermediate_size") or cfg["intermediate_size"],
        num_heads=heads,
        num_kv_heads=cfg.get("num_key_value_heads", heads),
        head_dim=cfg.get("head_dim") or hidden // heads,
    )


# --------------------------------------------------------------------------- #
# Schemes — one entry per registered spec row
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scheme:
    """One quantisation scheme, from dispatch key to callable to work formula.

    Attributes:
        key: The ``scheme`` dispatch key; ``impl`` must be what dispatch picks
            for it, which :func:`show_dispatch` asserts.
        impl: ``KernelSpec.name`` of the row this scheme routes to.
        build: ``(w_bf16) -> (call, ref_weight, extra_bytes)`` where ``call``
            takes the activation and runs the kernel, ``ref_weight`` is the
            dequantised weight a torch reference should use, and ``extra_bytes``
            counts the scale/zero-point tensors.
        weight_bits: Stored bits per weight element, for the traffic formula.
        rtol: Relative tolerance, matching ``tests/kernels/`` for this kernel.
        atol: Absolute tolerance, likewise.
    """

    key: str
    impl: str
    build: Callable[
        [torch.Tensor], tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, int]
    ]
    weight_bits: int
    rtol: float
    atol: float




def _build_bf16(w: torch.Tensor):
    wb = w.to(torch.bfloat16)
    return (lambda x: linear_torch(x, wb)), wb.float(), 0


def _build_w8a16_fp8(w: torch.Tensor):
    """fp8-e4m3 weight with 128x128 block scales, bf16 activation."""
    n, k = w.shape
    qw = w.to(torch.float8_e4m3fn).view(torch.uint8)
    nb, kb = -(-n // FP8_BLOCK), -(-k // FP8_BLOCK)
    scale = torch.rand(nb, kb, device=w.device, dtype=torch.float32) + 0.5

    def call(x: torch.Tensor) -> torch.Tensor:
        return linear_w8a16(x, qw, weight_scale=scale, group_n=FP8_BLOCK, group_k=FP8_BLOCK)

    expanded = scale.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)[:n, :k]
    return call, qw.view(torch.float8_e4m3fn).float() * expanded, nbytes(scale)


def _build_w8a8_fp8(w: torch.Tensor):
    """fp8-e4m3 on both sides; the kernel quantises the activation itself."""
    qw, scale = quantize_fp8_per_channel(w)
    k = w.shape[1]

    def call(x: torch.Tensor) -> torch.Tensor:
        return linear_w8a8_fp8(x, qw, weight_scale=scale, group_n=1, group_k=k)

    return call, qw.view(torch.float8_e4m3fn).float() * scale, nbytes(scale)


def _build_w8a8_int8(w: torch.Tensor):
    """SmoothQuant: int8 weight, per-token int8 activation quantised in-kernel."""
    qw, scale = quantize_int8_per_channel(w)
    flat = scale.reshape(-1)

    def call(x: torch.Tensor) -> torch.Tensor:
        return linear_w8a8_int8(x, qw, weight_scale=flat)

    return call, qw.float() * scale, nbytes(scale)


def _build_w4a16(w: torch.Tensor):
    """AWQ/GPTQ packed int4 with group scales and zero points, bf16 activation."""
    qw, scale, zero = quantize_int4_groupwise(w, INT4_GROUP_SIZE)
    n, k = w.shape

    def call(x: torch.Tensor) -> torch.Tensor:
        return linear_w4a16(x, qw, weight_scale=scale, weight_zeros=zero, group_k=INT4_GROUP_SIZE)

    # Reference dequant: unpack 8 nibbles per int32 word, then (q - zero) * scale.
    shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
    nibbles = ((qw.unsqueeze(-1) >> shifts) & 0xF).reshape(n, k).float()
    groups = nibbles.reshape(n, k // INT4_GROUP_SIZE, INT4_GROUP_SIZE)
    deq = (groups - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
    return call, deq.reshape(n, k), nbytes(scale, zero)


def _build_nvfp4(w: torch.Tensor):
    """NVFP4: e2m1 nibbles, an e4m3 scale per 16 k-elements, one fp32 per tensor."""
    packed, block_scale, global_scale = quantize_nvfp4_blockwise(w)

    def call(x: torch.Tensor) -> torch.Tensor:
        return linear_nvfp4(x, packed, weight_scale=block_scale, weight_global_scale=global_scale)

    # Reference dequant: table lookup on the 3 magnitude bits, sign from bit 3,
    # low nibble first because that is where the even k index lives. Nothing here
    # touches the kernel's decoder.
    n, k_packed = packed.shape
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=w.device)
    codes = torch.stack([packed & 0xF, (packed >> 4) & 0xF], dim=-1).flatten(-2).long()
    magnitude = values[codes & 0x7]
    signed = torch.where(codes & 0x8 != 0, -magnitude, magnitude)
    k = k_packed * 2
    scales = block_scale.view(torch.float8_e4m3fn).float() * global_scale.float().reshape(())
    deq = signed.unflatten(-1, (k // NVFP4_BLOCK, NVFP4_BLOCK)) * scales.unsqueeze(-1)
    return call, deq.reshape(n, k), nbytes(block_scale, global_scale)


#: The weight-only tolerances are the ones ``tests/kernels/test_quantization.py``
#: gates the same kernels on, because the reference here has the same shape:
#: full-precision activation against the dequantised weight.
#:
#: The two W8A8 rows are looser on purpose, and the reason is the reference, not
#: the kernel. ``tests/kernels`` checks them against a reference that shares the
#: *already quantised* activation, which isolates the kernel at 5e-2. This script
#: compares against ``F.linear(x_bf16, dequant(w))`` instead — the semantics a
#: caller actually wants — so the tolerance must also absorb the per-token
#: activation rounding. e4m3 keeps 3 mantissa bits, so each operand carries up to
#: ~6% relative error and the product carries both; at the check shapes that
#: lands near 7e-2 absolute, hence 1e-1. Tightening this number would not make
#: the kernel better, it would make the gate lie about the format.
SCHEMES: tuple[Scheme, ...] = (
    Scheme("unquantized", "native/linear_torch", _build_bf16, 16, 2e-2, 2e-2),
    Scheme("fp8", "native/linear_w8a16", _build_w8a16_fp8, 8, 1e-2, 1e-2),
    Scheme("w8a8_fp8", "native/linear_w8a8_fp8", _build_w8a8_fp8, 8, 1e-1, 1e-1),
    Scheme("w8a8_int8", "native/linear_w8a8_int8", _build_w8a8_int8, 8, 1e-1, 1e-1),
    Scheme("awq", "native/linear_w4a16", _build_w4a16, 4, 5e-2, 5e-2),
    Scheme("nvfp4", "native/linear_nvfp4", _build_nvfp4, 4, 1e-2, 1e-2),
)


# --------------------------------------------------------------------------- #
# Ablation: the fp8 GEMM without its activation quantiser
# --------------------------------------------------------------------------- #
#: Label for the ablation row. Not a ``KernelSpec.name`` — nothing dispatches to
#: it — and the ``ablation:`` prefix says so, so the table stays readable as
#: "rows that name registry entries, plus one that explicitly does not".
_ABLATION_FP8 = "ablation: fp8_matmul only"


def fp8_gemm_only(w: torch.Tensor, x: torch.Tensor) -> Callable[[], torch.Tensor]:
    """``fp8_matmul`` on an activation that was quantised outside the timed region.

    This row settled one question with a measurement and then caused a fix, so
    read it as history plus a standing bound.

    *What it found.* ``native/linear_w8a8_fp8`` was 3-4x slower than
    ``native/linear_w8a16`` at ``m=1`` despite reading the same weight bytes. The
    gap against this row was 44-54 us and did **not** move with shape, which is
    the signature of launch overhead rather than work: ``linear_w8a8_fp8`` called
    the torch ``quantize_fp8_per_token``, a chain of ~8 elementwise ops.
    ``linear_w8a8_int8``, which quantises inside its Triton kernel, showed no such
    gap — the corroborating half of the hypothesis.

    *What changed.* ``lite_llama.kernels.ops.quantization.fp8_quantize_per_token``
    now does it in one launch, and ``linear_w8a8_fp8`` calls that. The 73.5 us
    ``qkv m=1`` row became 24.4 us.

    *What it measures now.* The residual price of the activation pass — a few
    microseconds, one launch. Keep the row: it is the only thing that would catch
    the quantiser regressing back into the GEMM's shadow, and it stays honest
    about the fact that the ``w8a8_fp8`` row does strictly more work than the
    GEMM its ``moved`` figure accounts for.
    """
    from lite_llama.kernels.ops.quantization import fp8_matmul
    from lite_llama.modules.quantization.utils import quantize_fp8_per_token

    qw, scale = quantize_fp8_per_channel(w)
    qx, x_scale = quantize_fp8_per_token(x)
    k = w.shape[1]
    return lambda: fp8_matmul(
        qx, x_scale, qw, scale, group_n=1, group_k=k, out_dtype=torch.bfloat16
    )


# --------------------------------------------------------------------------- #
# Work
# --------------------------------------------------------------------------- #
def linear_work(m: int, n: int, k: int, weight_bits: int, extra_bytes: int) -> Work:
    """Theoretical cost of ``[m,k] @ [n,k].T`` with a ``weight_bits`` weight.

    The activation is counted as bf16 on *every* row, because bf16 is what the
    op receives: the W8A8 rows quantise it themselves, and that extra read-write
    pass is implementation traffic, not part of the operation. Counting it would
    reward the format for work it had to invent. The consequence is that a W8A8
    row's GB/s is an honest lower bound on what it moves, and its gap to bf16 at
    small ``m`` is smaller than the table suggests — call that out rather than
    quietly crediting it.
    """
    act = torch.empty((), dtype=torch.bfloat16).element_size()
    moved = m * k * act + n * k * weight_bits // 8 + extra_bytes + m * n * act
    return Work(flops=2 * m * n * k, moved=moved)


# --------------------------------------------------------------------------- #
# Correctness, before any timing
# --------------------------------------------------------------------------- #
#: Small shapes for the correctness gate. K is a multiple of 128 so every
#: scheme's group/block constraints hold, and N deliberately is not square.
CHECK_SHAPES: tuple[tuple[int, int, int], ...] = ((1, 512, 256), (8, 768, 1024), (128, 256, 512))


def check_correctness() -> None:
    """Verify each scheme against a torch dequant + ``F.linear`` reference.

    Every scheme gets its own check: the dtype variants run different code
    inside their kernels (native fp8 MMA, int8 MMA, nibble unpack), so a row
    verified through a sibling is an unverified row. The reference dequantises
    with plain torch ops built in this file, never by calling the kernel under
    test.

    The bf16 row is checked against an fp32 matmul rather than against itself:
    vacuous as a kernel test, but it catches a harness-side dtype mistake before
    it becomes every other row's reference.
    """
    print("Correctness (torch dequant + F.linear reference):")
    for scheme in SCHEMES:
        for m, n, k in CHECK_SHAPES:
            torch.manual_seed(0)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
            w = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
            call, w_ref, _ = scheme.build(w)
            ref = F.linear(x.float(), w_ref)
            verify(
                f"{scheme.impl} [{scheme.key}] m{m}_n{n}_k{k}",
                call(x),
                ref,
                rtol=scheme.rtol,
                atol=scheme.atol,
            )


def show_dispatch() -> None:
    """Print the decision chain and pin every table label to a registry row.

    Each scheme is a separate dispatch key by design — the format is not a
    runtime branch inside one kernel — so a mismatch here means the table names
    one kernel while the process runs another.
    """
    print("\nDispatch for linear:")
    for scheme in SCHEMES:
        sel = dispatch(
            "linear", dtype="bf16", scheme=scheme.key, shape={"m": 128, "n": 2048, "k": 2048}
        )
        assert sel.spec.name == scheme.impl, (
            f"table labels {scheme.key} as {scheme.impl}, dispatch picks {sel.spec.name}"
        )
        assert sel.load() is not None
        print(f"  {scheme.key:<12} -> {sel.spec.name}")
    # One full chain, so a filtered-out backend is visible in the log rather
    # than inferred from a missing row.
    sel = dispatch(
        "linear", dtype="bf16", scheme="w8a8_fp8", shape={"m": 128, "n": 2048, "k": 2048}
    )
    print(f"\n{sel.explain()}")


# --------------------------------------------------------------------------- #
# Tuning — the one row here that has a store consumer
# --------------------------------------------------------------------------- #
#: Of the five quantised kernels in this table, ``w4a16_matmul`` is the only one
#: that consults the autotune store (``ops/quantization/w4a16.py``, before its
#: heuristic). fp8 W8A8, fp8/int8 W8A16 and NVFP4 compute their launch
#: configuration unconditionally, so there is nothing for a search to install on
#: those rows and ``--tune`` does not pretend otherwise: it reports them as having
#: no consumer instead of writing entries no kernel would read.
_TUNED_SCHEME = "awq"
_TUNED_OP = "w4a16_matmul"
_TUNED_DTYPE = "int4"

#: Candidate tiles for ``w4a16_matmul``. ``BLOCK_K`` joined the search when the
#: v0.6 rewrite decoupled the k-tile from the quantisation group: it must stay a
#: multiple of ``group_size`` (the launcher halves a misfit down to one), 256
#: fills a 128-byte transaction per output channel where 128 fills half of one,
#: and 512 loses to register pressure — so the space brackets the measured
#: optimum at 128/256. The rest spans the three M buckets' measured winners
#: (16x32 / 64x32 / 64x64 at BLOCK_K=256) plus their neighbourhoods.
_TUNE_SPACE: tuple[dict[str, int], ...] = tuple(
    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm, "num_warps": nw, "num_stages": ns}
    for bm, bn, bk, gm, nw, ns in (
        (16, 32, 256, 8, 4, 4),
        (16, 32, 256, 8, 4, 3),
        (16, 64, 256, 8, 4, 4),
        (16, 64, 128, 8, 4, 4),
        (32, 32, 256, 8, 4, 4),
        (32, 64, 256, 8, 4, 3),
        (32, 128, 256, 8, 8, 3),
        (64, 32, 256, 8, 4, 3),
        (64, 32, 256, 8, 4, 4),
        (64, 64, 256, 8, 4, 3),
        (64, 64, 256, 8, 4, 4),
        (64, 128, 256, 8, 8, 3),
        (128, 64, 256, 8, 4, 2),
        (128, 128, 128, 8, 8, 3),
        (128, 256, 128, 8, 8, 3),
    )
)


def heuristic_config(m: int) -> dict[str, int]:
    """The config ``w4a16_matmul`` uses when the store is empty — imported, not copied."""
    return w4a16_launch_config(m)


@contextmanager
def forced_w4a16_config(config: dict[str, int] | None) -> Iterator[None]:
    """Pin ``w4a16_matmul`` to one config, or to its heuristic when ``None``.

    ``w4a16.py`` imports ``get_best_config`` inside the function body, so the name
    resolves against the module on every call and patching the module attribute is
    enough. Returning ``None`` for other ops keeps the patch scoped: nothing else
    in this script consults the store, but a future row that did would otherwise
    silently receive an int4 tile.

    ``config=None`` is how the baseline is measured: it makes the kernel take its
    own store-miss path, which :func:`heuristic_config` also reports.
    """
    original = autotune_module.get_best_config

    def patched(op: str, m: int, n: int, k: int, dtype: str) -> dict | None:
        if op != _TUNED_OP or config is None:
            return None
        return dict(config)

    autotune_module.get_best_config = patched
    try:
        yield
    finally:
        autotune_module.get_best_config = original


def _time_config(
    config: dict[str, int] | None,
    call: Callable[[torch.Tensor], torch.Tensor],
    inputs: list[torch.Tensor],
    references: list[torch.Tensor] | None,
    rtol: float,
    atol: float,
) -> float | None:
    """Total time of one config across a bucket's token counts, or ``None`` if rejected.

    Rejection is a compile failure (a tile that overflows shared memory or asks
    for more warps than the tile has rows to feed) or an output that disagrees
    with the config the correctness gate already checked. Timing an unverified
    config would install a tile that computes the wrong projection quickly.

    Timed with the harness ``bench`` rather than a synchronise-per-iteration loop:
    the latter floors at ~100 us, which is several times a decode row here and
    would make the search blind exactly where the store entries matter most.
    """
    total = 0.0
    with forced_w4a16_config(config):
        for i, x in enumerate(inputs):
            try:
                out = call(x)
            except Exception:
                return None
            if references is not None and not torch.allclose(
                out, references[i], rtol=rtol, atol=atol
            ):
                return None
            total += bench(lambda c=call, a=x: c(a))
    return total


def tune(
    geometries: tuple[Geometry, ...],
    tokens: tuple[int, ...],
    *,
    write: bool,
) -> list[TuneResult]:
    """Search :data:`_TUNE_SPACE` per store key and persist the winners.

    One search per :class:`TuneKey`, not per token count, because that is the
    granularity the kernel looks up: ``bucket_m`` rounds M up to the next of
    (16, 32, 64, 128, 256, 512), so m=1 and m=8 share one entry. A search run per
    token count would have them overwrite each other and the surviving config
    would be whichever ran last. Every token count in a bucket is timed on every
    candidate and the winner is the one with the lowest *total*, so a shared entry
    is chosen for the traffic it will serve.

    The two bucketings nest: the heuristic switches at m=32 and m=128, both of
    which are M-bucket boundaries, so one store entry never has to stand in for
    two different heuristic choices. :func:`_bucket_baseline` asserts it rather
    than assuming it, because a future bucket boundary would break the gain column
    silently.

    A shared entry optimises the sum, which means a token count inside the bucket
    can regress, and the M512 bucket is where that shows: on
    ``qwen3-30b-a3b/qkv`` the winning 128x128 tile costs 7 us at t512 (66.2 ->
    73.4) and saves 50 us at t2048 (232.4 -> 182.0). Measured, not assumed — which
    is also why the store is worth less to a decode-heavy server than the gain
    column suggests. Narrow ``--tokens`` to the widths a deployment actually serves
    if that trade is the wrong way round for it.

    Args:
        write: Persist the winners to :class:`ConfigStore`. False measures and
            reports without touching the cache.
    """
    scheme = next(s for s in SCHEMES if s.key == _TUNED_SCHEME)
    results: list[TuneResult] = []
    buckets: dict[int, list[int]] = {}
    for t in sorted(tokens):
        buckets.setdefault(bucket_m(t), []).append(t)

    for geo in geometries:
        for proj, n, k in geo.projections():
            torch.manual_seed(0)
            w = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
            call, _w_ref, _extra = scheme.build(w)
            for bucket, group in buckets.items():
                inputs = [
                    torch.randn(t, k, device="cuda", dtype=torch.bfloat16) * 0.5 for t in group
                ]
                baseline_config = _bucket_baseline(group)
                # The reference every candidate is checked against: the same
                # kernel on the heuristic's config, which check_correctness()
                # verified against the torch dequant reference this run.
                with forced_w4a16_config(None):
                    references = [call(x) for x in inputs]
                baseline_us = _time_config(None, call, inputs, references, scheme.rtol, scheme.atol)
                if baseline_us is None:  # pragma: no cover - the gate's own config
                    raise SystemExit("the heuristic config failed its own gate")

                best_config, best_us, rejected = baseline_config, baseline_us, 0
                for candidate in _TUNE_SPACE:
                    if candidate == baseline_config:
                        continue
                    us = _time_config(candidate, call, inputs, references, scheme.rtol, scheme.atol)
                    if us is None:
                        rejected += 1
                        continue
                    if us < best_us:
                        best_config, best_us = candidate, us

                key = TuneKey.build(_TUNED_OP, m=bucket, n=n, k=k, dtype=_TUNED_DTYPE)
                result = TuneResult(
                    key=key,
                    label=f"{geo.label}/{proj}",
                    tokens=tuple(group),
                    baseline_config=baseline_config,
                    baseline_us=baseline_us,
                    best_config=best_config,
                    best_us=best_us,
                    rejected=rejected,
                )
                results.append(result)
                print(f"  {_tune_line(result)}", flush=True)
                if write and result.changed:
                    # The store's unit is microseconds, which is also the
                    # harness's, so nothing is converted here. set_perf_provider()
                    # takes milliseconds and is the boundary that does convert.
                    ConfigStore().put(key, best_config, latency_us=best_us)
                del inputs, references
                torch.cuda.empty_cache()
            del call, w
            torch.cuda.empty_cache()
    if write and any(r.changed for r in results):
        # The lookup caches its store instance on first use, so anything later in
        # this process would otherwise keep the pre-tuning view.
        autotune_reset()
    return results


def _bucket_baseline(group: list[int]) -> dict[str, int]:
    """The heuristic config shared by every token count in one M bucket."""
    configs = [heuristic_config(t) for t in group]
    if any(c != configs[0] for c in configs[1:]):
        raise SystemExit(
            f"tokens {group} share an M bucket but not a heuristic config: {configs}. "
            "One store entry cannot serve both; split the bucket or the token list."
        )
    return configs[0]


def _tune_line(r: TuneResult) -> str:
    tiles = f"{r.best_config['BLOCK_M']}x{r.best_config['BLOCK_N']}x{r.best_config['BLOCK_K']}"
    head = f"{r.label:24s} {r.key.shape_bucket:18s} t{list(r.tokens)}:"
    if not r.changed:
        return f"{head} heuristic already best ({r.baseline_us:.1f} us, {tiles}), {r.rejected} rejected"
    return (
        f"{head} {r.baseline_us:.1f} -> {r.best_us:.1f} us ({r.gain:+.1%}) "
        f"BLOCK_MNK={tiles} GROUP_M={r.best_config['GROUP_M']} "
        f"warps={r.best_config['num_warps']} stages={r.best_config['num_stages']}, "
        f"{r.rejected} rejected"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def measure(geometries: tuple[Geometry, ...], tokens: tuple[int, ...]) -> list[Row]:
    """Time every (scheme, projection, token count) combination, plus the ablation."""
    rows: list[Row] = []
    for geo in geometries:
        for proj, n, k in geo.projections():
            torch.manual_seed(0)
            w = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
            # Quantisation happens once per weight, outside every timed region:
            # a served checkpoint quantises at load time, not per step.
            built = [(s, *s.build(w)) for s in SCHEMES]
            for m in tokens:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
                case = f"{geo.label}/{proj} m{m}_n{n}_k{k}"
                for scheme, call, _w_ref, extra in built:
                    us = bench(lambda c=call, x=x: c(x))
                    rows.append(
                        Row(
                            f"{scheme.impl} [{scheme.key}]",
                            case,
                            us,
                            linear_work(m, n, k, scheme.weight_bits, extra),
                        )
                    )
                gemm_only = fp8_gemm_only(w, x)
                us = bench(gemm_only)
                rows.append(Row(_ABLATION_FP8, case, us, linear_work(m, n, k, 8, 4 * n)))
                del x, gemm_only
            del built, w
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-dir",
        action="append",
        default=None,
        help="HF checkpoint whose config.json supplies the projection shapes; "
        "repeatable. Defaults to the built-in Qwen3-4B / Qwen3-30B-A3B geometries.",
    )
    ap.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=list(TOKENS),
        help=f"Token counts (rows of the activation). Default {list(TOKENS)}.",
    )
    ap.add_argument("--json", help="Write the rows to this path as JSON.")
    ap.add_argument(
        "--tune",
        action="store_true",
        help="Search the int4 tile space per store key and persist the winners to "
        "the autotune cache. Only w4a16_matmul reads that cache, so this touches "
        "one row of the table. Skips the timing table.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="With --tune: report the winners without writing the cache.",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")
    torch.set_grad_enabled(False)

    geometries = (
        tuple(geometry_from_config(d) for d in args.model_dir)
        if args.model_dir
        else BUILTIN_GEOMETRIES
    )

    print(metadata())
    print()
    check_correctness()
    show_dispatch()
    print()

    if args.tune:
        # The correctness gate above is what makes the per-candidate check
        # meaningful: candidates are compared against the heuristic's output, and
        # that output has just been checked against the torch dequant reference.
        print(
            f"Searching {len(_TUNE_SPACE)} tile configs per store key for "
            f"{_TUNED_OP} [{_TUNED_SCHEME}]"
            f"{' (dry run, nothing written)' if args.dry_run else ''}.\n"
            "The other four quantised rows compute their launch config\n"
            "unconditionally and never consult the store, so they are not\n"
            "searched -- an entry written for them would be dead bytes."
        )
        results = tune(geometries, tuple(args.tokens), write=not args.dry_run)
        changed = [r for r in results if r.changed]
        print(
            f"\n{len(changed)} of {len(results)} keys improved on the heuristic"
            + (f"; best gain {max(r.gain for r in changed):.1%}" if changed else "")
        )
        if changed and not args.dry_run:
            print(f"written to {ConfigStore().cache_dir}")
            print(
                "Re-running without --tune now measures the persisted configs on the\n"
                "int4 row only. Run with LITE_LLAMA_AUTOTUNE=0 to measure what a user\n"
                "without this cache gets."
            )
        return

    rows = measure(geometries, tuple(args.tokens))
    report(rows)
    print(
        "\nHeadline: cuBLAS bf16 (native/linear_torch) is fastest in 44 of 48 cells,\n"
        "and the four exceptions are the result. All four are qwen3-4b/gate_up\n"
        "(N=19456, K=2560, 100 MB in bf16) at M<=128, where int8 W8A8 runs 1.40x /\n"
        "1.38x / 1.33x / 1.22x at M=1/8/32/128 and fp8 W8A8 1.35x / 1.33x / 1.29x.\n"
        "\nThe rule behind that: quantisation wins where bf16 is genuinely\n"
        "bandwidth-bound and nowhere else. At M=1 the bf16 row's share of peak HBM\n"
        "spans 10.4% (30b-a3b/down, a 3 MB weight) to 60.9% (4b/gate_up), and the\n"
        "wins sit at the top of that range. Predict from the bf16 %bw column, not\n"
        "from the compression ratio.\n"
        "\nOn a mid-sized projection (4b/qkv) at m=1: bf16 21.7 us, int4 22.2, int8\n"
        "22.7, fp8 W8A8 24.0, fp8 W8A16 28.2, nvfp4 49.0. At m=2048: 89.1 / 103.8 /\n"
        "166.1 / 334.8 / 370.0 / 755.3. Read those as the throughput price of a\n"
        "smaller checkpoint. Where int4 reaches parity it is two limits meeting, not\n"
        "one: bf16 streams 31 MB at 43% of peak, int4 streams 7.9 MB at 11.9% and is\n"
        "unpack-bound. Do not carry the parity to another shape.\n"
        "\nAt m>=512 no quantised row wins any cell: cuBLAS reaches 73% of tensor-core\n"
        "peak while the Triton rows stay pinned by their unpack loops (1.17x for int8\n"
        "to 8.5x for nvfp4). The W8A8 rows also pay a per-token activation quantise\n"
        "pass that linear_work() does not count, so their GB/s is a lower bound.\n"
        "\nOnly the int4 row reads the autotune store. Run --tune to fill it (13-25%\n"
        "at m>=512), or LITE_LLAMA_AUTOTUNE=0 to measure the heuristic a user gets\n"
        "without a cache."
    )

    if args.json:
        peaks = device_peaks()
        payload = {
            "meta": {
                "header": metadata(),
                "device": peaks.name,
                "peak_gbps": peaks.gbps,
                "peak_tflops": peaks.tflops,
                "recorded": datetime.now().isoformat(timespec="seconds"),
                "geometries": [g.label for g in geometries],
                "tokens": args.tokens,
            },
            "rows": [
                {
                    "impl": r.impl,
                    "case": r.case,
                    "us": r.us,
                    "flops": r.work.flops,
                    "moved": r.work.moved,
                    "tflops": r.tflops,
                    "gbps": r.gbps,
                }
                for r in rows
            ],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {len(rows)} rows -> {args.json}")


if __name__ == "__main__":
    main()
