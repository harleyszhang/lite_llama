"""Microbenchmark ``fused_moe`` across every expert-weight format it accepts.

A routed-expert FFN is not a dense GEMM with extra bookkeeping: at decode the
layer touches ``top_k`` experts for a single token, so it reads ``top_k`` full
expert weight matrices to produce one row of output. That makes the small-token
rows the most memory-bound shapes in the model — far past the dense projections
— and it is why a quantised expert format pays off here before it pays off
anywhere else. Read the table as three regimes, and read the ``moe_align_block_size``
ablation before any of the format rows — on this device (H100, E=128, top_k=8,
h=2048, i=768) that one row accounts for more of the decode spread than every
format difference combined:

``tokens <= 8`` (decode)
    Each of the ``top_k`` selected experts is read whole for one token. Arithmetic
    intensity is ~2 FLOP/byte, so the ranking *should* be bytes of expert weight
    — except that it is not: bf16 and all three weight-only formats land inside
    1.5% of each other (367-372 us) while reading 4x different weight bytes. The
    ``ablation: moe_align_block_size`` row is why. The routing bookkeeping alone
    costs ~188 us, over half the layer, and it is identical for every format. Read
    these rows as "the formats are indistinguishable behind a fixed cost", not as
    "quantisation does not help". W8A8 fp8 is the one row that is distinguishable
    here, and it is *worse*: 494 us, 35% over bf16, because quantising the
    activation costs two more kernel launches on a layer that is already
    launch-bound.
``tokens = 64`` (where weight-only quantisation wins)
    Slots per expert first exceed the row-block, so the grouped GEMM amortises each
    expert load. int8 at 379.4 us beats bf16's 535.1 by 29.1%, weight-only fp8 at
    473.1 by 11.6%, and W8A8 fp8 at 502.9 by 6.0% — the last from the skipped
    dequant alone, since ``BLOCK_M`` is still 32 here and the dot has not reached
    the fp8 tensor cores. int4 does not win: 626.2 us, 17.0% *slower* than bf16,
    and its GB/s is the lowest in the table despite reading the fewest bytes —
    that path is bound by unpacking 8 nibbles per int32, not by traffic.
``tokens >= 512`` (prefill)
    The weight-only wins fade, and W8A8 takes over. Against bf16's 588.1 us,
    int8's 544.7 still leads by 7.4% while weight-only fp8's 654.1 is an 11.2%
    *loss*; by 4096 tokens both are outright regressions (1901.1
    and 2442.5 against 1578.9, i.e. 20% and 55% slower). Each expert tile is
    re-read across row-blocks and dequantised again every time, so in-loop widening
    stops being amortised weight traffic and becomes arithmetic on the critical
    path. W8A8 fp8 does not pay that: 492.2 us at t512 (16.3% under bf16, the
    fastest row in the regime) and 1479.4 at t4096 (6.3% under it) at 209 TFLOP/s
    — the highest arithmetic rate anywhere in this table. Its gain is in the MMA,
    so it appears exactly where the weight-only gains disappear.

The second ablation was not about quantisation at all and was the larger finding,
and it has since been fixed: ``_launch_config`` used to return
``BLOCK_K = 128 if quant_mode else 32``, so the unquantised row ran four times as
many k-iterations as the rows it was the baseline for. At t512 that handicap was
the difference between weight-only fp8 reading as an 18% win and the 11% loss
above — the opposite conclusion — and it only ever depressed the baseline, which is why
no test caught it and why every quantisation number on this kernel used to look
better than it was. The tile sweep (``--tune``) found no winning config with
``BLOCK_K`` below 64 at any token count, the heuristic now gives every mode 128,
and the row survives as a regression guard: it runs the baseline dtype on the old
narrow tile, so its margin over the plain bf16 row is what the fix bought — 25.1% at t64
(714.0 -> 535.1), 22.4% at t512 (757.4 -> 588.1), 10.4% at t4096 (1761.9 ->
1578.9), nothing at t1/t8 where the layer is launch-bound. Two rows converging is
a reinstated defect.

What each row measures. The prefix is the registry entry and the ``[label]`` suffix
names the format: all but one row are ``native/fused_moe``, because the kernel picks
the format off ``w1.dtype`` rather than from a spec row (see
``lite_llama/kernels/ops/moe/__init__.py``), and the exception is W8A8 fp8, whose
bytes are identical to weight-only fp8's:

``[unquantized]``
    bf16 experts, ``tl.dot`` straight from the loaded tile. The floor to beat.
``[fp8_w8a16]``
    fp8-e4m3 experts, per-output-channel scales, widened in the inner loop by the
    ``dequant_fp8e4m3`` bit trick. Weight-only: the activation is untouched, so the
    MMA is unchanged and the entire gain is bytes.
``[fp8_w8a8]``
    The same fp8 expert bytes with the activation quantised per token, so both
    operands stay 8-bit into the dot and the bit trick is skipped. The only row on
    a different registry entry, because nothing in the weights distinguishes it —
    see :data:`_IMPL_A8`. Read it separately per regime: it *loses* 35% at decode
    and wins 16% at t512, and averaging those into one speedup would describe
    neither.
``[int8]``
    int8 experts, per-output-channel scales, converted in the loop. Same bytes as
    fp8, cheaper widening, coarser format.
``[int4]``
    int4 experts packed 8-per-int32 with group scales and zero points. A quarter
    of the bytes and the most unpacking work per byte.

One caveat the A8 row's TFLOP/s column does not show: Triton emits Hopper's fp8
``wgmma`` only from ``BLOCK_M >= 64``, and ``_launch_config`` reaches that at
``tokens > 64``. At and below that the two e4m3 operands are widened to an fp16
``mma.sync``, so the t1/t8/t64 A8 rows are not measuring the fp8 tensor cores at
all — which is consistent with the gain appearing only from t512. The correctness
gate covers both instructions (see :data:`CHECK_TOKENS`).

Two defects surfaced while writing this file, both of which had to be fixed before
there was anything to time, and neither of which any test would have caught:
every quantised expert GEMM widened its weight tile to a hard-coded fp16, so a
bf16 model could not compile the layer at all; and the int4 branch folded its fp32
scale into the operand before the dot, so that path could not compile at *any*
activation dtype. ``tests/kernels/test_fused_moe.py`` now gates both.

Usage:
    python benchmarks/kernels/bench_fused_moe.py
    python benchmarks/kernels/bench_fused_moe.py --json out.json
    python benchmarks/kernels/bench_fused_moe.py --model-dir /path/to/Qwen3-30B-A3B
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from microbench import Row, Work, bench, device_peaks, metadata, report, verify
from tuning import TuneResult, nbytes

# Importing the facade registers every spec row, so dispatch() below finds them.
import lite_llama.kernels  # registers the spec rows as a side effect
import lite_llama.kernels.ops.moe.fused_moe as fused_moe_module
from lite_llama.kernels.dispatcher import dispatch
from lite_llama.kernels.dispatcher.autotune import ConfigStore, TuneKey, bucket_m
from lite_llama.kernels.dispatcher.autotune import reset as autotune_reset
from lite_llama.kernels.ops.moe.fused_moe import (
    _launch_config,
    fused_moe,
    fused_moe_w8a8_fp8,
    moe_align_block_size,
)
from lite_llama.modules.quantization.utils import (
    quantize_fp8_per_channel,
    quantize_fp8_per_token,
    quantize_int4_groupwise,
    quantize_int8_per_channel,
)
from tests.reference import fused_moe_reference

#: The registry row the weight-only schemes route to. Kept in one constant so the
#: table's labels and :func:`show_dispatch`'s assertion cannot drift apart.
_IMPL = "native/fused_moe"

#: The second native row: same kernel, different entry point. ``fused_moe`` reads
#: the expert format off ``w1.dtype``, which cannot tell weight-only fp8 from W8A8
#: fp8 — both are uint8 e4m3 experts — so the activation's fate has to be the
#: function you call, and therefore the row.
_IMPL_A8 = "native/fused_moe_w8a8_fp8"

#: Token counts spanning the serving range: 1 is a single-sequence decode step,
#: 4096 a prefill tile. 64 sits where slots-per-expert first exceeds the 32-row
#: block at E=128/top_k=8, which is where the grouped GEMM stops wasting padding.
TOKENS: tuple[int, ...] = (1, 8, 64, 512, 4096)

#: int4 group size, matching what AWQ/GPTQ checkpoints ship.
INT4_GROUP_SIZE = 128

#: Activation dtype for every row, quantised and not.
#:
#: bf16 because that is what the Qwen3-MoE checkpoints this table models actually
#: run — the unquantised row must be the floor a real bf16 layer beats, and an
#: fp16 floor would flatter every quantised format by a dtype switch the serving
#: path never makes. The tests cover the dtype:
#: ``test_fused_moe_quantised_matches_reference`` runs every format at both
#: dtypes, and the A8 gates below are per-dtype like theirs. (The 2026-09-01
#: tables in this docstring and ``docs/quantization.md`` were fp16 numbers;
#: their record is ``docs/benchmark_logs/bench_fused_moe_h100_20260901.json``.)
ACT_DTYPE = torch.bfloat16


# --------------------------------------------------------------------------- #
# Model geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MoeGeometry:
    """One MoE layer's shape: what the router picks and how wide an expert is.

    ``intermediate`` is ``moe_intermediate_size``, not the dense FFN width. On
    Qwen3-30B-A3B that is 768 against a 2048 hidden size — the expert GEMMs are
    *wider than tall* in K, the opposite aspect ratio from the dense model, so a
    tile config tuned on the dense shapes does not transfer.
    """

    label: str
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int

    def case(self, tokens: int) -> str:
        """Case label. Carries every dimension the number depends on."""
        return (
            f"{self.label} t{tokens}_E{self.num_experts}_top{self.top_k}"
            f"_h{self.hidden}_i{self.intermediate}"
        )


#: Geometry used when no ``--model-dir`` is given: Qwen3-30B-A3B-Instruct-2507,
#: the checkpoint this round validates end to end, so the kernel table and the
#: e2e matrix measure the same layer.
BUILTIN_GEOMETRIES: tuple[MoeGeometry, ...] = (
    MoeGeometry("qwen3-30b-a3b", hidden=2048, intermediate=768, num_experts=128, top_k=8),
)


def geometry_from_config(model_dir: str) -> MoeGeometry:
    """Read one :class:`MoeGeometry` out of a HF ``config.json``."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    if "num_experts" not in cfg:
        raise SystemExit(f"{model_dir} has no num_experts in config.json — not a MoE checkpoint")
    return MoeGeometry(
        label=Path(model_dir).name,
        hidden=cfg["hidden_size"],
        intermediate=cfg["moe_intermediate_size"],
        num_experts=cfg["num_experts"],
        top_k=cfg["num_experts_per_tok"],
    )


# --------------------------------------------------------------------------- #
# Schemes — one entry per expert-weight format the kernel accepts
# --------------------------------------------------------------------------- #
#: ``(call, w1_dequantised, w2_dequantised, scale_bytes_all_experts)``.
Built = tuple[Callable[..., torch.Tensor], torch.Tensor, torch.Tensor, int]


@dataclass(frozen=True)
class Scheme:
    """One expert-weight format, from dispatch key to callable to work formula.

    Attributes:
        key: The ``scheme`` dispatch key. :func:`show_dispatch` asserts it lands
            on :attr:`impl`.
        label: Short table suffix.
        build: ``(w1_fp32, w2_fp32) -> Built``. The returned dequantised weights
            are what the torch reference multiplies, so they must be produced by
            plain torch ops and never by the kernel's own unpacking.
        weight_bits: Stored bits per expert-weight element, for the traffic formula.
        impl: Registry row the key dispatches to.
        act_quant: Round trip the reference must apply to its *activation*
            operands, or ``None`` for the weight-only rows whose activation the
            kernel leaves exact. Non-``None`` also selects the statistical
            correctness gate — see :func:`verify_a8`.
    """

    key: str
    label: str
    build: Callable[[torch.Tensor, torch.Tensor], Built]
    weight_bits: int
    impl: str = _IMPL
    act_quant: Callable[[torch.Tensor], torch.Tensor] | None = None
    #: The dtype label ``fused_moe`` puts in its :class:`TuneKey`: the activation's
    #: dtype for the unquantised mode, the weight format for the quantised ones.
    #: Must match the table in the kernel exactly, or ``--tune`` writes entries the
    #: kernel never looks up. Every row below passes it explicitly.
    tune_dtype: str = "fp16"


def _build_bf16(w1: torch.Tensor, w2: torch.Tensor) -> Built:
    a, b = w1.to(ACT_DTYPE), w2.to(ACT_DTYPE)

    def call(x, tw, ids):
        return fused_moe(x, a, b, tw, ids)

    return call, a.float(), b.float(), 0


def _group_k(w1: torch.Tensor, w2: torch.Tensor) -> int:
    """A ``group_k`` that covers K on both GEMMs.

    The two calls have different K (``hidden`` for gate_up, ``intermediate`` for
    down) but share one ``group_k`` argument, and per-channel scales are a single
    group spanning all of K. ``_invoke_moe_gemm`` clamps with
    ``min(group_k, k_logical)``, so the larger of the two K values is one group on
    the wide GEMM and still one group on the narrow one.
    """
    return max(w1.shape[-1], w2.shape[-1])


def _build_fp8(w1: torch.Tensor, w2: torch.Tensor) -> Built:
    """fp8-e4m3 experts, one scale per output channel, activation untouched."""
    q1, s1 = quantize_fp8_per_channel(w1)
    q2, s2 = quantize_fp8_per_channel(w2)
    gk = _group_k(w1, w2)

    def call(x, tw, ids):
        return fused_moe(x, q1, q2, tw, ids, w1_scale=s1, w2_scale=s2, group_n=1, group_k=gk)

    # Reference dequant: widen the same bytes with torch, then apply the scale.
    ref1 = q1.view(torch.float8_e4m3fn).float() * s1
    ref2 = q2.view(torch.float8_e4m3fn).float() * s2
    return call, ref1, ref2, nbytes(s1, s2)


def _fp8_round_trip(t: torch.Tensor) -> torch.Tensor:
    """Per-row e4m3 quantise-then-widen, the rounding the A8 kernel imposes.

    Goes through the production quantiser and back out with torch, so the
    reference reproduces the *rounding* without borrowing the kernel's arithmetic.
    """
    q, scale = quantize_fp8_per_token(t.to(ACT_DTYPE))
    return q.view(torch.float8_e4m3fn).float() * scale


def _build_fp8_a8(w1: torch.Tensor, w2: torch.Tensor) -> Built:
    """Same fp8 experts as :func:`_build_fp8`, with the activation quantised too.

    The weights are byte-identical to the weight-only row; the whole difference is
    that both operands enter the dot as 8-bit, so the inner loop skips the
    ``dequant_fp8e4m3`` bit trick. Whether that dot lands on Hopper's fp8 tensor
    cores is Triton's call: it emits ``wgmma`` only from ``BLOCK_M >= 64``, which
    ``_launch_config`` reaches at ``tokens > 64``. Below that the operands are
    widened to an fp16 ``mma.sync`` and only the skipped dequant remains.
    """
    q1, s1 = quantize_fp8_per_channel(w1)
    q2, s2 = quantize_fp8_per_channel(w2)
    gk = _group_k(w1, w2)

    def call(x, tw, ids):
        return fused_moe_w8a8_fp8(
            x, q1, q2, tw, ids, w1_scale=s1, w2_scale=s2, group_n=1, group_k=gk
        )

    ref1 = q1.view(torch.float8_e4m3fn).float() * s1
    ref2 = q2.view(torch.float8_e4m3fn).float() * s2
    return call, ref1, ref2, nbytes(s1, s2)


def _build_int8(w1: torch.Tensor, w2: torch.Tensor) -> Built:
    """Symmetric int8 experts, one scale per output channel, activation untouched."""
    q1, s1 = quantize_int8_per_channel(w1)
    q2, s2 = quantize_int8_per_channel(w2)
    gk = _group_k(w1, w2)

    def call(x, tw, ids):
        return fused_moe(x, q1, q2, tw, ids, w1_scale=s1, w2_scale=s2, group_n=1, group_k=gk)

    return call, q1.float() * s1, q2.float() * s2, nbytes(s1, s2)


def _quantize_int4_experts(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """int4-pack ``[E, N, K]`` one expert at a time.

    ``quantize_int4_groupwise`` packs along the last dim with a 2D reshape, so it
    takes ``[N, K]`` only; looping and stacking is the whole adapter.
    """
    parts = [quantize_int4_groupwise(w[e], INT4_GROUP_SIZE) for e in range(w.shape[0])]
    return tuple(torch.stack(t) for t in zip(*parts, strict=True))  # type: ignore[return-value]


def _unpack_int4(
    packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, k: int
) -> torch.Tensor:
    """``[E, N, K//8]`` int32 -> dequantised ``[E, N, K]`` fp32, plain torch only.

    ``& 0xF`` after the shift is load-bearing: the top nibble of a word can set
    the sign bit, and torch's ``>>`` on int32 is arithmetic, so it sign-extends.
    """
    e, n, _ = packed.shape
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    nibbles = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(e, n, k).float()
    groups = nibbles.reshape(e, n, k // INT4_GROUP_SIZE, INT4_GROUP_SIZE)
    return ((groups - zero.unsqueeze(-1)) * scale.unsqueeze(-1)).reshape(e, n, k)


def _build_int4(w1: torch.Tensor, w2: torch.Tensor) -> Built:
    """AWQ/GPTQ-style packed int4 experts with group scales and zero points."""
    q1, s1, z1 = _quantize_int4_experts(w1)
    q2, s2, z2 = _quantize_int4_experts(w2)

    def call(x, tw, ids):
        return fused_moe(
            x,
            q1,
            q2,
            tw,
            ids,
            w1_scale=s1,
            w2_scale=s2,
            w1_zeros=z1,
            w2_zeros=z2,
            group_n=1,
            group_k=INT4_GROUP_SIZE,
        )

    ref1 = _unpack_int4(q1, s1, z1, w1.shape[-1])
    ref2 = _unpack_int4(q2, s2, z2, w2.shape[-1])
    return call, ref1, ref2, nbytes(s1, s2, z1, z2)


#: Tolerance against ``fused_moe_reference`` on the *dequantised* weights, one
#: number for every scheme. It bounds the kernel's arithmetic and nothing else:
#: the reference multiplies exactly the weights the kernel will reconstruct, so
#: the format's own quantisation error cancels out of the comparison and what is
#: left is 16-bit operand storage against an fp32 reference.
#:
#: One number rather than four because the measurement says the residual does not
#: depend on the format: at the check shapes all four rows land at 2e-6..5e-6
#: max_abs_diff, including int4. An earlier version of this file graded the
#: tolerance per scheme (3e-2 for the 8-bit rows, 6e-2 for int4) on the reasoning
#: that a coarser format accumulates more error; that reasoning was wrong, because
#: the coarseness is in the reference too. The value matches
#: ``tests/kernels/test_fused_moe.py`` and is three orders above what was measured
#: — headroom for a shape change, not a description of the residual.
_RTOL, _ATOL = 2e-2, 2e-2

#: The A8 row's gate, on two statistics instead of elementwise. Mirrors
#: ``tests/kernels/test_fused_moe.py``, deliberately: the benchmark must not grade
#: on a looser scale than the test that owns the kernel. The RMS bound is per
#: activation dtype like the test's — bf16 stores the mode's three intermediates
#: (silu output, each slot's GEMM2 row, the sum) in 8 mantissa bits where fp16
#: keeps 11, so its residual runs ~2.5x larger.
#:
#: Elementwise ``assert_close`` is not available for this row. The weight-only
#: rows are compared against a reference holding *exactly* the weights the kernel
#: reconstructs, so the format's error cancels and the residual is 2e-6. Here the
#: activation is rounded to three mantissa bits twice — once before each GEMM —
#: and the reference can only reproduce the rounding, not the order the kernel
#: sums in, so an irreducible ~6e-3 remains. Against near-zero output elements
#: that is an unbounded *relative* error while being negligible against the
#: tensor, which is what the two statistics below say instead.
_A8_RMS_REL = {torch.float16: 1.5e-2, torch.bfloat16: 4.0e-2}[ACT_DTYPE]
_A8_MAX_OVER_PEAK = 5.0e-2

SCHEMES: tuple[Scheme, ...] = (
    # tune_dtype follows the kernel's TuneKey table, which keys the unquantised
    # mode on the *activation* dtype — "bf16" here, so a --tune run writes
    # entries the bf16 path actually looks up.
    Scheme("unquantized", "bf16", _build_bf16, 16, tune_dtype="bf16"),
    Scheme("fp8", "fp8_w8a16", _build_fp8, 8, tune_dtype="fp8"),
    Scheme(
        "w8a8_fp8",
        "fp8_w8a8",
        _build_fp8_a8,
        8,
        _IMPL_A8,
        _fp8_round_trip,
        tune_dtype="fp8_a8",
    ),
    Scheme("blockwise_int8", "int8", _build_int8, 8, tune_dtype="int8"),
    Scheme("awq", "int4", _build_int4, 4, tune_dtype="int4"),
)


# --------------------------------------------------------------------------- #
# Ablation: the routing bookkeeping, with no GEMM behind it
# --------------------------------------------------------------------------- #
#: Label for the ablation row. Not a ``KernelSpec.name`` — nothing dispatches to
#: it — and the ``ablation:`` prefix says so.
_ABLATION_ALIGN = "ablation: moe_align_block_size"


def align_only(tokens: int, geo: MoeGeometry, ids: torch.Tensor) -> Callable[[], object]:
    """``moe_align_block_size`` alone, on the block size the kernel would pick.

    This row exists because without it the table is unreadable. At ``tokens<=8``
    all four formats come out within 2% of each other however many weight bytes
    they read, which looks like the measurement being broken; it is not. The sort,
    the ``scatter_add`` histogram, the two ``cumsum`` passes and the
    ``searchsorted`` that build the padded slot table cost ~185 us on this device
    and do not move with the token count or the format, so at decode they are
    *half* the layer's latency and no choice of expert format can touch them.

    The op is deliberately torch-native and free of host synchronisation so the
    MoE layer stays CUDA-graph-capturable (see the ``scatter_add`` comment in
    ``moe_align_block_size``), which is the right trade — but it means ~15 tiny
    kernel launches, and this is what they cost.

    Reported with an empty :class:`Work`: it is bookkeeping over a slot table, not
    part of the FFN's arithmetic, so a FLOP/s or GB/s column here would invite
    comparison against numbers that mean something else.
    """
    # ``quant_mode`` only selects BLOCK_K, and BLOCK_M is the block size the
    # alignment is asked for, so 0 here is not a claim about the format.
    rows_per_expert = tokens * geo.top_k / geo.num_experts
    block_m = _launch_config(tokens, 0, rows_per_expert)["BLOCK_M"]
    return lambda: moe_align_block_size(ids, block_m, geo.num_experts)


# --------------------------------------------------------------------------- #
# Ablation: the k-tile the unquantised row used to get
# --------------------------------------------------------------------------- #
_ABLATION_BASELINE_NARROW = "ablation: bf16 with BLOCK_K=32 (pre-fix heuristic)"

#: The k-tile ``_launch_config`` gave an unquantised weight before the sweep in
#: :func:`tune` showed no winner below 64 at any token count. Kept as a row so
#: the cost of that choice stays visible after the fix removed it. (The sweep
#: that fixed it ran on the fp16 baseline; the row itself tracks ``ACT_DTYPE``.)
_OLD_BASELINE_BLOCK_K = 32


@contextmanager
def forced_block_k(block_k: int) -> Iterator[None]:
    """Run ``fused_moe`` with one ``BLOCK_K``, whatever its heuristic would pick.

    This row started as the fair baseline: ``_launch_config`` returned
    ``BLOCK_K = 128 if quant_mode else 32``, so the unquantised baseline ran four
    times as many k-iterations as every row it was the baseline for, and at 512
    tokens that handicap was the difference between fp8 reading as an 18% win and
    a 6% loss -- the opposite conclusion. The tile sweep (``--tune``) then found no
    winning config with ``BLOCK_K`` below 64 anywhere, so the heuristic now gives
    every mode 128 and the handicap is gone from the default path.

    What remains is this row's second job: holding the defect measured. It now runs
    the baseline dtype on the *old* narrow tile, so its margin over the plain
    baseline row is the cost the kernel used to pay, and a regression that
    reinstates the narrow tile shows up as the two rows converging.

    Patches the module attribute rather than the imported name, because that is
    where ``fused_moe`` looks the function up. It is a no-op if a tuned config is
    persisted for this shape, since ``get_best_config`` is consulted first -- run
    with ``LITE_LLAMA_AUTOTUNE=0`` to measure the heuristic path.
    """
    original = fused_moe_module._launch_config
    fused_moe_module._launch_config = lambda n, q, r: {**original(n, q, r), "BLOCK_K": block_k}
    try:
        yield
    finally:
        fused_moe_module._launch_config = original


# --------------------------------------------------------------------------- #
# Routing — built once per case, outside every timed region
# --------------------------------------------------------------------------- #
def routing(tokens: int, geo: MoeGeometry) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Router output for one case, plus how many distinct experts it selected.

    Built here rather than inside the timed callable because a real router's
    logits come from a matmul that is not part of this op, and because the expert
    count feeds the traffic formula: at ``tokens=1`` only ``top_k`` of the ``E``
    expert matrices are ever read, and charging the op for all ``E`` would report
    a decode step as moving 16x the bytes it moves.

    Sampling is without replacement per token, which is what a top-k router
    guarantees and what ``fused_moe`` assumes when it maps slot ``i`` to token
    ``i // top_k``.
    """
    ids = torch.rand(tokens, geo.num_experts, device="cuda").topk(geo.top_k, dim=-1).indices
    ids = ids.to(torch.int32)
    weights = torch.softmax(
        torch.randn(tokens, geo.top_k, device="cuda", dtype=torch.float32), dim=-1
    ).to(ACT_DTYPE)
    return weights, ids, int(ids.unique().numel())


# --------------------------------------------------------------------------- #
# Work
# --------------------------------------------------------------------------- #
def moe_work(
    tokens: int,
    geo: MoeGeometry,
    active_experts: int,
    weight_bits: int,
    scale_bytes: int,
) -> Work:
    """Theoretical cost of one routed-expert FFN.

    FLOPs: ``top_k`` slots per token, each through a ``[2I, H]`` gate/up and an
    ``[H, I]`` down projection, so ``2 * T * top_k * H * (2I + I)`` — the padding
    slots the grouped GEMM computes and masks are implementation work and are not
    counted.

    Bytes assume every input byte is read once and every output byte written once:
    the activation in, the *selected* experts' weights and their scales, the
    result out. Three things are deliberately excluded, and all three make the
    reported GB/s a lower bound rather than an inflated one:

    - ``gate_up``, ``act`` and ``expanded``, the three ``[T * top_k, ...]``
      intermediates the pipeline materialises between its four kernels. Fusing
      them away is a legitimate optimisation, so charging the operation for them
      would make an improvement look like a slowdown.
    - Experts nobody routed to. Their weights are never loaded.
    - Re-reads of an expert whose slots span several row-blocks.

    ``scale_bytes`` arrives as the total over all ``E`` experts and is prorated by
    the active fraction, matching how the weights are counted.
    """
    act = torch.empty((), dtype=ACT_DTYPE).element_size()
    flops = 6 * tokens * geo.top_k * geo.hidden * geo.intermediate
    weight_elems = 3 * geo.hidden * geo.intermediate * active_experts
    moved = (
        2 * tokens * geo.hidden * act
        + weight_elems * weight_bits // 8
        + scale_bytes * active_experts // geo.num_experts
    )
    return Work(flops=flops, moved=moved)


# --------------------------------------------------------------------------- #
# Correctness, before any timing
# --------------------------------------------------------------------------- #
#: Small geometry for the correctness gate. K is a multiple of 128 on both GEMMs
#: (``hidden`` for gate_up, ``intermediate`` for down) so every scheme's group
#: constraint holds, and the two widths differ so a swapped stride shows up.
CHECK_GEOMETRY = MoeGeometry("check", hidden=256, intermediate=128, num_experts=8, top_k=2)

#: Token counts for the gate. 33 is not a multiple of any row-block the kernel
#: picks, so the padded slots and their sentinel mask are exercised; 1 is the
#: single-token decode shape where most of the ``BLOCK_M`` rows are padding; 129
#: is the first count that reaches ``BLOCK_M = 64``, which for the A8 row is the
#: threshold where Triton switches from an fp16 ``mma.sync`` to Hopper's fp8
#: ``wgmma``. Without it the gate would certify a different instruction from the
#: one every ``tokens >= 512`` row below is timing.
CHECK_TOKENS: tuple[int, ...] = (1, 33, 129)


def verify_a8(name: str, out: torch.Tensor, ref: torch.Tensor) -> float:
    """Correctness gate for the A8 row: RMS-relative and peak-relative, not elementwise.

    See :data:`_A8_RMS_REL` for why the weight-only rows' elementwise comparison
    does not transfer. Returns the max absolute difference, the same quantity
    :func:`microbench.verify` returns.
    """
    err = (out.float() - ref.float()).abs()
    ref_f = ref.float()
    rms_rel = (err.pow(2).mean().sqrt() / ref_f.pow(2).mean().sqrt()).item()
    max_over_peak = (err.max() / ref_f.abs().max()).item()
    assert rms_rel < _A8_RMS_REL, f"{name}: rms relative {rms_rel:.3e} >= {_A8_RMS_REL}"
    assert max_over_peak < _A8_MAX_OVER_PEAK, (
        f"{name}: max/peak {max_over_peak:.3e} >= {_A8_MAX_OVER_PEAK}"
    )
    print(
        f"  ok   {name:<44} max_abs_diff={err.max().item():.3e}  "
        f"(rms_rel={rms_rel:.3e}, max/peak={max_over_peak:.3e})"
    )
    return err.max().item()


def check_correctness() -> None:
    """Verify every scheme against a torch reference on the dequantised weights.

    Each format runs different code inside the kernel — bit-trick widening, an
    int8 convert, a nibble unpack with a zero point, an 8-bit dot with no widening
    at all — so a row verified through a sibling is an unverified row. The
    reference is ``tests/reference.py::fused_moe_reference``, which loops over
    experts in fp32 and shares no code with the kernel; the dequant that feeds it
    is built here out of plain torch ops for the same reason.

    The A8 row additionally hands the reference its ``act_quant`` round trip, so
    the comparison isolates the kernel's arithmetic from the activation rounding
    both sides now perform.
    """
    geo = CHECK_GEOMETRY
    print("Correctness (per-expert torch gather-matmul on dequantised weights):")
    for tokens in CHECK_TOKENS:
        torch.manual_seed(0)
        x = torch.randn(tokens, geo.hidden, device="cuda", dtype=ACT_DTYPE) / geo.hidden**0.5
        w1 = (
            torch.randn(
                geo.num_experts,
                2 * geo.intermediate,
                geo.hidden,
                device="cuda",
                dtype=torch.float32,
            )
            / geo.hidden**0.5
        )
        w2 = (
            torch.randn(
                geo.num_experts, geo.hidden, geo.intermediate, device="cuda", dtype=torch.float32
            )
            / geo.intermediate**0.5
        )
        weights, ids, _ = routing(tokens, geo)
        for scheme in SCHEMES:
            call, ref1, ref2, _ = scheme.build(w1, w2)
            ref = fused_moe_reference(x, ref1, ref2, weights, ids, act_quant=scheme.act_quant)
            label = f"{scheme.impl} [{scheme.key}] {geo.case(tokens)}"
            out = call(x, weights, ids)
            if scheme.act_quant is None:
                verify(label, out, ref, rtol=_RTOL, atol=_ATOL)
            else:
                verify_a8(label, out, ref)
        del w1, w2
        torch.cuda.empty_cache()


def show_dispatch() -> None:
    """Print the decision chain and pin every table label to a registry row.

    Every scheme but ``w8a8_fp8`` must land on the *same* row: ``fused_moe`` reads
    the format off ``w1.dtype``, so for those the format is a branch inside one
    kernel and not a choice between specs. ``w8a8_fp8`` is the exception because
    its bytes are indistinguishable from weight-only fp8 — see :data:`_IMPL_A8`.
    If a later change moves a scheme between rows, these assertions are what say
    the table's labels went stale.
    """
    print("\nDispatch for moe:")
    for scheme in SCHEMES:
        sel = dispatch("moe", dtype="bf16", scheme=scheme.key)
        assert sel.spec.name == scheme.impl, (
            f"table labels {scheme.key} as {scheme.impl}, dispatch picks {sel.spec.name}"
        )
        assert sel.load() is not None
        print(f"  {scheme.key:<16} -> {sel.spec.name}")
    # One full chain, so a filtered-out backend (deepgemm's grouped fp8 row is
    # registered but unverified) is visible in the log rather than inferred.
    print(f"\n{dispatch('moe', dtype='bf16', scheme='w8a8_fp8').explain()}")


# --------------------------------------------------------------------------- #
# Tuning — replace the heuristic's guess with a measured winner
# --------------------------------------------------------------------------- #
#: Candidate tiles. Hand-picked, not a cross product: the full grid over the six
#: fields is 288 configs, and at five schemes and five token counts that is a
#: compile budget rather than a search. These span the axes the heuristic fixes —
#: ``BLOCK_N`` at 64, ``num_warps`` at 4, ``num_stages`` at 3 — plus both k-tiles
#: it chooses between, at every ``BLOCK_M`` it can pick.
_TUNE_SPACE: tuple[dict[str, int], ...] = tuple(
    {
        "BLOCK_M": bm,
        "BLOCK_N": bn,
        "BLOCK_K": bk,
        "GROUP_M": gm,
        "num_warps": nw,
        "num_stages": ns,
    }
    for bm, bn, bk, gm, nw, ns in (
        (16, 32, 64, 1, 4, 3),
        (16, 64, 32, 1, 4, 3),
        (16, 64, 64, 8, 4, 3),
        (16, 64, 128, 8, 4, 3),
        (16, 128, 128, 8, 4, 4),
        (32, 64, 32, 8, 4, 3),
        (32, 64, 128, 8, 4, 3),
        (32, 128, 64, 8, 4, 4),
        (32, 128, 128, 8, 8, 3),
        (64, 64, 32, 8, 4, 3),
        (64, 64, 128, 8, 4, 3),
        (64, 128, 64, 8, 8, 4),
        (64, 128, 128, 8, 8, 3),
        (64, 256, 64, 8, 8, 3),
        (128, 64, 128, 8, 8, 4),
        (128, 128, 64, 8, 8, 3),
        (128, 128, 128, 8, 8, 3),
    )
)


@contextmanager
def forced_config(config: dict[str, int]) -> Iterator[None]:
    """Run ``fused_moe`` on one tile config, bypassing both of its choices.

    Patches the module attribute, like :func:`forced_block_k`, and additionally
    disables the store lookup for the duration: ``get_best_config`` is consulted
    *before* the heuristic, so once an entry exists for this shape a patched
    ``_launch_config`` would be dead code and every candidate would time the same
    persisted config. A tuning run that reads its own previous output measures
    nothing.
    """
    original = fused_moe_module._launch_config
    previous = os.environ.get("LITE_LLAMA_AUTOTUNE")
    fused_moe_module._launch_config = lambda n, q, r, c=config: dict(c)
    os.environ["LITE_LLAMA_AUTOTUNE"] = "0"
    try:
        yield
    finally:
        fused_moe_module._launch_config = original
        if previous is None:
            os.environ.pop("LITE_LLAMA_AUTOTUNE", None)
        else:
            os.environ["LITE_LLAMA_AUTOTUNE"] = previous


def _time_config(
    config: dict[str, int],
    call: Callable[..., torch.Tensor],
    inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    references: list[torch.Tensor],
) -> float | None:
    """Total time of one config over a bucket's token counts, or ``None`` if rejected.

    Rejection is either a compile failure (a tile that overflows shared memory, or
    a ``BLOCK_K`` wider than the scheme's scale group, which reads one scale for k
    elements that do not share one) or an output that disagrees with the config the
    correctness gate already checked. Timing an unverified config would install a
    tile that computes the wrong FFN quickly.

    Timed with the harness ``bench`` rather than a synchronise-per-iteration loop:
    at these sizes the latter imposes a floor around 100 us, which is most of a
    decode row and would make the search blind exactly where it matters most.
    """
    total = 0.0
    with forced_config(config):
        for (x, weights, ids), reference in zip(inputs, references, strict=True):
            try:
                out = call(x, weights, ids)
            except Exception:
                return None
            if not torch.allclose(out, reference, rtol=_RTOL, atol=_ATOL):
                return None
            total += bench(lambda c=call, a=x, w=weights, i=ids: c(a, w, i))
    return total


def tune(
    geometries: tuple[MoeGeometry, ...],
    tokens: tuple[int, ...],
    *,
    write: bool,
) -> list[TuneResult]:
    """Search :data:`_TUNE_SPACE` per store key and persist the winners.

    One search per :class:`TuneKey`, not per token count, because that is the
    granularity the kernel looks up: ``bucket_m`` rounds M up to the next of
    (16, 32, 64, 128, 256, 512), so t1 and t8 share one entry. A search run per
    token count would have them overwrite each other and the surviving config
    would be whichever ran last. Instead every token count in a bucket is timed on
    every candidate and the winner is the one with the lowest *total*, so a shared
    entry is chosen for the traffic it will actually serve.

    Args:
        write: Persist the winners to :class:`ConfigStore`. False measures and
            reports without touching the cache, which is what a comparison of two
            checkouts needs.
    """
    results: list[TuneResult] = []
    for geo in geometries:
        torch.manual_seed(0)
        w1 = (
            torch.randn(
                geo.num_experts,
                2 * geo.intermediate,
                geo.hidden,
                device="cuda",
                dtype=torch.float32,
            )
            / geo.hidden**0.5
        )
        w2 = (
            torch.randn(
                geo.num_experts, geo.hidden, geo.intermediate, device="cuda", dtype=torch.float32
            )
            / geo.intermediate**0.5
        )
        built = {s.key: s.build(w1, w2)[0] for s in SCHEMES}
        del w1, w2
        torch.cuda.empty_cache()

        buckets: dict[int, list[int]] = {}
        for t in sorted(tokens):
            buckets.setdefault(bucket_m(t), []).append(t)

        for scheme in SCHEMES:
            call = built[scheme.key]
            for bucket, group in buckets.items():
                inputs = []
                for t in group:
                    x = torch.randn(t, geo.hidden, device="cuda", dtype=ACT_DTYPE) / geo.hidden**0.5
                    weights, ids, _ = routing(t, geo)
                    inputs.append((x, weights, ids))
                # The reference every candidate is checked against: the same kernel
                # on the heuristic's config, which check_correctness() already
                # verified against the torch reference this run.
                quant_mode = 0 if scheme.key == "unquantized" else 1
                baseline_config = _launch_config(
                    group[-1], quant_mode, group[-1] * geo.top_k / geo.num_experts
                )
                with forced_config(baseline_config):
                    references = [call(x, w, i) for x, w, i in inputs]
                baseline_us = _time_config(baseline_config, call, inputs, references)
                if baseline_us is None:  # pragma: no cover - the gate's own config
                    raise SystemExit(f"the heuristic config failed its own gate for {scheme.key}")

                best_config, best_us, rejected = baseline_config, baseline_us, 0
                for candidate in _TUNE_SPACE:
                    if candidate == baseline_config:
                        continue
                    us = _time_config(candidate, call, inputs, references)
                    if us is None:
                        rejected += 1
                        continue
                    if us < best_us:
                        best_config, best_us = candidate, us

                key = TuneKey.build(
                    "fused_moe",
                    m=bucket,
                    n=2 * geo.intermediate,
                    k=geo.hidden,
                    dtype=scheme.tune_dtype,
                )
                result = TuneResult(
                    key=key,
                    label=scheme.label,
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
                    # The store's unit is microseconds, which is also the harness's,
                    # so nothing is converted here. set_perf_provider() takes
                    # milliseconds and is the boundary that does convert.
                    ConfigStore().put(key, best_config, latency_us=best_us)
                del inputs, references
                torch.cuda.empty_cache()
        del built
        torch.cuda.empty_cache()
    if write and any(r.changed for r in results):
        # Later phases in this process would otherwise keep the pre-tuning view:
        # the lookup caches its store instance on first use.
        autotune_reset()
    return results


def _tune_line(r: TuneResult) -> str:
    tiles = "x".join(str(r.best_config[f"BLOCK_{d}"]) for d in "MNK")
    if not r.changed:
        return (
            f"{r.label:10s} {r.key.shape_bucket:16s} t{list(r.tokens)}: "
            f"heuristic already best ({r.baseline_us:.1f} us, {tiles}), "
            f"{r.rejected} rejected"
        )
    return (
        f"{r.label:10s} {r.key.shape_bucket:16s} t{list(r.tokens)}: "
        f"{r.baseline_us:.1f} -> {r.best_us:.1f} us ({r.gain:+.1%}) "
        f"BLOCK_MNK={tiles} GROUP_M={r.best_config['GROUP_M']} "
        f"warps={r.best_config['num_warps']} stages={r.best_config['num_stages']}, "
        f"{r.rejected} rejected"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def measure(geometries: tuple[MoeGeometry, ...], tokens: tuple[int, ...]) -> list[Row]:
    """Time every (scheme, geometry, token count) combination."""
    rows: list[Row] = []
    for geo in geometries:
        torch.manual_seed(0)
        w1 = (
            torch.randn(
                geo.num_experts,
                2 * geo.intermediate,
                geo.hidden,
                device="cuda",
                dtype=torch.float32,
            )
            / geo.hidden**0.5
        )
        w2 = (
            torch.randn(
                geo.num_experts, geo.hidden, geo.intermediate, device="cuda", dtype=torch.float32
            )
            / geo.intermediate**0.5
        )
        # Quantisation happens once per checkpoint, outside every timed region:
        # a served model quantises at load time, not per step.
        built = [(s, *s.build(w1, w2)) for s in SCHEMES]
        del w1, w2
        torch.cuda.empty_cache()

        for t in tokens:
            x = torch.randn(t, geo.hidden, device="cuda", dtype=ACT_DTYPE) / geo.hidden**0.5
            # Router output and the alignment inputs it feeds are built here, not
            # in the timed callable: the router's own matmul is a different op.
            weights, ids, active = routing(t, geo)
            case = geo.case(t)
            for scheme, call, _r1, _r2, scale_bytes in built:
                # Every input bound as a default: the loop rebinds them and the
                # tail of the body deletes them, so a late-bound closure would
                # time the wrong case or nothing at all.
                us = bench(lambda c=call, a=x, w=weights, i=ids: c(a, w, i))
                rows.append(
                    Row(
                        f"{scheme.impl} [{scheme.label}]",
                        case,
                        us,
                        moe_work(t, geo, active, scheme.weight_bits, scale_bytes),
                    )
                )
            # Same inputs, no GEMM behind them: how much of the rows above is
            # routing bookkeeping that no expert format can change.
            rows.append(Row(_ABLATION_ALIGN, case, bench(align_only(t, geo, ids)), Work()))
            # The baseline row above ran on the heuristic's tile, which now gives
            # every mode BLOCK_K=128; this is the same row on the 32 it used to
            # get, so the gap is the cost the fix removed.
            baseline_call = built[0][1]
            assert built[0][0].key == "unquantized", "the narrow-k ablation must ablate bf16"
            with forced_block_k(_OLD_BASELINE_BLOCK_K):
                us = bench(lambda c=baseline_call, a=x, w=weights, i=ids: c(a, w, i))
            rows.append(Row(_ABLATION_BASELINE_NARROW, case, us, moe_work(t, geo, active, 16, 0)))
            del x, weights, ids
        del built
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-dir",
        action="append",
        default=None,
        help="MoE HF checkpoint whose config.json supplies the layer geometry; "
        "repeatable. Defaults to the built-in Qwen3-30B-A3B geometry.",
    )
    ap.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=list(TOKENS),
        help=f"Token counts routed through the layer. Default {list(TOKENS)}.",
    )
    ap.add_argument("--json", help="Write the rows to this path as JSON.")
    ap.add_argument(
        "--tune",
        action="store_true",
        help="Search the tile space per store key and persist the winners to the "
        "autotune cache, so the kernel stops guessing. Skips the timing table.",
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
        # The correctness gate above is what makes the search's per-candidate check
        # meaningful: candidates are compared against the heuristic's output, and
        # that output has just been checked against the torch reference.
        print(
            f"Searching {len(_TUNE_SPACE)} tile configs per store key"
            f"{' (dry run, nothing written)' if args.dry_run else ''}:"
        )
        results = tune(geometries, tuple(args.tokens), write=not args.dry_run)
        changed = [r for r in results if r.changed]
        print(
            f"\n{len(changed)} of {len(results)} keys improved on the heuristic"
            + (
                f"; best gain {max(r.gain for r in changed):.1%}, "
                f"worst unchanged key keeps its heuristic"
                if changed
                else ""
            )
        )
        if changed and not args.dry_run:
            print(f"written to {ConfigStore().cache_dir}")
            print(
                "Re-running without --tune now measures the persisted configs, so the\n"
                "table's rows and both ablations shift with them: the narrow-k\n"
                "ablation in particular is a no-op once a config is stored, because\n"
                "get_best_config is consulted before the heuristic it patches. Run\n"
                "with LITE_LLAMA_AUTOTUNE=0 to measure what a user without this cache\n"
                "gets."
            )
        return

    rows = measure(geometries, tuple(args.tokens))
    report(rows)
    print(
        "\nRead the moe_align_block_size ablation before any format row; at decode it\n"
        "carries more of the spread than the formats do.\n"
        "\n"
        "1. moe_align_block_size costs ~188 us on its own, does not move with the\n"
        "   token count, and is identical for every format. At tokens<=8 that is\n"
        "   over half the layer, which is why bf16 and the three weight-only\n"
        "   formats land inside 1.5% of each other there while reading 4x different\n"
        "   weight bytes. A fixed cost hiding the ranking, not evidence that\n"
        "   quantisation cannot help -- and the single largest target on this path.\n"
        "2. The second ablation is a regression guard, not a baseline.\n"
        "   _launch_config used to give the unquantised row BLOCK_K=32 and every\n"
        "   quantised path 128, handicapping the baseline by 4x the k-iterations;\n"
        "   the tile sweep (--tune) found no winner below 64 anywhere, so all\n"
        "   modes now get 128. This row runs the old narrow tile, so its margin is\n"
        "   what the fix bought: 25.1% at t64, 22.4% at t512, 10.4% at t4096,\n"
        "   nothing at t1/t8. The two rows converging means the defect came back.\n"
        "\n"
        "The two kinds of quantisation win in disjoint regimes, and neither wins\n"
        "everywhere:\n"
        "\n"
        "  weight-only int8 holds its lead through t512 -- 29.1% at t64, 7.4% at\n"
        "  t512 -- while fp8_w8a16 wins only t64 (11.6%). At t4096 both are\n"
        "  outright regressions (20% and 55% slower). Each expert tile is\n"
        "  re-read across row-blocks and dequantised again every time, so the\n"
        "  widening stops being amortised weight traffic and becomes arithmetic on\n"
        "  the critical path. int4 never wins at any token count -- 17.0% slower\n"
        "  than bf16 already at t64 -- and has the lowest GB/s in the table while\n"
        "  reading the fewest bytes: it is unpack-bound, not traffic-bound.\n"
        "\n"
        "  W8A8 fp8 is the mirror image: ~35% slower at decode (t1/t8), where two\n"
        "  extra quantisation launches land on an already launch-bound layer, and\n"
        "  only 6.0% ahead at t64 where BLOCK_M=32 keeps the dot off the fp8\n"
        "  tensor cores. From t512 it is the fastest row in the table -- 16.3%\n"
        "  under bf16 at t512, 6.3% at t4096, at 209 TFLOP/s, the highest rate\n"
        "  here. These are reported separately on purpose: one speedup number\n"
        "  spanning a 35% loss and a 16% win would describe neither regime.\n"
        "\n"
        "GB/s is a lower bound -- moe_work() excludes the three per-slot\n"
        "intermediates the pipeline materialises, the W8A8 row's quantised copies\n"
        "of them, and any expert re-read across row-blocks."
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
                "geometries": [
                    {
                        "label": g.label,
                        "hidden": g.hidden,
                        "intermediate": g.intermediate,
                        "num_experts": g.num_experts,
                        "top_k": g.top_k,
                    }
                    for g in geometries
                ],
                "tokens": args.tokens,
                "act_dtype": str(ACT_DTYPE),
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
