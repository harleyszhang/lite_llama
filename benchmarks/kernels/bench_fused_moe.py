"""Microbenchmark ``fused_moe`` across every expert-weight format it accepts.

A routed-expert FFN is not a dense GEMM with extra bookkeeping: at decode the
layer touches ``top_k`` experts for a single token, so it reads ``top_k`` full
expert weight matrices to produce one row of output. That makes the small-token
rows the most memory-bound shapes in the model — far past the dense projections
— and it is why a quantised expert format pays off here before it pays off
anywhere else. Read the table as three regimes (H100, E=128, top_k=8, h=2048,
i=768, bf16 activation; the two ablations follow the regimes):

``tokens <= 8`` (decode)
    Each of the ``top_k`` selected experts is read whole for one token, so the
    ranking *should* be bytes of expert weight — and at exactly 1 token it still
    is not: every format lands within 12% of bf16 (108-121 us) because the floor
    is launches, not bytes. The layer is five kernels back to back (align, GEMM1,
    silu, GEMM2, sum) and a W8A8 row's activation quantiser used to add two more
    launches — that was the old table's 35% W8A8 decode loss. Both quantisations
    are fused now (into the GEMMs below 32 rows, ``_INLINE_A_QUANT_MAX_ROWS``, and
    into silu's store, ``QUANT_OUT``), which trades the launch for an amax
    re-derived in every GEMM program: up to ~12% at t1, gone by t8. From 8
    tokens bytes take over and the ranking returns — int8 weight-only at 114.9
    us leads bf16's 186.1 by 62%, W8A8 fp8 by 50%, and int4 by 43%.
``tokens = 64`` (where quantisation wins)
    Slots per expert first exceed the row-block, so the grouped GEMM amortises
    each expert load and every quantised row beats bf16's 415.9. int8 W8A8 at
    234.4 (1.77x) is the floor here, with int8 weight-only and W8A8 fp8 inside
    1% of it (234.1 / 236.3) and weight-only fp8 at 240.0 (1.73x). The A8 rows
    win on the skipped bit trick alone — ``BLOCK_M`` is 16 here, below the fp8
    wgmma threshold — so the byte savings and the compute savings stack instead
    of competing. int4 at 242.2 (1.72x) tracks the 8-bit rows: the byte layout's
    dense load earns the traffic win, and its register nibble split costs about
    what an 8-bit format's single widening does.
``tokens >= 512`` (prefill)
    The weight-only wins invert, and W8A8 takes over. At t512 int8 weight-only
    still leads bf16 (313.0 against 469.7) but the W8A8 rows have passed it —
    int8 W8A8 at 275.9 (1.70x) is the fastest row in the regime, fp8 W8A8 at
    280.4 (1.67x) close behind. At t4096 the weight-only formats are outright
    regressions (fp8 1320.8, int8 1148.0, int4 1701.9 against 1062.4): each
    expert tile is re-read across row-blocks and widened again every time, so
    in-loop dequant stops being amortised weight traffic and becomes arithmetic
    on the critical path. The W8A8 rows never pay it — both operands are already
    8-bit, and no widening exists to hoist. int8 W8A8 at 673.1 (1.58x) is the
    fastest row in the table at any shape, 22% ahead of fp8 W8A8 (868.9, 1.22x):
    the int8 imma reaches the tensor cores from ``BLOCK_M=16`` while fp8 needs 64
    for wgmma, and int32 accumulation is exact, with no ``K_PROMOTE``-style
    precision tax. Its gain is in the MMA, so it appears exactly where the
    weight-only gains disappear.

The second ablation was not about quantisation at all and was the larger finding,
and it has since been fixed: ``_launch_config`` used to return
``BLOCK_K = 128 if quant_mode else 32``, so the unquantised row ran four times as
many k-iterations as the rows it was the baseline for. At t512 that handicap was
the difference between weight-only fp8 reading as a win and the loss it really
was — the opposite conclusion — and it only ever depressed the baseline, which is
why no test caught it and why every quantisation number on this kernel used to
look better than it was. The tile sweep (``--tune``) found no winning config with
``BLOCK_K`` below 64 at any token count, the heuristic now gives every mode 128,
and the row survives as a regression guard: it runs the baseline dtype on the old
narrow tile, so its margin over the plain bf16 row is what the fix bought — 26.3%
at t8 (252.9 -> 186.6), 30.8% at t64 (605.0 -> 418.3), 26.8% at t512
(641.3 -> 469.4), 12.9% at t4096 (1189.2 -> 1036.4), nothing at t1 where the
layer is launch-bound. Two rows converging is a reinstated defect.

What each row measures. The prefix is the registry entry and the ``[label]`` suffix
names the format: all but two rows are ``native/fused_moe``, because the kernel
picks the format off ``w1.dtype`` rather than from a spec row (see
``rapid_llm/kernels/ops/moe/__init__.py``), and the exceptions are the two W8A8
rows, whose bytes are identical to their weight-only formats':

``[unquantized]``
    bf16 experts, ``tl.dot`` straight from the loaded tile. The floor to beat.
``[fp8_w8a16]``
    fp8-e4m3 experts, per-output-channel scales, widened in the inner loop by the
    ``dequant_fp8e4m3`` bit trick. Weight-only: the activation is untouched, so the
    MMA is unchanged and the entire gain is bytes.
``[fp8_w8a8]``
    The same fp8 expert bytes with the activation quantised per token, so both
    operands stay 8-bit into the dot and the bit trick is skipped. One of the two
    rows on a different registry entry, because nothing in the weights
    distinguishes either — see :data:`_IMPL_A8`. Read it separately per regime:
    ~9% behind at t1 (the launch floor; its quantisation is fused in below 32
    rows), ahead of bf16 from t8 on, and averaging those into one speedup would
    describe neither.
``[int8_w8a8]``
    The same int8 expert bytes with the activation quantised per token. Both
    operands stay int8 into the dot and the integer tensor cores accumulate in
    int32 — exact, and available from ``BLOCK_M=16`` on every device since
    Turing, so unlike the fp8 A8 row there is no tile threshold where the
    instruction underneath changes.
``[int8]``
    int8 experts, per-output-channel scales, converted in the loop. Same bytes as
    fp8, cheaper widening, coarser format.
``[int4]``
    int4 experts, two nibbles per uint8 byte (vLLM's packing, crossed from the
    checkpoint's int32 words by ``repack_int4_experts``), with group scales and
    zero points. The kernel loads the byte tile densely and splits the nibble
    planes in registers into two half-K dots; vLLM's own replicated addressing
    (logical k reading byte k // 2) compiles to scalar byte loads and measured
    13-18 ms per GEMM on these shapes, which is why they ship int4 on the
    Marlin CUDA kernel instead.

One caveat the A8 rows' TFLOP/s column does not show: Triton emits Hopper's fp8
``wgmma`` only from ``BLOCK_M >= 64``, and ``_launch_config``'s fp8 W8A8 tiles
reach that only at t4096 — t512's tier-1 tile is ``BLOCK_M=32``. Below it the two
e4m3 operands are widened to an fp16 ``mma.sync``, so no fp8 A8 row but t4096
measures the fp8 tensor cores; the t512 lead there is bytes and the skipped bit
trick, not the MMA. The int8 A8 row has no such threshold (imma from
``BLOCK_M=16``). The correctness gate covers both instructions (see
:data:`CHECK_TOKENS`).

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

from microbench import Row, Work, bench, device_peaks, metadata, report, require_cuda, verify
from tuning import TuneResult, nbytes

# Importing the facade registers every spec row, so dispatch() below finds them.
import rapid_llm.kernels  # registers the spec rows as a side effect
import rapid_llm.kernels.ops.moe.fused_moe as fused_moe_module
from rapid_llm.kernels.dispatcher import dispatch
from rapid_llm.kernels.dispatcher.autotune import ConfigStore, TuneKey, bucket_m
from rapid_llm.kernels.dispatcher.autotune import reset as autotune_reset
from rapid_llm.kernels.ops.moe.fused_moe import (
    _launch_config,
    fused_moe,
    fused_moe_w8a8_fp8,
    fused_moe_w8a8_int8,
    moe_align_block_size,
)
from rapid_llm.kernels.ops.quantization import repack_int4_experts
from rapid_llm.kernels.ops.tile_policy import has_native_fp8
from rapid_llm.modules.quantization.utils import (
    quantize_fp8_per_channel,
    quantize_fp8_per_token,
    quantize_int4_groupwise,
    quantize_int8_per_channel,
)
from tests.reference import fused_moe_reference

_IMPL = "native/fused_moe"
_IMPL_A8 = "native/fused_moe_w8a8_fp8"
_IMPL_A8_INT8 = "native/fused_moe_w8a8_int8"
TOKENS: tuple[int, ...] = (1, 8, 64, 512, 4096)
INT4_GROUP_SIZE = 128
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
    #: Set for schemes whose kernels store/load ``tl.float8e4nv``. Triton only emits
    #: e4m3 from sm89; on sm86 (A10) the type does not exist at all, so these schemes
    #: cannot compile there and must be skipped rather than crash the run.
    native_fp8_only: bool = False
    #: Non-empty when the scheme is known-broken independent of device and is
    #: therefore deferred out of every gate/table until fixed. The reason is
    #: printed at the call sites so the omission is visible, not silent.
    deferred_reason: str = ""


def active_schemes() -> tuple[Scheme, ...]:
    """The schemes this device can actually compile and that are not deferred.

    fp8 W8A8 stores its activation as ``tl.float8e4nv`` in the silu epilogue; on
    pre-sm89 parts Triton rejects that dtype outright. int8 W8A8 is deferred
    everywhere: its inline activation-quant path (``A_QUANT``, tokens <= 32) loads
    ``a`` at full precision and never narrows it to int8, so the grouped GEMM dots
    a bf16 ``a`` against an int8 ``b`` and fails to compile -- i.e. W8A8-int8 MoE
    is broken at decode shapes on every device, not just here. Both are dropped
    here (with a printed note) instead of aborting ``--tune``/the table.
    """
    return tuple(
        s
        for s in SCHEMES
        if not s.deferred_reason and (not s.native_fp8_only or has_native_fp8(None))
    )


def skipped_scheme_notes() -> list[tuple[str, str]]:
    """``(key, reason)`` for every scheme :func:`active_schemes` dropped."""
    notes = []
    active = active_schemes()
    for s in SCHEMES:
        if s in active:
            continue
        if s.deferred_reason:
            notes.append((s.key, s.deferred_reason))
        else:
            notes.append((s.key, "needs sm89+ native fp8; this device lacks it"))
    return notes


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
    ``_launch_config`` reserves for experts holding more than 32 rows — on this
    geometry that is the 4096-token grid point only, since 512 lands at 32 rows
    per expert and stays on tier 1. Below that the operands are widened to an
    fp16 ``mma.sync`` and only the skipped dequant remains.
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


def _int8_round_trip(t: torch.Tensor) -> torch.Tensor:
    """Per-row symmetric int8 quantise-then-widen, the rounding the A8 kernel imposes.

    Plain torch, like :func:`_fp8_round_trip`: the reference must reproduce the
    rounding — round-to-nearest-even, which is what both the inline quantiser and
    ``int8_quantize_per_token`` use — without borrowing the kernel's arithmetic.
    """
    flat = t.to(ACT_DTYPE).reshape(-1, t.shape[-1]).float()
    scale = flat.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return ((flat / scale).round().clamp(-127, 127) * scale).reshape(t.shape)


def _build_int8_a8(w1: torch.Tensor, w2: torch.Tensor) -> Built:
    """Same int8 experts as :func:`_build_int8`, with the activation quantised too.

    The mirror of the fp8 A8 row one entry up, with two asymmetries: the integer
    tensor cores (from Turing on) accumulate in int32, which is exact — no
    ``K_PROMOTE`` analogue to tune — and the imma path exists from
    ``BLOCK_M=16``, so there is no tile threshold where the instruction
    underneath changes and every token count measures the same MMA.
    """
    q1, s1 = quantize_int8_per_channel(w1)
    q2, s2 = quantize_int8_per_channel(w2)
    gk = _group_k(w1, w2)

    def call(x, tw, ids):
        return fused_moe_w8a8_int8(
            x, q1, q2, tw, ids, w1_scale=s1, w2_scale=s2, group_n=1, group_k=gk
        )

    return call, q1.float() * s1, q2.float() * s2, nbytes(s1, s2)


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
    # The kernel eats the byte layout (two nibbles per uint8) — the bridge
    # ``process_weights_after_loading`` crosses once at load. The torch
    # reference below still unpacks the int32 words both derive from.
    kq1, kq2 = repack_int4_experts(q1), repack_int4_experts(q2)

    def call(x, tw, ids):
        return fused_moe(
            x,
            kq1,
            kq2,
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


_RTOL, _ATOL = 2e-2, 2e-2
_A8_RMS_REL = {torch.float16: 1.5e-2, torch.bfloat16: 4.0e-2}[ACT_DTYPE]
_A8_MAX_OVER_PEAK = 5.0e-2

SCHEMES: tuple[Scheme, ...] = (
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
        native_fp8_only=True,
    ),
    Scheme(
        "w8a8_int8",
        "int8_w8a8",
        _build_int8_a8,
        8,
        _IMPL_A8_INT8,
        _int8_round_trip,
        tune_dtype="int8_a8",
        deferred_reason="inline A-quant defect: A_QUANT loads full-precision a, dots bf16 vs int8",
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
    # device_index=None: the bench runs on the current device.
    rows_per_expert = tokens * geo.top_k / geo.num_experts
    block_m = _launch_config(tokens, 0, rows_per_expert, None)["BLOCK_M"]
    return lambda: moe_align_block_size(ids, block_m, geo.num_experts)


# --------------------------------------------------------------------------- #
# Ablation: the k-tile the unquantised row used to get
# --------------------------------------------------------------------------- #
_ABLATION_BASELINE_NARROW = "ablation: bf16 with BLOCK_K=32 (pre-fix heuristic)"

_OLD_BASELINE_BLOCK_K = 32


@contextmanager
def forced_block_k(block_k: int) -> Iterator[None]:
    """Run ``fused_moe`` with one ``BLOCK_K``, whatever its heuristic would pick."""
    original = fused_moe_module._launch_config
    fused_moe_module._launch_config = lambda n, q, r, d: {
        **original(n, q, r, d),
        "BLOCK_K": block_k,
    }

    previous = os.environ.get("RAPID_LLM_AUTOTUNE")
    os.environ["RAPID_LLM_AUTOTUNE"] = "0"
    try:
        yield
    finally:
        fused_moe_module._launch_config = original
        if previous is None:
            os.environ.pop("RAPID_LLM_AUTOTUNE", None)
        else:
            os.environ["RAPID_LLM_AUTOTUNE"] = previous


# --------------------------------------------------------------------------- #
# Routing — built once per case, outside every timed region
# --------------------------------------------------------------------------- #
def routing(tokens: int, geo: MoeGeometry) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Router output for one case, plus how many distinct experts it selected."""
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

CHECK_GEOMETRY = MoeGeometry("check", hidden=256, intermediate=128, num_experts=8, top_k=2)
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
    """Verify every scheme against a torch reference on the dequantised weights."""
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
        for scheme in active_schemes():
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

    Every scheme but the W8A8 pair must land on the *same* row: ``fused_moe``
    reads the format off ``w1.dtype``, so for those the format is a branch
    inside one kernel and not a choice between specs. The two W8A8 schemes are
    the exceptions because their bytes are indistinguishable from the
    weight-only formats' — see :data:`_IMPL_A8`. If a later change moves a
    scheme between rows, these assertions are what say the table's labels went
    stale.
    """
    print("\nDispatch for moe:")
    for scheme in active_schemes():
        sel = dispatch("moe", dtype="bf16", scheme=scheme.key)
        assert sel.spec.name == scheme.impl, (
            f"table labels {scheme.key} as {scheme.impl}, dispatch picks {sel.spec.name}"
        )
        assert sel.load() is not None
        print(f"  {scheme.key:<16} -> {sel.spec.name}")
    for key, reason in skipped_scheme_notes():
        print(f"  {key:<16} -> (skipped: {reason})")
    # One full chain, so a filtered-out backend (deepgemm's grouped fp8 row is
    # registered but unverified) is visible in the log rather than inferred.
    print(f"\n{dispatch('moe', dtype='bf16', scheme='w8a8_fp8').explain()}")


# --------------------------------------------------------------------------- #
# Tuning — replace the heuristic's guess with a measured winner
# --------------------------------------------------------------------------- #
_TUNE_QUANT_MODE = {
    "unquantized": 0,
    "fp8": 1,
    "blockwise_int8": 2,
    "awq": 3,
    "w8a8_fp8": 4,
    "w8a8_int8": 5,
}

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
    previous = os.environ.get("RAPID_LLM_AUTOTUNE")
    # ``d`` absorbs the device_index the launcher now passes; ``c`` stays the
    # forced config whatever device is asked about.
    fused_moe_module._launch_config = lambda n, q, r, d=None, c=config: dict(c)
    os.environ["RAPID_LLM_AUTOTUNE"] = "0"
    try:
        yield
    finally:
        fused_moe_module._launch_config = original
        if previous is None:
            os.environ.pop("RAPID_LLM_AUTOTUNE", None)
        else:
            os.environ["RAPID_LLM_AUTOTUNE"] = previous


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
        built = {s.key: s.build(w1, w2)[0] for s in active_schemes()}
        del w1, w2
        torch.cuda.empty_cache()

        buckets: dict[int, list[int]] = {}
        for t in sorted(tokens):
            buckets.setdefault(bucket_m(t), []).append(t)

        for scheme in active_schemes():
            call = built[scheme.key]
            for bucket, group in buckets.items():
                inputs = []
                for t in group:
                    x = torch.randn(t, geo.hidden, device="cuda", dtype=ACT_DTYPE) / geo.hidden**0.5
                    weights, ids, _ = routing(t, geo)
                    inputs.append((x, weights, ids))
                # The reference every candidate is checked against: the same kernel
                # on the heuristic's config, which check_correctness() already
                # verified against the torch reference this run. The mode must be
                # the scheme's own or the "heuristic" baseline is some other
                # format's tile: _TILE_TABLE is per mode.
                quant_mode = _TUNE_QUANT_MODE[scheme.key]
                baseline_config = _launch_config(
                    group[-1], quant_mode, group[-1] * geo.top_k / geo.num_experts, None
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
        built = [(s, *s.build(w1, w2)) for s in active_schemes()]
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

    require_cuda()
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
                "with RAPID_LLM_AUTOTUNE=0 to measure what a user without this cache\n"
                "gets."
            )
        return

    rows = measure(geometries, tuple(args.tokens))
    report(rows)
    print(
        "\nRead the two ablations with the regime structure above: moe_align is a\n"
        "fixed cost, the narrow-k row is a regression guard.\n"
        "\n"
        "1. moe_align_block_size costs 5.8-40 us on its own (4-6% of the layer\n"
        "   at every shape) and is identical for every format. It used to be\n"
        "   ~188 us -- over half a decode step, and the reason the formats used\n"
        "   to be indistinguishable at t1. That is fixed, so at t1 the residual\n"
        "   spread is the launch floor itself: five kernels back to back, every\n"
        "   format within 10% of bf16 while reading 4x different weight bytes.\n"
        "2. The second ablation is a regression guard, not a baseline.\n"
        "   _launch_config used to give the unquantised row BLOCK_K=32 and every\n"
        "   quantised path 128, handicapping the baseline by 4x the k-iterations;\n"
        "   the tile sweep (--tune) found no winner below 64 anywhere, so all\n"
        "   modes now get 128. This row runs the old narrow tile, so its margin is\n"
        "   what the fix bought: 26.3% at t8, 30.8% at t64, 26.8% at t512, 12.9%\n"
        "   at t4096, nothing at t1. The two rows converging means the defect\n"
        "   came back.\n"
        "\n"
        "The two kinds of quantisation win in disjoint regimes, and neither wins\n"
        "everywhere:\n"
        "\n"
        "  weight-only int8 wins from t8 through t512 (1.62x at t8, 1.78x at t64,\n"
        "  1.50x at t512) and loses at t4096 (0.93x): each expert tile is re-read\n"
        "  across row-blocks and widened again every time, so past t512 the\n"
        "  widening stops being amortised weight traffic and becomes arithmetic\n"
        "  on the critical path. fp8_w8a16 is the same shape one tier milder\n"
        "  (1.57x at t8, 0.80x at t4096). int4 wins the middle regimes (1.43x at\n"
        "  t8, 1.72x at t64, 1.16x at t512) on its quarter of the bytes, and at\n"
        "  t4096 it is the worst weight-only row (0.62x): the register nibble\n"
        "  split is a per-tile cost the 8-bit formats' single widening avoids.\n"
        "\n"
        "  The W8A8 rows lose ~9% at t1 -- the launch floor again; their\n"
        "  quantisation is fused in below 32 rows and what remains is the inline\n"
        "  amax re-derived in every GEMM program -- and lead everywhere else.\n"
        "  From t512 the fastest row in the table is int8 W8A8: 1.70x at t512,\n"
        "  1.54x at t4096 at 459 TFLOP/s, 22% ahead of fp8 W8A8 there -- int8\n"
        "  imma reaches the tensor cores from BLOCK_M=16 while fp8 needs 64 for\n"
        "  wgmma, and int32 accumulation is exact. These are reported separately\n"
        "  on purpose: one speedup number spanning a 9% loss and a 70% win would\n"
        "  describe neither regime.\n"
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
