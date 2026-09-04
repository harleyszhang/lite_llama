"""Freeze measured dispatch ranking records (ROADMAP v0.10, foundation 2).

Each op's feasible specs are timed against a reference with a per-spec
tolerance, and the winner is written into the frozen store that
``install_frozen_perf_provider`` later replays in production.

Usage:
    python benchmarks/kernels/freeze_dispatch_ranking.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from microbench import bench, bench_stateful, metadata, verify

import rapid_llm.kernels  # registers every spec row
from rapid_llm.kernels.dispatcher import REGISTRY, dispatch
from rapid_llm.kernels.dispatcher.autotune import freeze_record, frozen_store
from rapid_llm.kernels.dispatcher.autotune.config_key import normalize_gpu_name
from rapid_llm.kernels.dispatcher.autotune.frozen import install_frozen_perf_provider
from rapid_llm.kernels.dispatcher.dispatch import resolve_target
from rapid_llm.kernels.dispatcher.spec import KernelSpec
from rapid_llm.platform.spec import PlatformInfo, capabilities_match

#: Same serving geometry as bench_flashinfer.py, so the two tables compare.
BATCH, Q_HEADS, KV_HEADS, HEAD_DIM = 8, 32, 8, 128
HIDDEN = 4096
SEQ = 2048
DT = torch.bfloat16
DEVICE = "cuda"


@dataclass(frozen=True)
class Case:
    """How to measure one op: the dispatch-key context plus the builder."""

    label: str
    scheme: str
    layout: frozenset[str]
    measure: Callable[[list[KernelSpec]], dict[str, float]]


def _feasible(
    specs: list[KernelSpec], *, dtype: str, scheme: str, layout: frozenset[str], info: PlatformInfo
) -> list[KernelSpec]:
    """Rows surviving every physical gate (the golden gate stays out of this)."""
    out = []
    for spec in specs:
        if not spec.dtype_ok(dtype) or not spec.scheme_ok(scheme):
            continue
        if not capabilities_match(spec.capability, info):
            continue
        if spec.layout_missing(layout):
            continue
        if spec.available is not None:
            try:
                if not resolve_target(spec.available)():
                    continue
            except Exception:
                continue
        out.append(spec)
    return out


def _reference(specs: list[KernelSpec]) -> KernelSpec:
    return next(s for s in specs if s.backend == "native")


def _atol(spec: KernelSpec, *, default: float) -> float:
    """Verify window: twice the row's golden record, or the op default.

    The margin keeps the bf16 cancellation floor from failing on the last ulp;
    the golden gate, not this tool, owns the strict number.
    """
    record = spec.golden.max_abs_diff
    return 2 * record if record is not None else default


def measure_rmsnorm(specs: list[KernelSpec]) -> dict[str, float]:
    torch.manual_seed(0)
    x = torch.randn(BATCH * 8, HIDDEN, dtype=DT, device=DEVICE)
    residual = torch.randn_like(x)
    residual0 = residual.clone()
    weight = torch.randn(HIDDEN, dtype=DT, device=DEVICE)

    # Correctness before any timing: the stateful loop below leaves its
    # operands rotated/added, so every verify runs on pristine inputs.
    ref_spec = _reference(specs)
    ref_out, ref_res = resolve_target(ref_spec.target)(x.clone(), residual.clone(), weight, 1e-5)
    for spec in specs:
        if spec is ref_spec:
            continue
        out, res = resolve_target(spec.target)(x.clone(), residual.clone(), weight, 1e-5)
        atol = _atol(spec, default=1e-2)
        verify(f"rmsnorm {spec.name}", out, ref_out, rtol=1e-2, atol=atol)
        verify(f"rmsnorm residual {spec.name}", res, ref_res, rtol=1e-2, atol=atol)

    times: dict[str, float] = {}
    for spec in specs:
        fn = resolve_target(spec.target)
        # skip_rmsnorm mutates its residual in place; a functional row pays the
        # same reset outside the timed window, so one timer serves both.
        times[spec.name] = bench_stateful(
            lambda fn=fn: fn(x, residual, weight, 1e-5), lambda: residual.copy_(residual0)
        )
    return times


def measure_rope(specs: list[KernelSpec]) -> dict[str, float]:
    torch.manual_seed(0)
    q = torch.randn(BATCH * 128, Q_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    k = torch.randn(BATCH * 128, KV_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, device=DEVICE).float() / HEAD_DIM))
    pos = torch.arange(128, device=DEVICE).float()
    freqs = torch.einsum("s,d->sd", pos, inv_freq)
    cos = freqs.cos().repeat(BATCH, 1, 1).to(DT)
    sin = freqs.sin().repeat(BATCH, 1, 1).to(DT)

    ref_spec = _reference(specs)
    ref_q, ref_k = resolve_target(ref_spec.target)(q.clone(), k.clone(), cos, sin)
    for spec in specs:
        if spec is ref_spec:
            continue
        out_q, out_k = resolve_target(spec.target)(q.clone(), k.clone(), cos, sin)
        atol = _atol(spec, default=2e-2)
        verify(f"rope q {spec.name}", out_q, ref_q, rtol=1e-2, atol=atol)
        verify(f"rope k {spec.name}", out_k, ref_k, rtol=1e-2, atol=atol)

    q0, k0 = q.clone(), k.clone()
    times: dict[str, float] = {}
    for spec in specs:
        fn = resolve_target(spec.target)
        # The native row rotates in place; the reset keeps the timer honest.
        times[spec.name] = bench_stateful(
            lambda fn=fn: fn(q, k, cos, sin), lambda: (q.copy_(q0), k.copy_(k0))
        )
    return times


def measure_prefill(specs: list[KernelSpec]) -> dict[str, float]:
    torch.manual_seed(0)
    lens = [SEQ, SEQ // 2, SEQ, SEQ // 4, SEQ, SEQ // 2, SEQ, SEQ]
    total = sum(lens)
    b_seq_len = torch.tensor(lens, dtype=torch.int32, device=DEVICE)
    b_start_loc = torch.cumsum(b_seq_len, 0) - b_seq_len
    q = torch.randn(total, Q_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    k = torch.randn(total, KV_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    v = torch.randn_like(k)
    scale = HEAD_DIM**-0.5

    ref_spec = _reference(specs)
    ref = resolve_target(ref_spec.target)(q, k, v, scale, b_start_loc, b_seq_len, SEQ)
    for spec in specs:
        if spec is ref_spec:
            continue
        atol = _atol(spec, default=2e-2)
        verify(
            f"prefill {spec.name}",
            resolve_target(spec.target)(q, k, v, scale, b_start_loc, b_seq_len, SEQ),
            ref,
            rtol=2e-2,
            atol=atol,
        )

    times: dict[str, float] = {}
    for spec in specs:
        fn = resolve_target(spec.target)
        times[spec.name] = bench(lambda fn=fn: fn(q, k, v, scale, b_start_loc, b_seq_len, SEQ))
    return times


def measure_decode(specs: list[KernelSpec]) -> dict[str, float]:
    torch.manual_seed(0)
    kv_buffer = torch.randn(2 * BATCH * SEQ, KV_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    b_seq_len = torch.full((BATCH,), SEQ, dtype=torch.int32, device=DEVICE)
    b_req_idx = torch.arange(BATCH, dtype=torch.int64, device=DEVICE)
    b_req_tokens_table = (
        torch.arange(BATCH * SEQ, dtype=torch.int64, device=DEVICE).view(1, -1).repeat(BATCH, 1)
    )
    for b in range(BATCH):
        b_req_tokens_table[b] += b * SEQ
    q = torch.randn(BATCH, Q_HEADS, HEAD_DIM, dtype=DT, device=DEVICE)
    scale = HEAD_DIM**-0.5
    k_cache, v_cache = kv_buffer[: BATCH * SEQ], kv_buffer[BATCH * SEQ :]

    ref_spec = _reference(specs)
    ref = resolve_target(ref_spec.target)(
        q, k_cache, v_cache, scale, b_req_tokens_table, b_req_idx, b_seq_len, SEQ
    )
    for spec in specs:
        if spec is ref_spec:
            continue
        atol = _atol(spec, default=2e-2)
        verify(
            f"decode {spec.name}",
            resolve_target(spec.target)(
                q, k_cache, v_cache, scale, b_req_tokens_table, b_req_idx, b_seq_len, SEQ
            ),
            ref,
            rtol=2e-2,
            atol=atol,
        )

    times: dict[str, float] = {}
    for spec in specs:
        fn = resolve_target(spec.target)
        times[spec.name] = bench(
            lambda fn=fn: fn(
                q, k_cache, v_cache, scale, b_req_tokens_table, b_req_idx, b_seq_len, SEQ
            )
        )
    return times


#: The ops with a real choice on a serving GPU today. ``linear``/``moe`` have
#: contenders only on sm90+ (deepgemm), so the <2-feasible rule skips them here.
MEASURERS: dict[str, Case] = {
    "rmsnorm": Case("b8_s8_h4096", "unquantized", frozenset(), measure_rmsnorm),
    "rope": Case("b8_s128_h128", "unquantized", frozenset(), measure_rope),
    "attention.prefill": Case("ragged11k_gqa4x", "unquantized", frozenset(), measure_prefill),
    "attention.decode": Case(
        "b8_s2048_gqa4x", "unquantized", frozenset({"kv:paged"}), measure_decode
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", default=None, help="comma-separated op ids (default: all known)")
    parser.add_argument("--store-dir", default=None, help="autotune cache root override")
    parser.add_argument("--dry-run", action="store_true", help="measure and print, don't write")
    parser.add_argument("--log", default=None, help="JSON log path (default: docs/benchmark_logs/)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("freeze_dispatch_ranking needs a CUDA device — records are per-GPU")
    print(metadata())
    info = PlatformInfo.detect()
    gpu = normalize_gpu_name(info.gpu_name)
    store = frozen_store(args.store_dir)

    wanted = set(args.ops.split(",")) if args.ops else None
    log: dict = {"gpu": gpu, "case_dtype": str(DT), "ops": {}}
    for op, case in MEASURERS.items():
        if wanted is not None and op not in wanted:
            continue
        feasible = _feasible(
            list(REGISTRY.implementations(op)),
            dtype="bf16",
            scheme=case.scheme,
            layout=case.layout,
            info=info,
        )
        if len(feasible) < 2:
            names = [s.name for s in REGISTRY.implementations(op)]
            print(f"\nskip {op}: {len(feasible)} of {len(names)} rows feasible — nothing to rank")
            continue
        print(f"\n== {op}  ({case.label}) ==")
        times = case.measure(feasible)
        for name, us in sorted(times.items(), key=lambda kv: kv[1]):
            print(f"  {name:<36} {us:9.1f} us")
        winner = min(times, key=times.get)
        if not args.dry_run:
            tune_key = freeze_record(
                store,
                op=op,
                scheme=case.scheme,
                dims={},
                dtype="bf16",
                measurements=times,
                gpu=gpu,
            )
            print(f"  frozen -> {tune_key.shape_bucket}  ({store.cache_dir / (op + '.json')})")
        log["ops"][op] = {"case": case.label, "winner": winner, "latencies_us": times}

    if not log["ops"]:
        print("\nnothing frozen — no op had two feasible rows")
        return

    log_path = (
        Path(args.log)
        if args.log
        else Path(__file__).parents[2]
        / "docs"
        / "benchmark_logs"
        / f"freeze_dispatch_ranking_{gpu}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nlog    -> {log_path}")

    if not args.dry_run:
        # Proof the wiring works end to end: the same process now dispatches
        # the measured winner (decisions for these ops were not cached yet).
        install_frozen_perf_provider(store)
        print("\ndispatch with the frozen records:")
        for op, case in MEASURERS.items():
            if op in log["ops"]:
                sel = dispatch(op, dtype=DT, layout=case.layout)
                print(f"  {op:<20} -> {sel.spec.name}")


if __name__ == "__main__":
    main()
