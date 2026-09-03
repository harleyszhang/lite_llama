#!/usr/bin/env python
"""Prefix caching across data-parallel replicas.

Prompts share prefixes within a group; whether a cache-aware balancer
keeps a group on one replica decides if the prefix cache ever hits. The
benchmark measures both routing quality and the resulting KV savings.

Usage:
    python benchmarks/bench_dp_prefix_cache.py --model <ckpt> --dp 2
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import (
    TimedRow,
    add_dp_args,
    measure_dp,
    print_row_table,
    print_run_header,
    report_agreement,
    require_gpus,
    timestamped_log_path,
    write_json_log,
)
from lite_llama.engine.dp_load_balancer import make_load_balancer

#: One sentence of filler, repeated to build a prefix of the requested length. Its
#: content is irrelevant — what matters is that group members share it *exactly*, since
#: the cache matches on whole blocks of token ids.
_FILLER = "Follow every instruction carefully and answer as precisely as you can. "


@dataclass
class Row(TimedRow):
    """One (cache, policy) configuration's measurement."""

    prefix_cache: bool = False
    policy: str = ""

    @property
    def label(self) -> str:
        return f"cache={'on ' if self.prefix_cache else 'off'} lb={self.policy}"


def build_workload(
    groups: int, per_group: int, prefix_sentences: int, seed: int = 0
) -> tuple[list[str], list[int]]:
    """Requests from ``groups`` prefix groups, arriving in a shuffled order.

    The arrival order is what the experiment turns on, and a *regular* interleave
    invalidates it. Laying the groups out with a fixed stride, for instance, makes the
    group index a function of the arrival index with the same period as round-robin's
    ``i % dp``: with an even ``groups`` and an odd stride, every even-indexed arrival is
    an even-numbered group, so round-robin lands each group on exactly one replica and
    scores as perfectly affinity-aware without knowing anything. A seeded shuffle has no
    period to collide with, which is also the honest model of a server's arrivals.

    Every prompt ends in a unique suffix so that only the shared *prefix* can be reused,
    and no two prompts are the same request.

    Returns:
        The prompts, and each prompt's group index for :func:`describe_routing`.
    """
    prefixes = [
        f"You are assistant number {g}. " + _FILLER * prefix_sentences for g in range(groups)
    ]
    arrivals = [g for g in range(groups) for _ in range(per_group)]
    random.Random(seed).shuffle(arrivals)
    prompts = [
        f"{prefixes[g]}Question {i}: what is {i} plus {i + 1}?" for i, g in enumerate(arrivals)
    ]
    return prompts, arrivals


def describe_routing(prompts: list[str], group_of: list[int], dp: int, tokenizer) -> None:
    """Print, per policy, how many copies of each prefix the pool ends up holding.

    This is the mechanism the latencies are a consequence of, and it is worth showing
    separately because it is exact where a latency is noisy: a ``(group, replica)`` pair
    is one prefix that some replica has to prefill from scratch. Round-robin's count is
    ``dp x groups`` — every group on every replica — and ``groups`` is the floor.

    Replayed on throwaway balancers rather than read out of the engines, because the
    engines are gone by now and the decision is pure: same ids, same policy, same split.
    """
    token_ids = [list(ids) for ids in tokenizer(prompts, add_special_tokens=True)["input_ids"]]
    print(f"routing ({len(prompts)} requests, {len(set(group_of))} prefixes, dp={dp}):")
    for policy in ("round_robin", "cache_aware"):
        balancer = make_load_balancer(policy, dp)
        placed = [balancer.select(estimated_tokens=len(ids), token_ids=ids) for ids in token_ids]
        per_replica = [placed.count(r) for r in range(dp)]
        copies = len(set(zip(group_of, placed, strict=True)))
        print(
            f"  {policy:12} {copies} prefix copies across the pool "
            f"(floor {len(set(group_of))}), requests per replica {per_replica}"
        )


def measure(
    model: str,
    prompts: list[str],
    *,
    dp: int,
    policy: str,
    prefix_cache: bool,
    gen_len: int,
    iters: int,
    max_num_seqs: int,
    **kw,
) -> tuple[Row, list[str], object]:
    """Time one configuration end to end through the DP coordinator.

    The engine is rebuilt per row because the prefix cache is created with the
    scheduler (see :func:`measure_dp`).

    Returns:
        The measurement, the completions, and the checkpoint's tokenizer — the last so
        the caller can replay the routing decisions on exactly the ids the router saw.
    """
    latency, gen_tokens, texts, tokenizer = measure_dp(
        model,
        prompts,
        dp=dp,
        gen_len=gen_len,
        iters=iters,
        max_num_seqs=max_num_seqs,
        load_balancer=policy,
        enable_prefix_cache=prefix_cache,
        # Warm up on prompts from *outside* the workload, or the warm-up would leave
        # the measured run's prefixes already cached and every row would report a hit
        # rate it did not earn.
        warmup_prompts=["Warm up the replicas."] * dp,
        **kw,
    )
    row = Row(latency_s=latency, gen_tokens=gen_tokens, prefix_cache=prefix_cache, policy=policy)
    return row, texts, tokenizer


def print_table(rows: list[Row], baseline: Row) -> None:
    print_row_table(
        ["config", "latency (s)", "gen tokens", "speedup"],
        [30, 13, 13, 11],
        [
            [
                r.label,
                f"{r.latency_s:.3f}",
                str(r.gen_tokens),
                f"{baseline.latency_s / r.latency_s:.2f}x" if r.latency_s else "—",
            ]
            for r in rows
        ],
    )

    cache_only = next((r for r in rows if r.prefix_cache and r.policy == "round_robin"), None)
    affinity = next((r for r in rows if r.prefix_cache and r.policy == "cache_aware"), None)
    if not (cache_only and affinity):
        return
    print(
        f"prefix cache alone: {baseline.latency_s / cache_only.latency_s:.2f}x   "
        f"+ affinity routing: {baseline.latency_s / affinity.latency_s:.2f}x   "
        f"(affinity adds {cache_only.latency_s / affinity.latency_s:.2f}x over the cache alone)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    add_dp_args(
        parser,
        default_gen_len=4,
        default_iters=3,
        default_max_num_seqs=16,
        default_max_seq_len=2048,
        default_max_gpu_num_blocks=65536,
        dp_help="Replicas (fixed for every row)",
        gen_len_help="Tokens per request; short keeps the run prefill-bound",
        blocks_help=(
            "KV tokens per replica; stated rather than profiled, because profiling from a "
            "one-token forward hands the cache ~90%% of the card and leaves nothing for the "
            "logits of a wide prefill batch"
        ),
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=4,
        help="Distinct shared prefixes; > --dp is where affinity pays",
    )
    parser.add_argument("--per-group", type=int, default=16, help="Requests sharing each prefix")
    parser.add_argument(
        "--prefix-sentences", type=int, default=80, help="Filler sentences in the shared prefix"
    )
    args = parser.parse_args()

    require_gpus(args.dp)

    prompts, group_of = build_workload(args.groups, args.per_group, args.prefix_sentences)
    kw = {"max_seq_len": args.max_seq_len, "max_gpu_num_blocks": args.max_gpu_num_blocks}

    print_run_header(
        args.model,
        {
            "dp": args.dp,
            "workload": f"{args.groups} prefix groups x {args.per_group} requests",
            "gen_len": args.gen_len,
            "iters": args.iters,
            "max_num_seqs": args.max_num_seqs,
        },
        width=67,
    )

    rows: list[Row] = []
    texts_by_label: list[tuple[str, list[str]]] = []
    reference: list[str] = []
    for prefix_cache, policy in [
        (False, "round_robin"),
        (True, "round_robin"),
        (True, "cache_aware"),
    ]:
        row, texts, tokenizer = measure(
            args.model,
            prompts,
            dp=args.dp,
            policy=policy,
            prefix_cache=prefix_cache,
            gen_len=args.gen_len,
            iters=args.iters,
            max_num_seqs=args.max_num_seqs,
            **kw,
        )
        rows.append(row)
        if not reference:
            reference = texts
        else:
            texts_by_label.append((row.label, texts))
        print(f"  {row.label}: {row.latency_s:.3f}s")

    print_table(rows, rows[0])
    print()
    describe_routing(prompts, group_of, args.dp, tokenizer)
    print()
    report_agreement(reference, texts_by_label)

    if args.log_dir:
        path = timestamped_log_path(args.log_dir, f"bench_dp_prefix_{Path(args.model).name}")
        write_json_log(
            path,
            {
                "model": args.model,
                "gpu": torch.cuda.get_device_name(0),
                "dp": args.dp,
                "groups": args.groups,
                "per_group": args.per_group,
                "prefix_sentences": args.prefix_sentences,
                "gen_len": args.gen_len,
                "iters": args.iters,
                "max_num_seqs": args.max_num_seqs,
            },
            [r.as_dict() for r in rows],
        )


if __name__ == "__main__":
    main()
