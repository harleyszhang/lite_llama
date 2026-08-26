#!/usr/bin/env python
"""Benchmark prefix caching *across* data-parallel replicas.

Each replica owns its KV cache outright — there is no cross-replica transfer — so a
prompt whose prefix was prefilled on replica 0 hits nothing on replica 1. On a workload
where requests fall into a handful of prefix groups (a system prompt per tenant, a
few-shot preamble per task, a chat history per session) round-robin therefore scatters
every group over every replica and makes each one pay the prefill again. Prefix-affinity
routing (``--load-balancer cache_aware``) is what turns the per-replica cache into a
pool-wide one, and this script measures the three configurations that separate the two
effects:

* **off / round_robin** — the baseline: every prompt prefills in full.
* **on / round_robin** — the cache alone. Still wins, because within one replica the
  group members that happen to land together share their prefix; but each group is
  instantiated on all ``dp`` replicas, so the pool holds ``dp`` copies of it.
* **on / cache_aware** — the cache plus affinity, which is the configuration under test.

The workload is deliberately prefill-heavy (long shared prefix, short generation): the
saving is prefill work, and a long generation dilutes it until the decode steps, which
prefix caching cannot touch, dominate the wall clock. ``--gen-len 256`` on the same
workload measures a real regime, just not this one.

What the measurements say (Qwen2.5-0.5B, 2x A10, ~980-token prefixes, ``--gen-len 4``):
affinity's share of the win grows with the number of *distinct* prefixes and shrinks
with the number of requests per prefix, which is what the arithmetic predicts — the work
round-robin wastes is ``(dp - 1) x groups`` extra prefills however many requests share
each group, so it fades as a fraction of the run when the groups are large::

    groups x per-group    cache alone    + affinity
     4 x 32                  2.75x          2.93x
     8 x 16                  1.89x          1.87x
    16 x  8                  1.38x          1.42x
    32 x  8                  1.17x          1.34x
    32 x  4                  1.06x          1.26x
    64 x  4                  1.04x          1.14x

So the cache is what pays when a few prefixes are shared very widely, and affinity is
what pays when many prefixes are each shared narrowly — the regime a multi-tenant server
is actually in, and the one where the cache on its own is nearly worthless (1.04x). Below
about 64 requests the wall clock is decided by how many admission waves each replica
needs, so the ±1 request affinity costs against an exactly even split can swamp the
prefill it saves; that is a small-batch artefact, not the policy.

Usage:
    # where affinity pays: many prefixes, few requests each
    python benchmarks/bench_dp_prefix_cache.py --dp 2 --groups 32 --per-group 4

    # where the cache pays and affinity has nothing left to add
    python benchmarks/bench_dp_prefix_cache.py --dp 2 --groups 4 --per-group 32
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lite_llama import DataParallelEngine, SamplingParams
from lite_llama.engine.dp_load_balancer import make_load_balancer

#: Greedy, with the repetition guard off: a benchmark must not have its token count
#: decided by a heuristic that fires on some rows and not others.
_GREEDY = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "stop_on_repeat": False,
}

#: One sentence of filler, repeated to build a prefix of the requested length. Its
#: content is irrelevant — what matters is that group members share it *exactly*, since
#: the cache matches on whole blocks of token ids.
_FILLER = "Follow every instruction carefully and answer as precisely as you can. "


@dataclass
class Row:
    """One (cache, policy) configuration's measurement."""

    prefix_cache: bool
    policy: str
    latency_s: float
    gen_tokens: int

    @property
    def label(self) -> str:
        return f"cache={'on ' if self.prefix_cache else 'off'} lb={self.policy}"

    @property
    def tps(self) -> float:
        return self.gen_tokens / self.latency_s if self.latency_s else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "tps": round(self.tps, 1)}


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
    scheduler; and it is torn down before the next row so the rows do not contend for
    KV, which would price the later ones differently from the earlier ones.

    Returns:
        The measurement, the completions, and the checkpoint's tokenizer — the last so
        the caller can replay the routing decisions on exactly the ids the router saw.
    """
    params = SamplingParams(max_gen_len=gen_len, **_GREEDY)
    with DataParallelEngine(
        model=model,
        data_parallel_size=dp,
        load_balancer=policy,
        max_num_seqs=max_num_seqs,
        enable_prefix_cache=prefix_cache,
        **kw,
    ) as engine:
        # Warm up weights, autotune and the allocator — but on prompts from *outside*
        # the workload, or the warm-up would leave the measured run's prefixes already
        # cached and every row would report a hit rate it did not earn.
        engine.generate(["Warm up the replicas."] * dp, SamplingParams(max_gen_len=4, **_GREEDY))

        latencies, texts = [], []
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = engine.generate(prompts, params)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            texts = [out.text for out in outputs]
        tokenizer = engine.tokenizer
        gen_tokens = sum(len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts)
    torch.cuda.empty_cache()
    return Row(prefix_cache, policy, statistics.median(latencies), gen_tokens), texts, tokenizer


def print_table(rows: list[Row], baseline: Row) -> None:
    fmt = "{:<30}{:>13}{:>13}{:>11}"
    print(f"\n{'─' * 67}")
    print(fmt.format("config", "latency (s)", "gen tokens", "speedup"))
    print(f"{'─' * 67}")
    for r in rows:
        speedup = f"{baseline.latency_s / r.latency_s:.2f}x" if r.latency_s else "—"
        print(fmt.format(r.label, f"{r.latency_s:.3f}", r.gen_tokens, speedup))
    print(f"{'─' * 67}")

    cache_only = next((r for r in rows if r.prefix_cache and r.policy == "round_robin"), None)
    affinity = next((r for r in rows if r.prefix_cache and r.policy == "cache_aware"), None)
    if not (cache_only and affinity):
        return
    print(
        f"prefix cache alone: {baseline.latency_s / cache_only.latency_s:.2f}x   "
        f"+ affinity routing: {baseline.latency_s / affinity.latency_s:.2f}x   "
        f"(affinity adds {cache_only.latency_s / affinity.latency_s:.2f}x over the cache alone)"
    )


def report_agreement(reference: list[str], rows: list[tuple[str, list[str]]]) -> None:
    """Every configuration must return the same completions: routing is not sampling.

    A shared prefix that hits the cache is *copied* K/V, not recomputed, so it can differ
    from a fresh prefill in the last bits — and an fp16 greedy tie can flip on that. A
    low agreement rate here is the flag that says the reuse is not merely inexact but
    wrong.
    """
    for label, texts in rows:
        if len(texts) != len(reference):
            continue
        same = sum(a == b for a, b in zip(reference, texts, strict=True))
        print(f"{label}: {same}/{len(reference)} completions identical to the baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", default="my_weight/Qwen2.5-0.5B")
    parser.add_argument("--dp", type=int, default=2, help="Replicas (fixed for every row)")
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
    parser.add_argument(
        "--gen-len",
        type=int,
        default=4,
        help="Tokens per request; short keeps the run prefill-bound",
    )
    parser.add_argument("--iters", type=int, default=3, help="Timed repeats (median reported)")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="Replica concurrency ceiling")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--max-gpu-num-blocks",
        type=int,
        default=65536,
        help="KV tokens per replica; stated rather than profiled, because profiling from a "
        "one-token forward hands the cache ~90%% of the card and leaves nothing for the "
        "logits of a wide prefill batch",
    )
    parser.add_argument("--log-dir", default=None, help="Write a JSON log here")
    args = parser.parse_args()

    if torch.cuda.device_count() < args.dp:
        print(
            f"--dp {args.dp} needs {args.dp} GPUs, found {torch.cuda.device_count()}",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts, group_of = build_workload(args.groups, args.per_group, args.prefix_sentences)
    kw = {"max_seq_len": args.max_seq_len, "max_gpu_num_blocks": args.max_gpu_num_blocks}

    print(f"\n{'=' * 67}")
    print(
        f"{args.model}  |  dp={args.dp}  {args.groups} prefix groups x {args.per_group} requests"
        f"  gen_len={args.gen_len}  iters={args.iters}"
    )
    print(f"gpu={torch.cuda.get_device_name(0)}  max_num_seqs={args.max_num_seqs}")
    print(f"{'=' * 67}")

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
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"bench_dp_prefix_{Path(args.model).name}_{stamp}.json"
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "model": args.model,
                        "gpu": torch.cuda.get_device_name(0),
                        "dp": args.dp,
                        "groups": args.groups,
                        "per_group": args.per_group,
                        "prefix_sentences": args.prefix_sentences,
                        "gen_len": args.gen_len,
                        "iters": args.iters,
                        "max_num_seqs": args.max_num_seqs,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                    "results": [r.as_dict() for r in rows],
                },
                indent=2,
            )
        )
        print(f"\nsaved log -> {path}")


if __name__ == "__main__":
    main()
