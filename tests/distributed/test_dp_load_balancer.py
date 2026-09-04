"""Tests for the DP load-balancing policies.

Every policy advertised by ``LOAD_BALANCERS`` is built and driven through
the same harness: idle-pool filling, round-robin cycling, prefix-aware
placement — policy behaviour without any engine at all.

Usage:
    pytest tests/distributed/test_dp_load_balancer.py
"""

from __future__ import annotations

import pytest

from rapid_llm.engine.dp_load_balancer import (
    LOAD_BALANCERS,
    CacheAwareBalancer,
    RoundRobinBalancer,
    TotalRequestsBalancer,
    TotalTokensBalancer,
    make_load_balancer,
)
from rapid_llm.engine.prefix_cache import PREFIX_CACHE_BLOCK_SIZE

_ALL = [
    RoundRobinBalancer,
    TotalRequestsBalancer,
    TotalTokensBalancer,
    CacheAwareBalancer,
]


def _prompt(tag: int, blocks: int) -> list[int]:
    """A prompt of ``blocks`` full blocks, unique to ``tag``.

    Ids stay small and positive because :func:`iter_block_hashes` packs them as
    ``uint32``, and a whole number of blocks because a trailing partial block is never
    hashed — a prompt of ``block_size - 1`` tokens would index nothing and make an
    affinity assertion vacuously false.
    """
    return [tag * 1000 + i for i in range(blocks * PREFIX_CACHE_BLOCK_SIZE)]


def _shared(prefix: list[int], tag: int, blocks: int) -> list[int]:
    """``prefix`` followed by ``blocks`` blocks unique to ``tag``."""
    return prefix + _prompt(tag, blocks)


def test_factory_builds_every_advertised_policy():
    """Every name in ``LOAD_BALANCERS`` must actually construct."""
    for name in LOAD_BALANCERS:
        assert make_load_balancer(name, dp_size=2).dp_size == 2


def test_policy_names_match_sglang():
    """The public names are SGLang's ``LoadBalanceMethod`` spellings."""
    assert LOAD_BALANCERS == ("round_robin", "total_requests", "total_tokens", "cache_aware")


def test_factory_rejects_an_unknown_policy():
    with pytest.raises(ValueError, match="unknown load-balancer"):
        make_load_balancer("magic", dp_size=2)


@pytest.mark.parametrize("cls", _ALL)
def test_non_positive_replica_count_is_rejected(cls):
    with pytest.raises(ValueError, match="dp_size must be >= 1"):
        cls(0)


@pytest.mark.parametrize("cls", _ALL)
def test_every_policy_fills_an_idle_pool_in_index_order(cls):
    """Shared tie-break: on a cold pool every policy agrees on 0,1,2."""
    balancer = cls(dp_size=3)
    assert [balancer.select(estimated_tokens=10) for _ in range(3)] == [0, 1, 2]


def test_round_robin_cycles_regardless_of_completion():
    """Round-robin steps 0,1,0,1,... and ``release`` changes nothing."""
    rr = RoundRobinBalancer(dp_size=2)

    picks = [rr.select() for _ in range(4)]
    assert picks == [0, 1, 0, 1]

    rr.release(0)  # must not shift the sequence
    assert rr.select() == 0


# --------------------------------------------------------------------------- #
# total_requests
# --------------------------------------------------------------------------- #
def test_total_requests_prefers_the_replica_that_freed_up():
    """A completion on a replica makes it the next pick."""
    balancer = TotalRequestsBalancer(dp_size=2)

    assert balancer.select() == 0  # load [1, 0]
    assert balancer.select() == 1  # load [1, 1]
    assert balancer.select() == 0  # load [2, 1]  (tie broken low; 0 then had 1)

    balancer.release(0)  # load [1, 1]
    balancer.release(0)  # load [0, 1]
    assert balancer.select() == 0


def test_total_requests_ignores_prompt_length():
    """One request is one unit however long it is — that is the whole policy.

    A 4000-token prompt must not steer this balancer, otherwise it has silently become
    ``total_tokens`` and the two names mean the same thing.
    """
    balancer = TotalRequestsBalancer(dp_size=2)

    assert balancer.select(estimated_tokens=4000) == 0
    assert balancer.select(estimated_tokens=1) == 1
    assert balancer.load == (1, 1)


def test_total_requests_release_never_goes_negative():
    """A stray release must not make a replica look emptier than empty."""
    balancer = TotalRequestsBalancer(dp_size=2)
    balancer.release(0)
    balancer.release(0)

    assert balancer.load == (0, 0)
    assert [balancer.select() for _ in range(2)] == [0, 1]


# --------------------------------------------------------------------------- #
# total_tokens
# --------------------------------------------------------------------------- #
def test_total_tokens_weighs_prompts_by_length():
    """A long prompt must keep attracting nothing until the others catch up."""
    balancer = TotalTokensBalancer(dp_size=2)

    assert balancer.select(estimated_tokens=1000) == 0  # load [1000, 0]
    assert balancer.select(estimated_tokens=10) == 1  # load [1000, 10]
    assert balancer.select(estimated_tokens=10) == 1  # load [1000, 20]
    assert balancer.select(estimated_tokens=10) == 1  # load [1000, 30]
    assert balancer.load == (1000, 30)


def test_total_tokens_differs_from_total_requests_on_a_skewed_batch():
    """The two policies must split a skewed batch differently.

    Prompts of 1000, 10, 10, 10 tokens: counting requests stripes them 0,1,0,1 and
    leaves replica 0 with 1010 tokens against replica 1's 20. Counting tokens sends
    every short prompt to replica 1.
    """
    lengths = [1000, 10, 10, 10]

    by_requests = TotalRequestsBalancer(dp_size=2)
    by_tokens = TotalTokensBalancer(dp_size=2)

    assert [by_requests.select(estimated_tokens=n) for n in lengths] == [0, 1, 0, 1]
    assert [by_tokens.select(estimated_tokens=n) for n in lengths] == [0, 1, 1, 1]


def test_total_tokens_release_subtracts_what_select_added():
    """Releasing with the same estimate must restore the previous load exactly."""
    balancer = TotalTokensBalancer(dp_size=2)

    balancer.select(estimated_tokens=512)
    balancer.release(0, estimated_tokens=512)

    assert balancer.load == (0, 0)


def test_total_tokens_treats_an_empty_prompt_as_one_token():
    """Zero-weight requests would all pile onto replica 0; the floor prevents that."""
    balancer = TotalTokensBalancer(dp_size=2)

    assert [balancer.select(estimated_tokens=0) for _ in range(4)] == [0, 1, 0, 1]


# --------------------------------------------------------------------------- #
# cache_aware
# --------------------------------------------------------------------------- #
def test_cache_aware_sends_a_repeated_prefix_back_to_the_same_replica():
    """The whole point: the second request carrying a prefix follows the first.

    Under ``total_tokens`` these two would be striped 0,1 and the shared prefix would be
    prefilled on both replicas.
    """
    prefix = _prompt(tag=1, blocks=8)
    balancer = CacheAwareBalancer(dp_size=2)

    first = balancer.select(token_ids=_shared(prefix, tag=2, blocks=1))
    balancer.release(first)
    second = balancer.select(token_ids=_shared(prefix, tag=3, blocks=1))

    assert first == second == 0


def test_cache_aware_stripes_prompts_that_share_nothing():
    """Without a shared prefix there is no affinity to act on, so it balances."""
    balancer = CacheAwareBalancer(dp_size=2)
    picks = [balancer.select(token_ids=_prompt(tag=t, blocks=4)) for t in range(4)]
    assert picks == [0, 1, 0, 1]


def test_cache_aware_clusters_two_prefix_groups_one_per_replica():
    """Two prefixes and two loaded replicas must end up one prefix each.

    This is the arrangement the policy exists to find, and the one a load-only policy
    cannot: it makes each prefix prefilled once in the pool instead of once per replica.
    Arrival is interleaved rather than grouped so that round-robin striping would give
    the opposite (and wrong) answer for a 2-group, 2-replica pool.

    Nothing is released, which is what the offline batch API does — it routes the whole
    batch before any of it runs — and what an online pool under load looks like. It also
    matters: with requests arriving strictly one at a time and finishing before the next,
    consolidating *both* prefixes onto replica 0 is the cheaper answer and the policy
    correctly gives it.
    """
    a, b = _prompt(tag=1, blocks=8), _prompt(tag=2, blocks=8)
    balancer = CacheAwareBalancer(dp_size=2)

    picks = []
    for turn in range(4):
        for group, prefix in (("a", a), ("b", b)):
            replica = balancer.select(token_ids=_shared(prefix, tag=10 + turn, blocks=1))
            picks.append((group, replica))

    by_group = {group: {replica for g, replica in picks if g == group} for group in "ab"}
    assert by_group == {"a": {0}, "b": {1}}


def test_cache_aware_consolidates_when_nothing_is_in_flight():
    """An idle pool has no balance to protect, so both prefixes land on one replica.

    The counterpart to the clustering test: releasing each request before the next
    arrives makes spreading pure loss, and one replica holding both prefixes is the
    minimum-prefill answer. Pinned because it is the case where affinity and load do not
    disagree, and a policy that spread anyway would be paying for balance nobody wanted.
    """
    a, b = _prompt(tag=1, blocks=8), _prompt(tag=2, blocks=8)
    balancer = CacheAwareBalancer(dp_size=2)

    picks = []
    for turn in range(4):
        for prefix in (a, b):
            replica = balancer.select(token_ids=_shared(prefix, tag=10 + turn, blocks=1))
            balancer.release(replica)
            picks.append(replica)

    assert set(picks) == {0}


def test_cache_aware_abandons_affinity_when_the_owner_is_loaded_enough():
    """Affinity is a discount, not a rule — a busy owner eventually loses the request.

    Without this the popular prefix's owner would absorb every request carrying it while
    its sibling idled, which is the failure mode that makes affinity-only routing worse
    than the load-only policy it replaces. The crossover is arithmetic, not a tuned
    constant: replica 0 keeps winning until its outstanding work exceeds what replica 1
    would spend prefilling the prefix from scratch.
    """
    prefix = _prompt(tag=1, blocks=8)
    balancer = CacheAwareBalancer(dp_size=2)

    # Nothing is released, so replica 0's charges accumulate exactly as in-flight
    # requests would.
    picks = [balancer.select(token_ids=_shared(prefix, tag=10 + i, blocks=1)) for i in range(20)]

    assert picks[0] == 0
    assert 1 in picks, "a saturated owner must eventually give a request away"


def test_cache_aware_with_an_empty_index_is_exactly_total_tokens():
    """On prompts that share no prefix it must reproduce ``total_tokens`` decision for
    decision.

    That equivalence is what makes this policy a safe default: it can only diverge from
    load balancing where a cache hit gives it a reason to.
    """
    prompts = [_prompt(tag=t, blocks=blocks) for t, blocks in enumerate([64, 1, 1, 1], 1)]

    cache_aware = CacheAwareBalancer(dp_size=2)
    by_tokens = TotalTokensBalancer(dp_size=2)

    assert [cache_aware.select(token_ids=ids) for ids in prompts] == [
        by_tokens.select(estimated_tokens=len(ids)) for ids in prompts
    ]
    assert cache_aware.load == by_tokens.load


def test_cache_aware_release_restores_the_load_it_charged():
    """Releasing every request must leave the pool looking idle again.

    ``release`` is not told which request ended, so it pops the oldest charge instead of
    matching one. The aggregate is what the load term reads, so it is the aggregate that
    has to come back to zero — a leak here would make a long-lived router refuse to use
    a replica that has been idle for hours.
    """
    balancer = CacheAwareBalancer(dp_size=2)
    picks = [balancer.select(token_ids=_prompt(tag=t, blocks=4)) for t in range(6)]

    assert balancer.load != (0, 0)
    for replica in picks:
        balancer.release(replica)
    assert balancer.load == (0, 0)


def test_cache_aware_keeps_the_prefix_indexed_after_the_request_finishes():
    """A finished request must leave its prefix hittable, mirroring ``PrefixCache``.

    The replica's own cache keeps an unreferenced block resident precisely so a shared
    system prompt stays warm between requests; a router that forgot on release would
    route the next one as a miss and undo that.
    """
    prompt = _prompt(tag=1, blocks=4)
    balancer = CacheAwareBalancer(dp_size=2)

    replica = balancer.select(token_ids=prompt)
    balancer.release(replica)

    assert balancer.cached_tokens(prompt, replica) == len(prompt)


def test_cache_aware_credits_only_the_leading_run_of_cached_blocks():
    """A prompt whose *middle* matches gets no credit; hashes are chained.

    Block 3 of one prompt is only reusable if blocks 0-2 are the same too, so counting
    matches anywhere would over-credit a replica and route on a hit that cannot happen.
    """
    balancer = CacheAwareBalancer(dp_size=1)
    balancer.select(token_ids=_prompt(tag=1, blocks=4))

    diverges_at_block_zero = _prompt(tag=9, blocks=1) + _prompt(tag=1, blocks=4)
    assert balancer.cached_tokens(diverges_at_block_zero, 0) == 0


def test_cache_aware_forgets_the_least_recently_used_block_past_capacity():
    """The index is bounded, or a long-lived router grows one entry per block ever seen.

    Eviction has to fall on the oldest prefix, not the newest: the hot prefix is the one
    worth remembering, and it is also the one most recently routed.
    """
    balancer = CacheAwareBalancer(dp_size=1, index_capacity=4)

    old = _prompt(tag=1, blocks=4)
    balancer.select(token_ids=old)
    assert balancer.resident_blocks(0) == 4

    fresh = _prompt(tag=2, blocks=4)
    balancer.select(token_ids=fresh)

    assert balancer.resident_blocks(0) == 4
    assert balancer.cached_tokens(fresh, 0) == len(fresh)
    assert balancer.cached_tokens(old, 0) == 0


def test_cache_aware_ignores_a_trailing_partial_block():
    """Tokens past the last full block are not hashed, so they earn no affinity.

    Their K/V is not reusable until the block completes, which is the replica's rule too.
    """
    balancer = CacheAwareBalancer(dp_size=1)
    prompt = [*_prompt(tag=1, blocks=2), 777]
    balancer.select(token_ids=prompt)

    assert balancer.cached_tokens(prompt, 0) == 2 * PREFIX_CACHE_BLOCK_SIZE


def test_cache_aware_falls_back_to_the_estimate_without_ids():
    """Handed only a count it must still weigh the prompt, not treat it as empty.

    The router is supposed to pass ids and raises if it cannot, but a policy that read a
    missing argument as zero would silently pile a batch onto replica 0.
    """
    balancer = CacheAwareBalancer(dp_size=2)

    assert balancer.select(estimated_tokens=1000) == 0
    assert balancer.select(estimated_tokens=10) == 1
    assert balancer.load == (1000, 10)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block_size": 0}, "block_size must be >= 1"),
        ({"index_capacity": 0}, "index_capacity must be >= 1"),
    ],
)
def test_cache_aware_rejects_a_degenerate_index(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CacheAwareBalancer(dp_size=2, **kwargs)


# --------------------------------------------------------------------------- #
# The token contract the router relies on
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("policy", "needs_estimate", "needs_ids"),
    [
        ("round_robin", False, False),
        ("total_requests", False, False),
        ("total_tokens", True, False),
        ("cache_aware", False, True),
    ],
)
def test_each_policy_declares_exactly_what_it_reads(policy, needs_estimate, needs_ids):
    """The two flags are what buy the router a tokenizer pass, and for whom.

    Each must be true only where the argument is actually read: an over-declaration makes
    every ``round_robin`` batch pay for tokenisation, and an under-declaration makes
    ``total_tokens`` route on zeros or ``cache_aware`` route on ``None`` — which looks
    like a working router that never gets a cache hit.

    ``cache_aware`` declares only ``needs_token_ids``: the ids carry their own length, so
    also claiming the estimate would say it needs something it derives.
    """
    balancer = make_load_balancer(policy, dp_size=2)
    assert balancer.needs_token_estimate is needs_estimate
    assert balancer.needs_token_ids is needs_ids
