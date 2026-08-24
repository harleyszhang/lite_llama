"""Tests for the DP load-balancing policies in :mod:`lite_llama.engine.dp_load_balancer`.

These are pure objects — no GPU, no process, no checkpoint — so the whole file runs on
CPU in milliseconds. Three things are pinned down:

* **the tie-break** — every policy must emit 0,1,2,... on an idle pool, because that is
  what makes a load-aware policy a safe drop-in for round-robin;
* **what each policy counts** — ``total_requests`` reacts to completions,
  ``total_tokens`` reacts to prompt *length*, and the difference has to be observable or
  the second policy is just the first one under another name (which is the bug this
  suite exists to prevent regressing);
* **the token-estimate contract** — ``needs_token_estimate`` must be true exactly for
  the policies that read the argument, since the router uses it to decide whether to
  spend a tokenizer pass.
"""

from __future__ import annotations

import pytest

from lite_llama.engine.dp_load_balancer import (
    LOAD_BALANCERS,
    RoundRobinBalancer,
    TotalRequestsBalancer,
    TotalTokensBalancer,
    make_load_balancer,
)

_ALL = [RoundRobinBalancer, TotalRequestsBalancer, TotalTokensBalancer]


def test_factory_builds_every_advertised_policy():
    """Every name in ``LOAD_BALANCERS`` must actually construct."""
    for name in LOAD_BALANCERS:
        assert make_load_balancer(name, dp_size=2).dp_size == 2


def test_policy_names_match_sglang():
    """The public names are SGLang's ``LoadBalanceMethod`` spellings."""
    assert LOAD_BALANCERS == ("round_robin", "total_requests", "total_tokens")


def test_factory_rejects_an_unknown_policy():
    with pytest.raises(ValueError, match="unknown load-balancer"):
        make_load_balancer("magic", dp_size=2)


@pytest.mark.parametrize("cls", _ALL)
def test_non_positive_replica_count_is_rejected(cls):
    with pytest.raises(ValueError, match="dp_size must be >= 1"):
        cls(0)


@pytest.mark.parametrize("cls", _ALL)
def test_every_policy_fills_an_idle_pool_in_index_order(cls):
    """Shared tie-break: on a cold pool all three policies agree on 0,1,2."""
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
# The estimate contract the router relies on
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("policy", "needs_estimate"),
    [("round_robin", False), ("total_requests", False), ("total_tokens", True)],
)
def test_only_token_aware_policies_ask_for_an_estimate(policy, needs_estimate):
    """``needs_token_estimate`` is the flag that buys the router a tokenizer pass.

    It must be true only where the argument is actually read: an over-declaration makes
    every ``round_robin`` batch pay for tokenisation, and an under-declaration makes
    ``total_tokens`` route on zeros.
    """
    assert make_load_balancer(policy, dp_size=2).needs_token_estimate is needs_estimate
