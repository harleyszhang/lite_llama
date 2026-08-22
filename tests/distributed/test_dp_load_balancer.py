"""Tests for the DP load-balancing policies in :mod:`lite_llama.engine.dp_load_balancer`.

These are pure objects — no GPU, no process, no checkpoint — so the whole file runs on
CPU in milliseconds. What matters is the *policy*, and the two policies differ in
exactly one observable way: round-robin ignores completions, least-loaded reacts to
them. Both must reduce to the same sequence when every replica is idle, because that
tie-break (lowest index first) is what makes least-loaded a safe drop-in default.
"""

from __future__ import annotations

import pytest

from lite_llama.engine.dp_load_balancer import (
    LOAD_BALANCERS,
    LeastLoadedBalancer,
    RoundRobinBalancer,
    make_load_balancer,
)


def test_factory_builds_every_advertised_policy():
    """Every name in ``LOAD_BALANCERS`` must actually construct."""
    for name in LOAD_BALANCERS:
        assert make_load_balancer(name, dp_size=2).dp_size == 2


def test_factory_rejects_an_unknown_policy():
    with pytest.raises(ValueError, match="unknown load-balancer"):
        make_load_balancer("magic", dp_size=2)


@pytest.mark.parametrize("cls", [RoundRobinBalancer, LeastLoadedBalancer])
def test_non_positive_replica_count_is_rejected(cls):
    with pytest.raises(ValueError, match="dp_size must be >= 1"):
        cls(0)


def test_round_robin_cycles_regardless_of_completion():
    """Round-robin steps 0,1,0,1,... and ``release`` changes nothing."""
    rr = RoundRobinBalancer(dp_size=2)

    picks = [rr.select() for _ in range(4)]
    assert picks == [0, 1, 0, 1]

    rr.release(0)  # must not shift the sequence
    assert rr.select() == 0


def test_least_loaded_matches_round_robin_when_idle():
    """With no outstanding work, least-loaded fills replicas 0,1,2 in order.

    This shared tie-break is what lets least-loaded stand in for round-robin without
    surprising a caller who never releases (e.g. fire-and-forget).
    """
    ll = LeastLoadedBalancer(dp_size=3)
    assert [ll.select() for _ in range(3)] == [0, 1, 2]


def test_least_loaded_prefers_the_replica_that_freed_up():
    """A completion on a replica makes it the next pick."""
    ll = LeastLoadedBalancer(dp_size=2)

    # Load both replicas evenly, then one more onto replica 0.
    assert ll.select() == 0  # loads: [1, 0]
    assert ll.select() == 1  # loads: [1, 1]
    assert ll.select() == 0  # loads: [2, 1]  (tie broken low; 0 then had 1)

    ll.release(0)  # loads: [1, 1]
    ll.release(0)  # loads: [0, 1]
    assert ll.select() == 0  # emptiest replica


def test_least_loaded_release_never_goes_negative():
    """A stray release must not make a replica look emptier than empty.

    The router releases once per finished request; a defensive floor means a
    double-release (or a release with no matching select) cannot corrupt the counts.
    """
    ll = LeastLoadedBalancer(dp_size=2)
    ll.release(0)
    ll.release(0)

    # Both still read as 0 in-flight, so selection is the plain 0,1 order.
    assert [ll.select() for _ in range(2)] == [0, 1]
