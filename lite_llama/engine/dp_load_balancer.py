"""Load-balancing policies for the data-parallel router.

Deciding *which* replica a request goes to is the one interesting choice in data
parallelism, so it lives here as a small strategy object rather than inline in the
router — the same split SGLang draws between its ``DataParallelController`` and the
``LoadBalanceMethod`` it is configured with (``round_robin`` / ``total_requests`` /
``total_tokens``), and vLLM between ``DPLBAsyncMPClient`` and its balancing.

A balancer sees only two things: how many replicas exist, and \u2014 for the load-aware
policies \u2014 how many requests each is currently carrying. It never touches a queue, a
process or a tensor, which is what keeps it pure and unit-testable without a GPU.

Usage:
    balancer = make_load_balancer("round_robin", dp_size=2)
    replica = balancer.select(estimated_tokens=len(prompt_ids))
    balancer.release(replica)   # when that request finishes
"""

from __future__ import annotations

from abc import ABC, abstractmethod

#: Policy names accepted by :func:`make_load_balancer`, mirroring the subset of
#: SGLang's ``LoadBalanceMethod`` that a synchronous, offline engine can honour.
LOAD_BALANCERS = ("round_robin", "least_loaded")


class LoadBalancer(ABC):
    """Chooses a replica per request, and is told when a request frees one.

    Args:
        dp_size: Number of replicas to choose between.

    Raises:
        ValueError: If ``dp_size`` is below 1.
    """

    def __init__(self, dp_size: int) -> None:
        if dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {dp_size}")
        self.dp_size = dp_size

    @abstractmethod
    def select(self, estimated_tokens: int = 0) -> int:
        """Return the replica index this request should go to.

        Args:
            estimated_tokens: Prompt length, used by size-aware policies and ignored
                by the rest. Passing it always keeps the call sites uniform.
        """

    def release(self, replica: int) -> None:  # noqa: B027 - optional hook, not abstract
        """Note that a request on ``replica`` has finished.

        The default does nothing on purpose: load-unaware policies keep no counts, so
        this is an optional hook rather than an abstract method. Only load-aware
        policies override it. Kept on the base class so the router calls it
        unconditionally.
        """


class RoundRobinBalancer(LoadBalancer):
    """Hand replicas out in turn, ignoring how long each request runs.

    The cheapest policy and the right default for offline batches, where the prompts
    are known together and no single one dominates. It is exactly SGLang's
    ``round_robin_scheduler`` minus the liveness skip (an offline pool has no replicas
    dropping out mid-run). Striping the requests \u2014 rather than handing each replica a
    contiguous slice \u2014 mixes long and short prompts evenly, so no one replica inherits
    every long prompt from an already-sorted list.
    """

    def __init__(self, dp_size: int) -> None:
        super().__init__(dp_size)
        self._counter = 0

    def select(self, estimated_tokens: int = 0) -> int:
        replica = self._counter % self.dp_size
        self._counter += 1
        return replica


class LeastLoadedBalancer(LoadBalancer):
    """Send each request to the replica carrying the fewest in-flight requests.

    The offline analogue of SGLang's ``total_requests`` method: it keeps a running
    count per replica, picks the minimum (lowest index breaks ties, so it matches
    round-robin when every replica is idle), and relies on :meth:`release` to
    decrement as requests complete. This is what pays off when prompt lengths are
    skewed \u2014 a replica that drew several long prompts stops attracting new ones.
    """

    def __init__(self, dp_size: int) -> None:
        super().__init__(dp_size)
        self._inflight = [0] * dp_size

    def select(self, estimated_tokens: int = 0) -> int:
        replica = min(range(self.dp_size), key=lambda i: self._inflight[i])
        self._inflight[replica] += 1
        return replica

    def release(self, replica: int) -> None:
        if self._inflight[replica] > 0:
            self._inflight[replica] -= 1


def make_load_balancer(policy: str, dp_size: int) -> LoadBalancer:
    """Build the balancer named by ``policy``.

    Args:
        policy: One of :data:`LOAD_BALANCERS`.
        dp_size: Number of replicas.

    Raises:
        ValueError: On an unknown policy name.
    """
    if policy == "round_robin":
        return RoundRobinBalancer(dp_size)
    if policy == "least_loaded":
        return LeastLoadedBalancer(dp_size)
    raise ValueError(f"unknown load-balancer {policy!r}; choose from {LOAD_BALANCERS}")
