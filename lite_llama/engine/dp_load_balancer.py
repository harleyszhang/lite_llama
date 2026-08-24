"""Load-balancing policies for the data-parallel router.

Deciding *which* replica a request goes to is the one interesting choice in data
parallelism, so it lives here as a small strategy object rather than inline in the
router — the same split SGLang draws between its ``DataParallelController`` and the
``LoadBalanceMethod`` it is configured with (``round_robin`` / ``total_requests`` /
``total_tokens``), and vLLM between ``DPLBAsyncMPClient`` and its balancing.

A balancer sees only two things: how many replicas exist, and — for the load-aware
policies — how much work each is currently carrying. It never touches a queue, a
process or a tensor, which is what keeps it pure and unit-testable without a GPU.

Each policy declares through :attr:`LoadBalancer.needs_token_estimate` whether it
actually reads ``estimated_tokens``. That flag is the honest contract the router needs:
counting tokens costs a tokenizer pass over the whole batch, and only ``total_tokens``
spends it. A policy that quietly accepted the argument and ignored it (as
``least_loaded`` used to) made the router pay for an estimate no one read — and invited
it to pass a cheap wrong number instead.

Usage:
    balancer = make_load_balancer("total_tokens", dp_size=2)
    replica = balancer.select(estimated_tokens=len(prompt_ids))
    balancer.release(replica, estimated_tokens=len(prompt_ids))   # request finished
"""

from __future__ import annotations

from abc import ABC, abstractmethod

#: Policy names accepted by :func:`make_load_balancer`, spelled as SGLang's
#: ``LoadBalanceMethod`` spells them so that a reader who knows one knows the other.
LOAD_BALANCERS = ("round_robin", "total_requests", "total_tokens")


class LoadBalancer(ABC):
    """Chooses a replica per request, and is told when a request frees one.

    Args:
        dp_size: Number of replicas to choose between.

    Raises:
        ValueError: If ``dp_size`` is below 1.
    """

    #: Whether :meth:`select` and :meth:`release` read ``estimated_tokens``. The
    #: router skips tokenising the batch when every policy in play says ``False``.
    needs_token_estimate: bool = False

    def __init__(self, dp_size: int) -> None:
        if dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {dp_size}")
        self.dp_size = dp_size

    @abstractmethod
    def select(self, estimated_tokens: int = 0) -> int:
        """Return the replica index this request should go to.

        Args:
            estimated_tokens: Prompt length in tokens. Read only by policies whose
                :attr:`needs_token_estimate` is ``True``; passing it unconditionally
                keeps the call sites uniform.
        """

    def release(  # noqa: B027 - optional hook, not abstract
        self, replica: int, estimated_tokens: int = 0
    ) -> None:
        """Note that a request on ``replica`` has finished.

        The default does nothing on purpose: load-unaware policies keep no counts, so
        this is an optional hook rather than an abstract method. Only load-aware
        policies override it. Kept on the base class so the router calls it
        unconditionally.

        Args:
            replica: The replica the request ran on.
            estimated_tokens: The same estimate that was passed to :meth:`select`, so
                a token-aware policy can subtract exactly what it added.
        """


class RoundRobinBalancer(LoadBalancer):
    """Hand replicas out in turn, ignoring how long each request runs.

    The cheapest policy and the right default for offline batches, where the prompts
    are known together and no single one dominates. It is exactly SGLang's
    ``round_robin_scheduler`` minus the liveness skip (an offline pool has no replicas
    dropping out mid-run). Striping the requests rather than handing each replica a
    contiguous slice mixes long and short prompts evenly, so no one replica inherits
    every long prompt from an already-sorted list.
    """

    def __init__(self, dp_size: int) -> None:
        super().__init__(dp_size)
        self._counter = 0

    def select(self, estimated_tokens: int = 0) -> int:
        replica = self._counter % self.dp_size
        self._counter += 1
        return replica


class _CountingBalancer(LoadBalancer):
    """Send each request to the replica with the smallest running total.

    The two load-aware policies differ only in *what* they count, so the arithmetic —
    pick the minimum, add on select, subtract on release — is written once and the
    subclass supplies the weight of one request via :meth:`weigh`. Lowest index breaks
    ties, so an idle pool degenerates to round-robin.
    """

    def __init__(self, dp_size: int) -> None:
        super().__init__(dp_size)
        self._load = [0] * dp_size

    @abstractmethod
    def weigh(self, estimated_tokens: int) -> int:
        """How much this request adds to a replica's running total."""

    def select(self, estimated_tokens: int = 0) -> int:
        replica = min(range(self.dp_size), key=lambda i: self._load[i])
        self._load[replica] += self.weigh(estimated_tokens)
        return replica

    def release(self, replica: int, estimated_tokens: int = 0) -> None:
        self._load[replica] = max(0, self._load[replica] - self.weigh(estimated_tokens))

    @property
    def load(self) -> tuple[int, ...]:
        """Current running total per replica — the state a visualiser plots."""
        return tuple(self._load)


class TotalRequestsBalancer(_CountingBalancer):
    """Fewest in-flight *requests* wins — SGLang's ``total_requests``.

    Every request weighs 1, so this is the right policy when prompts are roughly
    interchangeable and what matters is not stacking four requests on one replica while
    another idles.
    """

    def weigh(self, estimated_tokens: int) -> int:
        return 1


class TotalTokensBalancer(_CountingBalancer):
    """Fewest in-flight *tokens* wins — SGLang's ``total_tokens``.

    Prefill cost is linear in prompt length, so with skewed lengths request counts are
    the wrong unit: two 4k prompts are not the load of two 40-token ones. This is the
    only policy that reads ``estimated_tokens``, and it declares so — the router
    tokenises the batch for it and for nothing else.

    A request weighs at least 1 so that a batch of empty prompts still round-robins
    instead of piling onto replica 0.
    """

    needs_token_estimate = True

    def weigh(self, estimated_tokens: int) -> int:
        return max(1, estimated_tokens)


def make_load_balancer(policy: str, dp_size: int) -> LoadBalancer:
    """Build the balancer named by ``policy``.

    Args:
        policy: One of :data:`LOAD_BALANCERS`.
        dp_size: Number of replicas.

    Raises:
        ValueError: On an unknown policy name.
    """
    builders = {
        "round_robin": RoundRobinBalancer,
        "total_requests": TotalRequestsBalancer,
        "total_tokens": TotalTokensBalancer,
    }
    if policy not in builders:
        raise ValueError(f"unknown load-balancer {policy!r}; choose from {LOAD_BALANCERS}")
    return builders[policy](dp_size)
