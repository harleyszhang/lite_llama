"""Expert-parallel dispatch/combine over equal-split all-to-all.

:class:`AllToAllDispatcher` is this repo's counterpart of sglang's
``DeepEPDispatcher`` (``srt/managers/schedule_batch.py`` / the DeepEP backend):
same two-phase contract — ``dispatch_a`` posts the exchange, ``dispatch_b``
consumes it, ``combine_a``/``combine_b`` mirror the return trip — so a TBO
strategy can sandwich the other micro-batch's compute between the halves.
The transport differs deliberately: DeepEP's low-latency kernels need sm90+,
which the A10 target (sm86) does not have, so this dispatcher is pure
``torch.distributed``.

The exchange is *capacity based*, like DeepEP's low-latency mode: every rank
pads its ``rows * top_k`` routing slots up to a per-destination capacity of
``rows * top_k`` (the worst case where one rank owns every slot) and the
all-to-all is a single equal-split ``all_to_all_single``. The pad cost is
bounded — decode batches are memory-bound on expert weights, whose traffic is
unchanged — and it buys two properties the alternatives lose:

* the split sizes are static on the host, so no GPU→CPU sync per layer
  (an unequal-split ``all_to_all_single`` would need received counts as
  Python ints, serialising every dispatch);
* shapes are fixed per (rows, top_k), keeping the path CUDA-graph capturable.

Routing weights never leave the sender: the expert side runs with unit
weights (``top_k=1`` form) and ``combine_b`` applies the sender's weights
when it scatter-adds the ``k`` expert results back onto each token. Padding
rows carry id/weight sentinels, run through the experts harmlessly, and are
dropped by the gather. An EP world of one degenerates to a local permute —
the same code path single-process tests exercise.

Usage:
    dispatcher = AllToAllDispatcher(num_experts, num_local_experts, offset)
    handle = dispatcher.dispatch_a(x, topk_ids, topk_weights)
    local_x, local_ids, local_w = dispatcher.dispatch_b(handle)
    local_out = fused_moe(local_x, w1, w2, local_w, local_ids)
    handle = dispatcher.combine_a(handle, local_out)
    out = dispatcher.combine_b(handle)   # [rows, hidden], already reduced
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..batch_overlap import CommStreamPool
from ..distributed.parallel_state import get_ep_group


@dataclass
class DispatchHandle:
    """Everything ``dispatch_a`` learned that the later phases need.

    Attributes:
        rows: Tokens in the dispatching batch (before top-k expansion).
        top_k: Routing slots per token.
        cap: Per-destination capacity of the exchange (``rows * top_k``).
        ep_size: Ranks in the EP group; the buffers are ``ep_size * cap`` rows.
        order: ``[ep*cap]`` permutation — flat slot index of each sorted row.
        send_pos: ``[n]`` positions of the sorted rows in the send buffer.
        flat_weights: ``[n]`` routing weights, kept on the sender.
        recv_x / recv_ids: Dispatch receive buffers (rows and expert ids).
        recv_out: Combine receive buffer, allocated by ``combine_a``.
        events: Comm-stream events outstanding phases must fence on.
    """

    rows: int
    top_k: int
    cap: int
    ep_size: int
    order: torch.Tensor
    send_pos: torch.Tensor
    flat_weights: torch.Tensor
    recv_x: torch.Tensor
    recv_ids: torch.Tensor
    recv_out: torch.Tensor | None = None
    events: list[torch.cuda.Event] = field(default_factory=list)


class AllToAllDispatcher:
    """Two-phase EP dispatch/combine over the shared comm stream.

    Args:
        num_experts: Global routed-expert count (routing ids live in
            ``[0, num_experts)``).
        num_local_experts: Experts this rank owns.
        expert_offset: Global id of this rank's first expert.
    """

    def __init__(self, num_experts: int, num_local_experts: int, expert_offset: int) -> None:
        if num_experts % num_local_experts != 0:
            raise ValueError(
                f"{num_experts} experts do not split into groups of {num_local_experts}"
            )
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.expert_offset = expert_offset

    # ------------------------------------------------------------------ #
    # dispatch: tokens out, per-expert batches in
    # ------------------------------------------------------------------ #

    def dispatch_a(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> DispatchHandle:
        """Permute routing slots by destination rank and post the exchange.

        Args:
            x: ``[rows, hidden]`` token embeddings.
            topk_ids: ``[rows, top_k]`` global expert ids.
            topk_weights: ``[rows, top_k]`` routing weights; these stay local.

        Returns:
            A handle carrying the buffers and the fence events; finish the
            phase with :meth:`dispatch_b`.
        """
        rows, hidden = x.shape
        k = topk_ids.shape[1]
        n = rows * k
        group = get_ep_group()
        ep_size = self._ep_size(group)
        # The placement must tile the group: rank r owns experts
        # [r*num_local, (r+1)*num_local), so ``dest = id // num_local`` lands in
        # [0, ep_size) exactly when ep_size * num_local == num_experts. A world
        # of one owns every expert (num_local == num_experts); anything else is
        # a mis-built dispatcher and would index the send buffer out of bounds.
        if ep_size * self.num_local_experts != self.num_experts:
            raise ValueError(
                f"EP group of {ep_size} cannot host {self.num_experts} experts in "
                f"groups of {self.num_local_experts}"
            )
        cap = n  # worst-case capacity: every slot could target one rank

        flat_ids = topk_ids.reshape(-1)
        dest = torch.div(flat_ids, self.num_local_experts, rounding_mode="floor")
        # Stable sort keeps slots of one token in routing order within a
        # destination segment — deterministic across ranks.
        sorted_dest, order = torch.sort(dest, stable=True)
        # Position of each row inside its destination segment: sort is stable,
        # so segment start = first index with the same dest (searchsorted left).
        seg_start = torch.searchsorted(sorted_dest, sorted_dest, side="left")
        pos_in_seg = torch.arange(n, device=x.device) - seg_start
        send_pos = sorted_dest * cap + pos_in_seg

        # Padding slots keep zeros: id 0 targets rank 0 / local expert 0 with
        # weight 0 on the sender side (never applied — combine gathers only
        # ``send_pos`` rows), so pad tokens run through the experts harmlessly.
        send_x = torch.zeros(ep_size * cap, hidden, dtype=x.dtype, device=x.device)
        send_ids = torch.zeros(ep_size * cap, dtype=flat_ids.dtype, device=x.device)
        # ``order[i]`` is the flat slot (token*k + j) placed at sorted position
        # ``i``; // k folds it back to its token row.
        send_x[send_pos] = x[order // k]
        send_ids[send_pos] = flat_ids[order]

        pool = CommStreamPool.for_device(x.device)
        recv_x = torch.empty_like(send_x)
        recv_ids = torch.empty_like(send_ids)
        events = [
            e
            for e in (
                pool.all_to_all_async(recv_x, send_x, group=group, label="ep.dispatch.x"),
                pool.all_to_all_async(recv_ids, send_ids, group=group, label="ep.dispatch.ids"),
            )
            if e is not None
        ]
        return DispatchHandle(
            rows=rows,
            top_k=k,
            cap=cap,
            ep_size=ep_size,
            order=order,
            send_pos=send_pos,
            flat_weights=topk_weights.reshape(-1).to(x.dtype),
            recv_x=recv_x,
            recv_ids=recv_ids,
            events=events,
        )

    def dispatch_b(
        self, handle: DispatchHandle
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fence the dispatch exchange; return this rank's expert batch.

        Returns:
            ``(local_x, local_ids, local_weights)``: ``[ep*cap, hidden]`` rows,
            ``[ep*cap, 1]`` local expert ids (pad rows clamp onto expert 0),
            and unit weights — the sender applies the real ones in
            :meth:`combine_b`.
        """
        self._fence(handle)
        local_ids = (handle.recv_ids - self.expert_offset).clamp(
            0, self.num_local_experts - 1
        )
        ones = torch.ones(
            handle.recv_x.shape[0], 1, dtype=handle.recv_x.dtype, device=handle.recv_x.device
        )
        return handle.recv_x, local_ids.reshape(-1, 1), ones

    # ------------------------------------------------------------------ #
    # combine: expert results back, weighted token sums out
    # ------------------------------------------------------------------ #

    def combine_a(self, handle: DispatchHandle, local_out: torch.Tensor) -> DispatchHandle:
        """Post the return exchange for ``local_out`` (``[ep*cap, hidden]``)."""
        pool = CommStreamPool.for_device(local_out.device)
        handle.recv_out = torch.empty_like(local_out)
        event = pool.all_to_all_async(
            handle.recv_out, local_out, group=get_ep_group(), label="ep.combine"
        )
        if event is not None:
            handle.events.append(event)
        return handle

    def combine_b(self, handle: DispatchHandle) -> torch.Tensor:
        """Fence the return exchange; reduce to ``[rows, hidden]``.

        Un-permutes the received results, scales each slot by the routing
        weight the sender kept, and sums the ``top_k`` slots per token. The
        output is complete on every rank — EP's routed path needs no
        all-reduce, unlike the TP expert split.
        """
        self._fence(handle)
        assert handle.recv_out is not None, "combine_b before combine_a"
        n = handle.rows * handle.top_k
        # Gather drops the padding rows of the sorted layout in one step.
        results_sorted = handle.recv_out[handle.send_pos]
        results_flat = torch.empty_like(results_sorted)
        results_flat[handle.order] = results_sorted
        weighted = results_flat * handle.flat_weights.unsqueeze(-1)
        out = torch.zeros(
            handle.rows, handle.recv_out.shape[1],
            dtype=weighted.dtype, device=weighted.device,
        )
        out.index_add_(0, torch.arange(n, device=out.device) // handle.top_k, weighted)
        handle.events.clear()
        return out

    # ------------------------------------------------------------------ #
    # synchronous convenience: a + b back to back (non-TBO path)
    # ------------------------------------------------------------------ #

    def dispatch(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> tuple[DispatchHandle, torch.Tensor, torch.Tensor, torch.Tensor]:
        """:meth:`dispatch_a` + :meth:`dispatch_b` with the fence immediate."""
        handle = self.dispatch_a(x, topk_ids, topk_weights)
        local_x, local_ids, local_weights = self.dispatch_b(handle)
        return handle, local_x, local_ids, local_weights

    def combine(self, handle: DispatchHandle, local_out: torch.Tensor) -> torch.Tensor:
        """:meth:`combine_a` + :meth:`combine_b` with the fence immediate."""
        return self.combine_b(self.combine_a(handle, local_out))

    # ------------------------------------------------------------------ #

    @staticmethod
    def _ep_size(group) -> int:
        """Rank count of ``group``; 1 when there is no live process group."""
        if group is None:
            return 1
        import torch.distributed as dist

        return dist.get_world_size(group)

    @staticmethod
    def _fence(handle: DispatchHandle) -> None:
        """Order the compute stream after every outstanding exchange event."""
        if handle.events and torch.cuda.is_available():
            stream = torch.cuda.current_stream()
            for event in handle.events:
                stream.wait_event(event)
