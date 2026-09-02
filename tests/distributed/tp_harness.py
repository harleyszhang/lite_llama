"""Run a payload on a real TP process group so sharded layers can be tested.

``run_on_tp_ranks`` spawns one process per TP rank, initialises the
grid, and returns the payload's results — the bridge between CPU unit
tests and full TP engine tests.

Usage:
    results = run_on_tp_ranks(payload, tp_size=2)
"""

from __future__ import annotations

import os
import queue as queue_module
import time
import traceback
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from lite_llama.distributed import parallel_state as ps
from lite_llama.executor.executor import free_port


def needs_gpus(count: int):
    """Mark a test as needing ``count`` real devices, one per rank."""
    return pytest.mark.skipif(
        torch.cuda.device_count() < count,
        reason=f"needs {count} CUDA devices, found {torch.cuda.device_count()}",
    )


def _worker(
    payload: Callable[[int], Any],
    rank: int,
    tp_size: int,
    dp_size: int,
    port: int,
    backend: str,
    results: mp.Queue,
) -> None:
    """One rank: take a device, join the grid, run the payload, report answer or traceback.

    The device is claimed *before* ``init_parallel`` because nccl binds the calling
    thread's current device at rendezvous; leaving every rank on device 0 is the classic
    way to get a hang instead of a result. On gloo there is no device to claim. The
    traceback travels as text rather than being raised: an exception in a child process
    is invisible to pytest, and a rank that dies silently leaves its peers blocked in a
    collective with nothing to explain why.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)  # not setdefault: an inherited value is stale
    try:
        if backend == "nccl":
            torch.cuda.set_device(rank)
        ps.init_parallel(
            global_rank=rank, tp_size=tp_size, dp_size=dp_size, master_port=port, backend=backend
        )
        results.put((rank, payload(rank), None))
    except BaseException:  # reported to the parent, which re-raises it verbatim
        results.put((rank, None, traceback.format_exc()))
    finally:
        ps.destroy_parallel()


def run_on_tp_ranks(
    payload: Callable[[int], Any],
    tp_size: int,
    *,
    dp_size: int = 1,
    timeout: float = 300.0,
    backend: str = "nccl",
) -> list[Any]:
    """Run ``payload(rank)`` on every rank of a ``dp_size x tp_size`` grid.

    Args:
        payload: Module-level function (spawn has to pickle it) called once per rank with
            its global rank; must return plain Python data.
        tp_size: Ranks per replica.
        dp_size: Number of replicas.
        timeout: Seconds to wait for all ranks. Exceeding it means a collective
            mismatch — a rank calling a collective its peers do not is a deadlock, not a
            wrong answer, so it has to fail as a bounded test rather than hang the suite.
        backend: ``"nccl"`` for the data plane (one device per rank, see
            :func:`needs_gpus`) or ``"gloo"`` for a device-free grid — enough to
            exercise the control plane, which is what
            :func:`~lite_llama.distributed.parallel_state.broadcast_object` and the
            executor's plan hand-off live on.

    Returns:
        One result per global rank, in rank order.

    Raises:
        AssertionError: If any rank raised; the message carries that rank's traceback.
        TimeoutError: If some rank never reported.
    """
    world_size = tp_size * dp_size
    if backend == "nccl" and torch.cuda.device_count() < world_size:
        raise RuntimeError(f"{world_size} ranks need {world_size} devices, one each")

    context = mp.get_context("spawn")
    results: mp.Queue = context.Queue()
    port = free_port()
    workers = [
        context.Process(
            target=_worker,
            args=(payload, rank, tp_size, dp_size, port, backend, results),
            daemon=True,
        )
        for rank in range(world_size)
    ]
    collected: dict[int, Any] = {}
    for worker in workers:
        worker.start()
    try:
        deadline = time.monotonic() + timeout
        while len(collected) < world_size:
            left = deadline - time.monotonic()
            if left <= 0:
                missing = sorted(set(range(world_size)) - set(collected))
                raise TimeoutError(f"ranks {missing} did not report within {timeout}s")
            try:
                rank, value, error = results.get(timeout=left)
            except queue_module.Empty:
                continue
            if error is not None:
                raise AssertionError(f"rank {rank} failed:\n{error}")
            collected[rank] = value
    finally:
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
    return [collected[rank] for rank in range(world_size)]
