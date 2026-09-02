"""Tests for :mod:`lite_llama.engine.async_data_parallel`.

Fakes replace processes and queues (``_ChannelQueue``,
``_FakeProcess``), so the asyncio surface — streamed chunks, replica
spread, failure propagation — is tested without a real replica.

Usage:
    pytest tests/distributed/test_async_data_parallel.py
"""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator
from typing import ClassVar

import pytest
import torch

from lite_llama.engine import data_parallel as dp_module
from lite_llama.engine.async_data_parallel import AsyncDataParallelEngine
from lite_llama.engine.async_engine import StreamedOutput

_TIMEOUT = 20.0


class _ChannelQueue:
    """``mp.Queue`` stand-in that blocks for real.

    The pump thread parks in ``get(timeout=...)``; a list-backed fake would make
    that either a busy loop or an instant ``Empty``, so this one delegates to
    ``queue.Queue`` and honours ``block`` and ``timeout`` the way the real queue
    does. ``sent`` keeps every message put, which is how tests assert routing.
    """

    def __init__(self) -> None:
        self._items: queue.Queue = queue.Queue()
        self.sent: list = []

    def put(self, item) -> None:
        self.sent.append(item)
        self._items.put(item)

    def get(self, block: bool = True, timeout: float | None = None):
        return self._items.get(block=block, timeout=timeout)


class _FakeProcess:
    """Records the spawn arguments, then reports ready and stays alive."""

    spawned: ClassVar[list[tuple]] = []

    def __init__(self, target, args, daemon) -> None:
        self.args = args
        self.pid = 4242  # the liveness error message names pids and exit codes
        self.exitcode = None
        self._alive = True

    def start(self) -> None:
        _FakeProcess.spawned.append(self.args)
        self.args[-1].put(("ready", self.args[0], None))  # result_queue

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self._alive = False

    def terminate(self) -> None:
        self._alive = False


def _build_engine(monkeypatch: pytest.MonkeyPatch, **kwargs) -> AsyncDataParallelEngine:
    """An ``AsyncDataParallelEngine`` over two fake replicas, no GPU involved."""
    monkeypatch.setattr(_FakeProcess, "spawned", [])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        dp_module.mp,
        "get_context",
        lambda method: type("_Ctx", (), {"Queue": _ChannelQueue, "Process": _FakeProcess})(),
    )
    return AsyncDataParallelEngine(model="unused", data_parallel_size=2, **kwargs)


@pytest.fixture
async def grid(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncDataParallelEngine]:
    engine = _build_engine(monkeypatch)
    yield engine
    await engine.shutdown()


async def _collect(engine, prompt, **kwargs) -> list[StreamedOutput]:
    return [chunk async for chunk in engine.generate(prompt, **kwargs)]


async def _first_chunk(stream) -> StreamedOutput:
    async for chunk in stream:
        return chunk
    raise AssertionError("the stream produced nothing")


async def _await_dispatch(engine, request_id: str) -> None:
    """Block until ``request_id`` has been routed to some replica's queue.

    ``generate`` registers its stream *before* the queue put, so once the add is
    visible any result message for the id will find its stream.
    """
    for _ in range(500):
        sent = [m for q in engine._request_queues for m in q.sent]
        if any(m[0] == "add" and m[1] == request_id for m in sent):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"request {request_id} was never dispatched")


# --------------------------------------------------------------------------- #
# Streaming through the pump
# --------------------------------------------------------------------------- #
async def test_generate_streams_chunks_until_the_request_finishes(grid):
    """Delta frames carry the running text; the last frame carries the reason."""
    task = asyncio.create_task(_collect(grid, "hi", request_id="mine"))
    await _await_dispatch(grid, "mine")

    grid._result_queue.put(("delta", "mine", "Hel", "Hel", 2, 1))
    grid._result_queue.put(("delta", "mine", "lo", "Hello", 2, 2))
    grid._result_queue.put(("finished", "mine", "eos", "Hello", 2, 2))

    chunks = await asyncio.wait_for(task, _TIMEOUT)

    assert [chunk.delta for chunk in chunks] == ["Hel", "lo", ""]
    assert [chunk.text for chunk in chunks] == ["Hel", "Hello", "Hello"]
    assert all(not chunk.is_finished for chunk in chunks[:-1])
    assert chunks[-1].is_finished
    assert chunks[-1].finish_reason == "eos"
    assert chunks[-1].prompt_tokens == 2
    assert chunks[-1].completion_tokens == 2


async def test_concurrent_requests_are_spread_over_both_replicas(grid):
    """Round-robin plus two concurrent coroutines must use both GPUs.

    A stream whose replica never answers hangs forever, so each task is fed
    after its dispatch is observed. The ``ghost`` finish is the race where a
    consumer left before its answer arrived: dropped by id, fatal to no one.
    """
    first = asyncio.create_task(_collect(grid, "a", request_id="r1"))
    second = asyncio.create_task(_collect(grid, "b", request_id="r2"))
    await _await_dispatch(grid, "r1")
    await _await_dispatch(grid, "r2")

    replica_of = {
        message[1]: index
        for index, q in enumerate(grid._request_queues)
        for message in q.sent
        if message[0] == "add"
    }
    assert replica_of == {"r1": 0, "r2": 1}

    grid._result_queue.put(("finished", "ghost", "eos", "?", 1, 1))
    for request_id in ("r1", "r2"):
        grid._result_queue.put(("finished", request_id, "eos", "ok", 1, 1))

    both = await asyncio.wait_for(asyncio.gather(first, second), _TIMEOUT)
    assert [chunks[-1].text for chunks in both] == ["ok", "ok"]


async def test_duplicate_live_request_id_is_rejected(grid):
    first = grid.generate("a", request_id="same")
    task = asyncio.create_task(_first_chunk(first))
    await _await_dispatch(grid, "same")

    with pytest.raises(ValueError, match="already active"):
        await anext(grid.generate("b", request_id="same"))

    grid._result_queue.put(("finished", "same", "eos", "ok", 1, 1))
    await asyncio.wait_for(task, _TIMEOUT)
    await first.aclose()


async def test_a_failed_request_raises_in_its_consumer(grid):
    """A refused prompt is that one caller's exception, not a server fault."""
    task = asyncio.create_task(_collect(grid, "hi", request_id="mine"))
    await _await_dispatch(grid, "mine")

    grid._result_queue.put(("failed", "mine", "ValueError: prompt of 9000 tokens"))

    with pytest.raises(RuntimeError, match="9000 tokens"):
        await asyncio.wait_for(task, _TIMEOUT)


async def test_generate_text_awaits_the_whole_completion(grid):
    task = asyncio.create_task(grid.generate_text("hi", request_id="mine"))
    await _await_dispatch(grid, "mine")
    grid._result_queue.put(("delta", "mine", "He", "He", 1, 1))
    grid._result_queue.put(("finished", "mine", "length", "Hello", 1, 2))

    final = await asyncio.wait_for(task, _TIMEOUT)

    assert final.text == "Hello"
    assert final.finish_reason == "length"


# --------------------------------------------------------------------------- #
# Cancellation and failure
# --------------------------------------------------------------------------- #
async def test_abandoning_a_stream_aborts_the_request_and_releases_the_balancer(
    monkeypatch: pytest.MonkeyPatch,
):
    """An abandoned connection must free both the replica slot and the load count.

    Uses the request-counting balancer because it keeps state the test can read:
    a ``release`` that never ran would leave its load at one forever, and every
    later request would pile onto the other replica.
    """
    engine = _build_engine(monkeypatch, load_balancer="total_requests")
    try:
        stream = engine.generate("hi", request_id="mine")
        task = asyncio.create_task(_first_chunk(stream))
        await _await_dispatch(engine, "mine")
        engine._result_queue.put(("delta", "mine", "x", "x", 1, 1))
        await asyncio.wait_for(task, _TIMEOUT)
        await stream.aclose()

        for _ in range(500):
            if ("abort", "mine") in engine._request_queues[0].sent:
                break
            await asyncio.sleep(0.01)
        assert ("abort", "mine") in engine._request_queues[0].sent
        assert engine._balancer.load == (0, 0)
    finally:
        await engine.shutdown()


async def test_a_dead_replica_fails_every_open_stream(monkeypatch: pytest.MonkeyPatch, grid):
    """Silence from a dead worker must become an error, not a hang.

    The bug this guards against is invisible until a server stops responding:
    a replica killed by the OOM killer leaves no message on the queue, and a
    plain blocking ``get`` would wait forever. Polling liveness turns that
    silence into every stream's exception.
    """
    monkeypatch.setattr(dp_module, "_LIVENESS_POLL_S", 0.05)
    task = asyncio.create_task(_collect(grid, "hi", request_id="mine"))
    await _await_dispatch(grid, "mine")

    grid._workers[0]._alive = False

    with pytest.raises(RuntimeError, match="data-parallel engine failed"):
        await asyncio.wait_for(task, _TIMEOUT)

    # The pump has stopped: accepting another request here would leave its
    # stream waiting for a result that no thread can ever consume.
    with pytest.raises(RuntimeError, match="has failed"):
        await anext(grid.generate("later"))


async def test_generate_after_shutdown_is_an_error(grid):
    await grid.shutdown()

    with pytest.raises(RuntimeError, match="has been shut down"):
        async for _ in grid.generate("hi"):
            pass


async def test_shutdown_is_idempotent(grid):
    await grid.shutdown()
    await grid.shutdown()
