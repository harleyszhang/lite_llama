"""L4 tile-signaling primitive: correctness, epochs, bounded waits, overlap.

The producer/consumer pair is tested against the torch reference across
shapes and dtypes, the two entry points must agree bit-for-bit (identical
kernels, identical grids — only the stream strategy differs), epochs must
advance monotonically without clearing flags, and a starved consumer must
give up loudly rather than hang or fabricate data. Device-side overlap of the
pipelined arms is asserted from timeline records.

Usage:
    pytest tests/kernels/test_tile_signal.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rapid_llm.kernels.tile_signal import (
    TileSignalBuffer,
    _tile_signal_consume_kernel,
    default_block_split,
    pipelined_gemm_swiglu,
    serial_gemm_swiglu,
)

_RTOL = {torch.bfloat16: 3e-2, torch.float16: 3e-2, torch.float32: 2e-4}
_ATOL = {torch.bfloat16: 3e-2, torch.float16: 3e-2, torch.float32: 2e-4}

_SHAPES = [
    (128, 256, 192, torch.bfloat16),
    (67, 130, 97, torch.float16),  # not a multiple of any block size
    (256, 512, 320, torch.float32),
    (1024, 512, 1536, torch.bfloat16),
]


def _problem(m, n, k, dtype):
    """Scaled random MLP step: activations and both [K, N] weight matrices."""
    a = torch.randn(m, k, dtype=dtype, device="cuda") * 0.1
    gate_w = torch.randn(k, n, dtype=dtype, device="cuda") * 0.05
    up_w = torch.randn(k, n, dtype=dtype, device="cuda") * 0.05
    return a, gate_w, up_w


def _reference(a, gate_w, up_w):
    """Torch serial baseline: two matmuls then the silu-mul epilogue."""
    ref = F.silu(a.float() @ gate_w.float()) * (a.float() @ up_w.float())
    return ref.to(a.dtype)


def test_block_split_respects_sm_budget():
    """Grids stay within the SM budget and never shrink to zero."""
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    producer, consumer = default_block_split(10_000, sm)
    assert producer >= 1 and consumer >= 1
    assert producer + consumer <= sm

    # Fewer tiles than blocks: clamp, still positive.
    producer, consumer = default_block_split(3, sm)
    assert producer <= 3 and consumer <= 3

    # An explicit over-budget pair is rejected — co-residency is the safety
    # argument for the consumer's bounded wait, so it cannot be waived.
    with pytest.raises(ValueError):
        default_block_split(10_000, sm, producer_blocks=sm, consumer_blocks=2)


@pytest.mark.parametrize("m,n,k,dtype", _SHAPES)
def test_pipelined_matches_torch_reference(m, n, k, dtype):
    """The overlapped pipeline computes silu(a@gw) * (a@uw) correctly."""
    a, gate_w, up_w = _problem(m, n, k, dtype)
    buffer = TileSignalBuffer.for_problem(m, n, 64, 64)

    out = pipelined_gemm_swiglu(a, gate_w, up_w, buffer)
    torch.cuda.synchronize()

    assert buffer.dropped_tiles() == 0
    ref = _reference(a, gate_w, up_w)
    torch.testing.assert_close(out, ref, rtol=_RTOL[dtype], atol=_ATOL[dtype])


@pytest.mark.parametrize("m,n,k,dtype", _SHAPES)
def test_serial_entry_is_the_bitwise_control_arm(m, n, k, dtype):
    """Same kernels, same grids: only the stream strategy differs.

    Every tile is computed by the same instructions in the same order
    whichever block picks it up, so the two entry points must agree exactly —
    the benchmark's A/B comparison relies on it.
    """
    a, gate_w, up_w = _problem(m, n, k, dtype)
    buffer = TileSignalBuffer.for_problem(m, n, 64, 64)

    pipelined = pipelined_gemm_swiglu(a, gate_w, up_w, buffer)
    serial = serial_gemm_swiglu(a, gate_w, up_w, buffer)
    torch.cuda.synchronize()

    assert buffer.dropped_tiles() == 0
    assert torch.equal(pipelined, serial)


def test_epochs_advance_without_clearing_flags():
    """Stale flags stay below every future epoch; reuse stays correct."""
    m, n, k = 128, 256, 192
    a, gate_w, up_w = _problem(m, n, k, torch.bfloat16)
    buffer = TileSignalBuffer.for_problem(m, n, 64, 64)
    num_tiles = buffer.num_tiles

    seen_epochs = []
    for run in range(1, 4):
        a_run = a * run  # different data every run
        out = pipelined_gemm_swiglu(a_run, gate_w, up_w, buffer)
        torch.cuda.synchronize()
        assert buffer.dropped_tiles() == 0

        # Flags were never reset: they hold the epochs of completed runs,
        # monotonically, and the latest run owns every tile.
        flags = buffer.flags
        assert int(flags.min()) >= 1
        assert int(flags.max()) == run
        seen_epochs.append(buffer.epoch)

        ref = _reference(a_run, gate_w, up_w)
        torch.testing.assert_close(out, ref, rtol=_RTOL[torch.bfloat16], atol=_ATOL[torch.bfloat16])

    assert seen_epochs == [1, 2, 3]
    assert num_tiles == int(buffer.flags.numel())


def test_starved_consumer_gives_up_loudly():
    """No producer + tiny MAX_SPIN: bounded exit, every tile accounted.

    This is the watchdog shape: a consumer whose producer never ran must
    neither hang (the whole test finishing is the proof — the host-side
    watchdog is any outer timeout runner) nor write a single output row; the
    drop counter reports the full tile count instead.
    """
    m, n = 128, 256
    block_m, block_n = 64, 64
    buffer = TileSignalBuffer.for_problem(m, n, block_m, block_n)
    num_tiles = buffer.num_tiles

    gate = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    up = torch.empty_like(gate)
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")

    epoch = buffer.advance_epoch()
    buffer.consumer_work.zero_()
    buffer.fail_count.zero_()
    _tile_signal_consume_kernel[(num_tiles,)](
        gate,
        up,
        out,
        buffer.flags,
        buffer.consumer_work,
        buffer.fail_count,
        epoch,
        m,
        n,
        gate.stride(0),
        gate.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        MAX_SPIN=64,  # tiny: the flags never reach this epoch
        num_warps=4,
        num_stages=2,
    )
    torch.cuda.synchronize()

    # Every tile gave up, nothing was written, and the count says so.
    assert buffer.dropped_tiles() == num_tiles
    assert torch.isnan(out).all()


def test_pipelined_overlap_shows_on_the_timeline():
    """Producer and consumer regions must overlap on the device.

    A large problem keeps the producer busy long after the first tiles are
    published, so the epilogue's timeline region must intersect the GEMM's —
    the direct device-side evidence that L4 overlaps work rather than merely
    producing correct output.
    """
    from rapid_llm.batch_overlap.overlap import Timeline

    m, n, k = 4096, 2048, 1536
    a, gate_w, up_w = _problem(m, n, k, torch.bfloat16)
    buffer = TileSignalBuffer.for_problem(m, n, 64, 64)
    timeline = Timeline(enabled=True, device="cuda")

    for _ in range(3):
        out = pipelined_gemm_swiglu(a, gate_w, up_w, buffer, timeline=timeline)
    torch.cuda.synchronize()
    assert buffer.dropped_tiles() == 0
    torch.testing.assert_close(
        out, _reference(a, gate_w, up_w), rtol=_RTOL[torch.bfloat16], atol=_ATOL[torch.bfloat16]
    )

    records = timeline.collect()
    producers = [r for r in records if r.stream == "producer"]
    consumers = [r for r in records if r.stream == "consumer"]
    assert len(producers) >= 1 and len(consumers) >= 1

    best = max(
        min(p.end_ms, c.end_ms) - max(p.start_ms, c.start_ms) for p in producers for c in consumers
    )
    assert best > 0, f"no producer/consumer overlap; best intersection {best} ms"
