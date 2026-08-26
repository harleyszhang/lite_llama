"""Row arithmetic of :meth:`SlotBatch.copy_prefix` — pure CPU, no model needed.

Fixed slot regions leave no indirection to share K/V through, so reusing a prefix
means physically moving rows between slots. The move is the one place where a
wrong offset is invisible: attention would read plausible numbers from the wrong
tokens and the request would simply produce worse text. So the offsets are pinned
here against a cache filled with a per-row fingerprint, on the CPU, where the
whole buffer can be read back and compared.

The stubs below implement only what :class:`SlotBatch` touches at construction
(a slot table to seed, a K/V buffer whose filler region is zeroed, and the
allocator hand-off) plus the buffer itself, which is all ``copy_prefix`` uses.
"""

from __future__ import annotations

import torch

from lite_llama.executor.slot_batch import SlotBatch

LAYERS = 2
HEADS = 2
HEAD_DIM = 3


class _StubKVCacheManager:
    """Just the per-layer K/V buffer and the row hand-off that takes it over."""

    def __init__(self, rows: int) -> None:
        self.gpu_kv_buffer = [torch.zeros(rows, HEADS, HEAD_DIM) for _ in range(LAYERS)]
        self.claimed: int | None = None

    def claim(self, rows: int) -> None:
        self.claimed = rows


class _StubRunner:
    """Minimal stand-in for :class:`~lite_llama.executor.model_runner.ModelRunner`."""

    def __init__(self, num_slots: int, max_seq_len: int) -> None:
        self.device = torch.device("cpu")
        self.max_seq_len = max_seq_len
        self.atten_info = object()
        self.b_req_tokens_table = torch.zeros(num_slots, max_seq_len, dtype=torch.int32)
        self.kv_cache_manager = _StubKVCacheManager(num_slots * max_seq_len)


def _batch(num_slots: int = 4, max_seq_len: int = 8) -> SlotBatch:
    """A batch whose cache holds ``row + 1000 * layer`` in every element.

    Distinct per row *and* per layer, so a copy that lands one row off, or that
    reads the wrong layer, cannot coincidentally match.
    """
    batch = SlotBatch(_StubRunner(num_slots, max_seq_len))
    for index, layer in enumerate(batch._runner.kv_cache_manager.gpu_kv_buffer):
        rows = layer.shape[0]
        layer.copy_(torch.arange(rows, dtype=layer.dtype).view(rows, 1, 1) + 1000.0 * index)
    return batch


def _rows(batch: SlotBatch, layer: int) -> torch.Tensor:
    """Row fingerprints of one layer, as a flat vector."""
    return batch._runner.kv_cache_manager.gpu_kv_buffer[layer][:, 0, 0].clone()


def test_a_run_lands_at_the_same_offset_in_the_destination_slot():
    """Source and destination share their in-slot offset, per layer.

    A chained block hash pins a block to one absolute prompt position, so a prefix
    that matches at all matches at the same rows — the copy shifts the slot base
    and nothing else.
    """
    batch = _batch()
    width = batch.max_seq_len
    before = [_rows(batch, layer) for layer in range(LAYERS)]

    batch.copy_prefix([(0, 1, 2, 3)])  # slot 0 -> slot 1, tokens [2, 5)

    for layer in range(LAYERS):
        after = _rows(batch, layer)
        moved = slice(width + 2, width + 5)
        assert torch.equal(after[moved], before[layer][2:5])
        # Everything else, including the rest of the destination slot, is untouched.
        untouched = torch.ones(after.shape[0], dtype=torch.bool)
        untouched[moved] = False
        assert torch.equal(after[untouched], before[layer][untouched])


def test_runs_of_one_call_are_independent():
    """Several runs, several sources: each lands at its own offset."""
    batch = _batch()
    width = batch.max_seq_len
    before = _rows(batch, 0)

    # Slot 0 owns tokens [0, 2), slot 1 owns [2, 4) — the split a prefix gets
    # when two requests computed different parts of it.
    batch.copy_prefix([(0, 2, 0, 2), (1, 2, 2, 2)])

    after = _rows(batch, 0)
    assert torch.equal(after[2 * width : 2 * width + 2], before[0:2])
    assert torch.equal(after[2 * width + 2 : 2 * width + 4], before[width + 2 : width + 4])


def test_no_segments_touches_nothing():
    """The common case is a miss, which must not cost a kernel launch."""
    batch = _batch()
    before = [_rows(batch, layer) for layer in range(LAYERS)]
    batch.copy_prefix(())
    assert all(torch.equal(_rows(batch, layer), before[layer]) for layer in range(LAYERS))
