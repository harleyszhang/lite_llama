"""Tests for the KV-cache scatter kernels.

These two kernels are pure bookkeeping, which is exactly why they are dangerous:
they produce no numbers of their own, so a wrong index silently poisons the
attention that reads the cache several layers later. Neither has a natural
"looks wrong" signature.

* ``update_kv_buffer`` scatters freshly computed K/V rows into the pool:
  ``KV_Buffer[Select_Index[i]] = KV_Values[i]``. The rows it must *not* touch
  matter as much as the ones it writes, so untouched rows are asserted too.
* ``update_kv_index`` records where a token landed:
  ``req_to_token_indexs[b_req_idx[i]][b_seq_len[i] - 1] = select_index[i]``.
  The ``- 1`` is the whole subtlety: ``b_seq_len`` is the length *including* the
  new token, so the write goes to the last occupied slot, not one past it.
"""

from __future__ import annotations

import pytest
import torch

from lite_llama.kernels import update_kv_buffer, update_kv_index


# --------------------------------------------------------------------------- #
# update_kv_buffer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "num_tokens,num_kv_heads,head_dim",
    [
        pytest.param(1, 2, 64, id="decode-single-token"),
        pytest.param(16, 4, 64, id="prefill-small"),
        pytest.param(37, 2, 128, id="ragged-count"),
        pytest.param(64, 8, 32, id="many-heads"),
    ],
)
def test_scatter_writes_selected_rows(num_tokens, num_kv_heads, head_dim):
    """Every source row must land at its selected destination, verbatim."""
    pool_rows = 256
    combined_heads = 2 * num_kv_heads  # K heads then V heads share one tensor

    values = torch.randn(num_tokens, combined_heads, head_dim, device="cuda", dtype=torch.float16)
    buffer = torch.zeros(pool_rows, combined_heads, head_dim, device="cuda", dtype=torch.float16)
    select = torch.randperm(pool_rows, device="cuda")[:num_tokens].to(torch.int32)

    update_kv_buffer(values, select, buffer)

    torch.testing.assert_close(buffer[select.long()], values)


def test_scatter_leaves_other_rows_untouched():
    """Rows outside ``Select_Index`` must keep their previous contents.

    A kernel that wrote a whole block instead of the selected rows would still
    pass the positive assertion above while trampling a neighbour's history.
    """
    pool_rows, heads, head_dim = 64, 4, 64
    buffer = torch.full((pool_rows, heads, head_dim), 7.0, device="cuda", dtype=torch.float16)
    values = torch.randn(4, heads, head_dim, device="cuda", dtype=torch.float16)
    select = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device="cuda")

    update_kv_buffer(values, select, buffer)

    untouched = torch.ones(pool_rows, dtype=torch.bool, device="cuda")
    untouched[select.long()] = False
    assert (buffer[untouched] == 7.0).all()


def test_scatter_to_non_monotonic_destinations():
    """Destinations are arbitrary: descending order must work like any other.

    Pins that the kernel indexes through the table rather than assuming the
    selection is sorted or contiguous.
    """
    heads, head_dim = 2, 64
    buffer = torch.zeros(32, heads, head_dim, device="cuda", dtype=torch.float16)
    values = torch.randn(4, heads, head_dim, device="cuda", dtype=torch.float16)
    select = torch.tensor([31, 3, 17, 0], dtype=torch.int32, device="cuda")

    update_kv_buffer(values, select, buffer)

    for src, dst in enumerate(select.tolist()):
        torch.testing.assert_close(buffer[dst], values[src])


# --------------------------------------------------------------------------- #
# update_kv_index
# --------------------------------------------------------------------------- #
def test_index_records_token_at_last_occupied_slot():
    """``b_seq_len`` counts the new token, so the write lands at ``len - 1``.

    Off-by-one here would make the next decode step read a stale or unwritten
    slot as the most recent token.
    """
    num_requests, max_seq_len = 4, 16
    table = torch.full((num_requests, max_seq_len), -1, dtype=torch.int32, device="cuda")

    b_req_idx = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor([1, 5, 16], dtype=torch.int32, device="cuda")
    select_index = torch.tensor([100, 200, 300], dtype=torch.int32, device="cuda")

    update_kv_index(table, b_req_idx, b_seq_len, select_index)

    assert table[0, 0].item() == 100  # first token of request 0
    assert table[2, 4].item() == 200  # fifth token of request 2
    assert table[3, 15].item() == 300  # last slot of request 3


def test_index_leaves_unrelated_entries_untouched():
    """Only one cell per request may change; request 1 was not in the batch."""
    table = torch.full((4, 8), -1, dtype=torch.int32, device="cuda")
    b_req_idx = torch.tensor([0], dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor([3], dtype=torch.int32, device="cuda")
    select_index = torch.tensor([42], dtype=torch.int32, device="cuda")

    update_kv_index(table, b_req_idx, b_seq_len, select_index)

    assert table[0, 2].item() == 42
    assert (table[1:] == -1).all()
    # Within request 0, the other slots stay untouched.
    assert (table[0, [0, 1, 3, 4, 5, 6, 7]] == -1).all()


def test_index_supports_out_of_order_request_ids():
    """Request ids need not be sorted or dense; the batch order is arbitrary."""
    table = torch.full((6, 8), -1, dtype=torch.int32, device="cuda")
    b_req_idx = torch.tensor([5, 1, 3], dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor([2, 7, 4], dtype=torch.int32, device="cuda")
    select_index = torch.tensor([11, 22, 33], dtype=torch.int32, device="cuda")

    update_kv_index(table, b_req_idx, b_seq_len, select_index)

    assert table[5, 1].item() == 11
    assert table[1, 6].item() == 22
    assert table[3, 3].item() == 33


def test_scatter_then_gather_round_trips():
    """The two kernels compose: what is scattered is what the table points at.

    This is the contract decode depends on -- writing a token's K/V into the
    pool and recording its slot must agree, or attention reads someone else's
    tensor. Testing them together is the only way to catch a mismatch in the
    convention they share.
    """
    heads, head_dim, pool_rows = 4, 64, 128
    buffer = torch.zeros(pool_rows, heads, head_dim, device="cuda", dtype=torch.float16)
    table = torch.full((2, 8), -1, dtype=torch.int32, device="cuda")

    values = torch.randn(2, heads, head_dim, device="cuda", dtype=torch.float16)
    select_index = torch.tensor([77, 12], dtype=torch.int32, device="cuda")
    b_req_idx = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor([3, 6], dtype=torch.int32, device="cuda")

    update_kv_buffer(values, select_index, buffer)
    update_kv_index(table, b_req_idx, b_seq_len, select_index)

    # Follow the table back to the pool, exactly as flash_decoding does.
    torch.testing.assert_close(buffer[table[0, 2].item()], values[0])
    torch.testing.assert_close(buffer[table[1, 5].item()], values[1])
