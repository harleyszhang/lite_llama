"""Paged KV-cache fixtures: the pool the production path actually hands a kernel.

:class:`PagedPool` builds a real row table, block table and buffers for
arbitrary sequence lengths, so benchmarks measure the kernel's true
input shapes — not a conveniently contiguous stand-in.

Usage:
    pool = paged_pool(seq_lens=[17, 23, 5])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from microbench import l2_bytes

#: How the rows of one sequence are placed in the pool.
#:
#: ``"fragmented"`` is the steady state of a running server and the default.
#: ``"contiguous"`` exists to *quantify* the difference, not as a stand-in — and
#: on the shapes here that difference measured at roughly 1%, which is itself the
#: useful result: it says a decode regression is not going to be hiding in the
#: gather's access pattern.
Layout = Literal["fragmented", "contiguous"]

#: Pool rows are kept past this multiple of L2. Not because a smaller pool would
#: be served from cache — ``do_bench`` flushes L2 before each replay and a decode
#: step reads each row once, so it would not be — but because a pool this size
#: forces the *attended* working set to be large enough that the kernel is
#: bandwidth-bound rather than launch-bound. Below roughly 8 MiB of attended KV
#: the measured GB/s is a statement about launch latency.
_MIN_L2_MULTIPLE = 8


@dataclass(frozen=True)
class PagedPool:
    """One layer's KV pool plus the tables a decode kernel indexes it through.

    Attributes:
        buffer: The layer tensor, ``[capacity, 2 * num_kv_heads, head_dim]`` —
            the allocation the model owns.
        k: ``buffer[:, :num_kv_heads]``, a strided view. Pass this, not a copy —
            ``.contiguous()`` here silently benchmarks a different allocation
            than the model owns, even where the stride turns out not to cost
            anything.
        v: ``buffer[:, num_kv_heads:]``, same stride.
        table: ``[num_seqs, max_seq_len]`` position-to-row map, the paged
            indirection itself.
        req_idx: ``[num_seqs]`` slot owning each batch row. Identity here; a
            permutation is what continuous batching produces after a request
            finishes, and it is worth a separate case.
        seq_lens: ``[num_seqs]`` history length per row, this step included.
        max_seq_len: Longest row, which sizes the split-K grid.
        layout: Which placement produced ``table``; belongs in every case label.
    """

    buffer: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    table: torch.Tensor
    req_idx: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    layout: Layout

    @property
    def total_tokens(self) -> int:
        """Cached tokens the batch will actually read, summed over sequences."""
        return int(self.seq_lens.sum().item())

    def kv_bytes_read(self) -> int:
        """Minimum KV traffic for one decode step over this pool.

        Every cached K and V element of every attended token, read exactly once.
        Head sharing under GQA does not multiply it: a KV row is one row no
        matter how many query heads consult it, and a kernel that reloads it per
        group is losing bandwidth, not moving more data.
        """
        _, combined_heads, head_dim = self.buffer.shape
        return self.total_tokens * combined_heads * head_dim * self.buffer.element_size()

    def describe(self) -> str:
        """Case label carrying everything the timing depends on."""
        heads = self.buffer.shape[1] // 2
        return (
            f"b{len(self.seq_lens)}_s{self.max_seq_len}_h{heads}"
            f"_d{self.buffer.shape[2]}_{self.layout}"
        )


def paged_pool(
    seq_lens: list[int],
    *,
    num_kv_heads: int,
    head_dim: int,
    capacity: int | None = None,
    dtype: torch.dtype = torch.float16,
    layout: Layout = "fragmented",
    device: str = "cuda",
    seed: int = 0,
) -> PagedPool:
    """Build a pool and table with the production layout.

    Args:
        seq_lens: History length per sequence, this step's token included.
        num_kv_heads: KV heads per layer.
        head_dim: Head size.
        capacity: Pool rows. Defaults to four times the tokens in use, which
            keeps the free list realistically sparse; raised automatically when
            it would put the whole pool inside L2.
        dtype: Cache dtype. ``torch.uint8`` stands for an fp8 e4m3 cache.
        layout: See :data:`Layout`.
        device: CUDA device.
        seed: Fixes the fragmentation pattern, so two runs compare like for like.

    Raises:
        ValueError: When an explicitly requested capacity would leave the
            attended working set small enough to be launch-bound. The default
            capacity is raised silently instead; an explicit one is a stated
            intent, and quietly overriding it would make the printed case label
            a lie.
    """
    torch.manual_seed(seed)
    tokens = sum(seq_lens)
    row_bytes = 2 * num_kv_heads * head_dim * torch.empty((), dtype=dtype).element_size()
    floor_rows = _MIN_L2_MULTIPLE * l2_bytes() // row_bytes
    # The contiguous layout addresses row ``i * max(seq_lens) + pos``, so both
    # layouts must be able to place every sequence at its own base.
    span_rows = len(seq_lens) * max(seq_lens)

    if capacity is None:
        rows = max(4 * tokens, span_rows, floor_rows)
    else:
        rows = max(capacity, span_rows)
        if rows < floor_rows:
            raise ValueError(
                f"pool of {rows} rows is {rows * row_bytes / 2**20:.1f} MiB; at that size "
                f"the gather is launch-bound, not bandwidth-bound "
                f"(need >= {floor_rows} rows)"
            )

    buffer = _pool_bytes(rows, num_kv_heads, head_dim, dtype, device)

    table = _row_table(seq_lens, rows, layout=layout, device=device)
    return PagedPool(
        buffer=buffer,
        # Views, never copies: the stride is part of what is being measured.
        k=buffer[:, :num_kv_heads, :],
        v=buffer[:, num_kv_heads:, :],
        table=table,
        req_idx=torch.arange(len(seq_lens), dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        max_seq_len=max(seq_lens),
        layout=layout,
    )


def _pool_bytes(
    rows: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Fill the pool with values a real cache could hold.

    An fp8 pool goes through the engine's own quantiser rather than a cast to
    ``uint8``: the container holds e4m3 *bit patterns*, and two of them (``0x7f``,
    ``0xff``) are NaN. Random bytes would seed NaNs that propagate through the
    softmax, making the correctness check meaningless while the timing still
    looks fine — a benchmark that prints a plausible GB/s for computing nothing.
    """
    ref = torch.randn(rows, 2 * num_kv_heads, head_dim, device=device, dtype=torch.float16)
    return _as_cache_dtype(ref, dtype)


def _as_cache_dtype(ref: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert fp16 reference values into the cache's storage dtype."""
    if dtype == torch.float16:
        return ref
    if dtype == torch.uint8:
        from lite_llama.modules.quantization.utils import quantize_fp8_per_tensor

        return quantize_fp8_per_tensor(ref, 1.0)
    return ref.to(dtype)


def _row_table(
    seq_lens: list[int],
    capacity: int,
    *,
    layout: Layout,
    device: str,
) -> torch.Tensor:
    """Map each sequence's positions onto pool rows.

    Both layouts hand out disjoint rows — two sequences aliasing one row is a
    correctness bug, not a fragmentation pattern — so the fragmented case is a
    single permutation carved into per-sequence slices rather than independent
    random draws.
    """
    width = max(seq_lens)
    table = torch.zeros(len(seq_lens), width, dtype=torch.int32, device=device)

    if layout == "fragmented":
        perm = torch.randperm(capacity, device=device).to(torch.int32)
        offset = 0
        for i, n in enumerate(seq_lens):
            table[i, :n] = perm[offset : offset + n]
            offset += n
        return table

    for i, n in enumerate(seq_lens):
        base = i * width
        table[i, :n] = torch.arange(base, base + n, dtype=torch.int32, device=device)
    return table


def fresh_rows(
    num_tokens: int,
    *,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(xk, xv)`` a projection just produced, in the layout it produces.

    Separate contiguous tensors on purpose: this is the write side, where K and V
    genuinely arrive as two projection outputs. ``update_kv_buffer`` takes them
    as two pointers precisely so the caller need not build a ``torch.cat`` per
    layer per step — so concatenating them in a benchmark would measure a copy
    the engine does not perform.

    An fp8 request goes through the same quantiser :func:`paged_pool` uses, for
    the same reason: the rows this returns end up in the pool a decode kernel
    then reads.
    """
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    return _as_cache_dtype(k, dtype), _as_cache_dtype(v, dtype)
