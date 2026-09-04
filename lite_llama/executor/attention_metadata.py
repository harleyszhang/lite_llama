"""Runtime attention / KV-cache bookkeeping passed through the forward pass.

:class:`AttentionMetadata` is the one dataclass every attention call
receives: the paged ``kv_buffer``, this step's cache rows, the
request->token table and per-sequence lengths — so kernel signatures
stay stable and graph replay can mutate fields in place.

Usage:
    attn = AttentionMetadata(kv_buffer=..., cur_select_index=...)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class AttentionMetadata:
    """Per-step KV-cache state shared by the model runner and the attention kernels.

    A single instance is reused across prefill and every decode step; the runner
    mutates the index/length fields in place before each ``forward``.

    Attributes:
        kv_buffer: One paged KV tensor per layer, each
            ``[max_tokens, 2 * num_kv_heads, head_dim]``.
        cur_select_index: Cache rows written by the current step.
        b_req_tokens_table: ``[max_requests, max_seq_len]`` map from request/position
            to cache row, used by the decode kernel to gather history.
        b_start_loc: Prefill only — start offset of each sequence in the flattened batch.
        b_req_idx: Request ids active this step.
        b_seq_len: Current length of each sequence (grows by one per decode step).
        max_actual_seq_len: Longest sequence length seen so far this generation.
        is_prefill: Whether this step is a prefill, set by whoever prepares the
            metadata. The kernels differ between phases (prefill runs causal attention
            over fresh K/V, decode gathers history from the paged buffer), and deriving
            the phase from ``seq_len > 1`` would misroute a single-token prompt to decode.
        b_prefix_len: Chunked prefill only — cached rows preceding each
            sequence's chunk (a resumed chunk or a prefix-cache hit). ``None``
            on every other pass, which is what routes ``context_forward``
            between the plain and the chunk-aware prefill kernel.
        b_kv_base: Chunked prefill only — cache row holding each sequence's
            KV row 0; the chunk kernel addresses a slot's history as one
            contiguous run from this base, no per-token table lookup.
        max_chunk_len: Chunked prefill only — widest chunk in the grid,
            sizing the query-block grid.
        b_seq_len_cpu: Host mirror of ``b_seq_len`` for the current decode
            step, written by whoever drives the host-side loop (the one-shot
            session's position arithmetic or the continuous batcher's
            scheduler). Lets a per-step preparation hook read the lengths
            without a device sync; ``None`` outside decode steps.
        decode_plan: Per-step payload produced once by the winning backend's
            preparation hook (``KernelSpec.step_prepare``) and read by every
            layer's kernel in that step — attention metadata is
            layer-invariant within a step. ``None`` unless a hook ran.
    """

    kv_buffer: list[torch.Tensor] = field(default_factory=list)
    cur_select_index: torch.Tensor | None = None
    b_req_tokens_table: torch.Tensor | None = None
    b_start_loc: torch.Tensor | None = None
    b_req_idx: torch.Tensor | None = None
    b_seq_len: torch.Tensor | None = None
    max_actual_seq_len: int = 0
    is_prefill: bool = True
    b_prefix_len: torch.Tensor | None = None
    b_kv_base: torch.Tensor | None = None
    max_chunk_len: int = 0
    b_seq_len_cpu: torch.Tensor | None = None
    decode_plan: object | None = None
