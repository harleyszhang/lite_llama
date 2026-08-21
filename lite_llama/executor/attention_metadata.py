"""Runtime attention / KV-cache bookkeeping passed through the model forward pass.

One dataclass threaded into every layer's attention: the paged ``kv_buffer``, the
cache rows written this step (``cur_select_index``), the request->token map and the
per-sequence lengths. Carrying it as a single object keeps the attention kernels'
signatures stable and lets CUDA-graph replay mutate its fields in place.

Usage:
    attn = AttentionMetadata(kv_buffer=..., cur_select_index=..., b_seq_len=...)
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
    """

    kv_buffer: list[torch.Tensor] = field(default_factory=list)
    cur_select_index: torch.Tensor | None = None
    b_req_tokens_table: torch.Tensor | None = None
    b_start_loc: torch.Tensor | None = None
    b_req_idx: torch.Tensor | None = None
    b_seq_len: torch.Tensor | None = None
    max_actual_seq_len: int = 0
