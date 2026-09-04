"""Per-layer sliding-window cache for DeepSeek-V4.

V4 bypasses the paged KV store: its sliding K==V window, the compressor
rolling buffers and the per-entry bookkeeping do not fit the paged layout, so
:class:`V4LayerCache` keeps them row-indexed instead and ``atten_info`` only
contributes step metadata (``is_prefill``, ``b_seq_len``). The engine still
allocates a (unused) paged cache for V4 — first-version cost, documented in
the release notes.
"""

from __future__ import annotations

import torch


class V4LayerCache:
    """Sliding-window K==V state plus the compressor bookkeeping hooks.

    Row-indexed (``rows[b]`` is sequence ``b``), with one structural
    assumption documented in the module docstring: batch row ``b`` maps to
    the same sequence across a prefill -> decode run; a scheduler that reuses
    a row for a *new* request must call :meth:`reset` first. The compressor /
    indexer modules keep their own per-row rolling state the same way.

    The sliding window keeps ``sliding_window - 1`` entries between steps, so
    together with the current token the attention sees exactly the
    ``sliding_window`` most recent positions — matching the reference's
    ``full[:, :, -sliding_window + 1:, :]`` retention. Entries whose stored
    position is ``-1`` are padding slots excluded by the attention mask.
    """

    def __init__(self, sliding_window: int) -> None:
        self.sliding_window = sliding_window
        self.sliding_kv: torch.Tensor | None = None  # [B, T, head_dim]
        self.sliding_pos: torch.Tensor | None = None  # [B, T], -1 = padding
        self.rows: list[dict] = []

    def ensure_rows(self, batch_size: int) -> None:
        while len(self.rows) < batch_size:
            self.rows.append({})

    def reset(self) -> None:
        self.sliding_kv = None
        self.sliding_pos = None
        self.rows = []

    def update_sliding(
        self, kv: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's K==V and keep the trailing window.

        Args:
            kv: ``[B, S, head_dim]`` fresh keys/values (padding slots hold
                garbage; their ``pos`` is ``-1`` so they never enter the mask).
            pos: ``[B, S]`` absolute positions, ``-1`` where padded.

        Returns:
            The *full* concatenation (old window + fresh step) that the
            attention reads this step, plus its positions — the reference
            semantics, where the returned tensor is un-truncated while the
            retained state shrinks.
        """
        if self.sliding_kv is None:
            full_kv, full_pos = kv, pos
        else:
            full_kv = torch.cat([self.sliding_kv, kv], dim=1)
            full_pos = torch.cat([self.sliding_pos, pos], dim=1)
        total = full_pos.shape[1]
        cap = self.sliding_window - 1
        keep = min(cap, total)
        # Largest positions win; ties are impossible (positions are unique
        # per row, padding is -1 and sorts below everything real).
        _, idx = torch.topk(full_pos, keep, dim=1)
        idx, _ = torch.sort(idx, dim=1)  # topk is descending; restore time order
        # The gather index covers ``keep`` rows, not the full concatenation —
        # once the window rolls (total > cap) the two differ.
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, full_kv.shape[-1])
        self.sliding_kv = full_kv.gather(1, gather_idx)
        self.sliding_pos = full_pos.gather(1, idx)
        return full_kv, full_pos


__all__ = ["V4LayerCache"]
