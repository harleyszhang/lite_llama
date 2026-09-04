"""Tests for the sequence-parallelism graph pass and its fused operator.

The :class:`~rapid_llm.distributed.sequence_parallel.SequenceParallelPass` marks
the ``AllReduce->RMSNorm`` seams; the operator
:func:`~rapid_llm.kernels.ops.layernorm.skip_rmsnorm.sequence_parallel_allreduce_rmsnorm`
decomposes each into ``ReduceScatter -> local RMSNorm -> AllGather``.

Correctness is checked against the all-reduce + fused-norm reference on a real
two-rank grid (the decomposition must reproduce the same numbers), and the
single-rank degenerate path against the plain fused norm. Pass recognition is
exercised on a real decoder block without a device.

Usage:
    pytest tests/distributed/test_sequence_parallel.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rapid_llm.distributed import parallel_state as ps
from rapid_llm.distributed.sequence_parallel import (
    SequenceParallelPass,
    is_sequence_parallel,
    sequence_parallel_enabled,
)
from rapid_llm.kernels.ops.layernorm.skip_rmsnorm import (
    fused_add_rmsnorm,
    sequence_parallel_allreduce_rmsnorm,
)
from tests.distributed.tp_harness import needs_gpus, run_on_tp_ranks


@pytest.fixture(autouse=True)
def _reset_grid():
    """Restore the world of one after each test (the grid is module-level state)."""
    yield
    ps.destroy_parallel()


def _tiny_decoder_layer() -> nn.Module:
    """A real ``DecoderLayer`` on the meta device, for pass-recognition tests.

    Built through the production class so the seam the pass looks for
    (``forward_attn_stage`` + ``_post_attention_norm``) is the real one, not a
    stand-in. Meta-device allocation keeps it free — no weights are materialised.
    """
    from transformers import LlamaConfig

    from rapid_llm.executor.loader import init_empty_parameters
    from rapid_llm.models.base import DecoderLayer
    from rapid_llm.models.config import ModelConfig

    hf = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=128,
    )
    config = ModelConfig(hf, max_seq_len=128)
    with init_empty_parameters():
        return DecoderLayer(config)


# --------------------------------------------------------------------------- #
# Pass recognition (single process, no device needed)
# --------------------------------------------------------------------------- #
def test_pass_recognizes_decoder_seam():
    """A DecoderLayer owns the o_proj all-reduce -> post-attention-norm seam."""
    assert SequenceParallelPass._is_ar_rmsnorm_seam(_tiny_decoder_layer())


def test_pass_ignores_non_seam_modules():
    """A module without the two-stage split is not a seam."""
    assert not SequenceParallelPass._is_ar_rmsnorm_seam(nn.Linear(4, 4))
    assert not SequenceParallelPass._is_ar_rmsnorm_seam(nn.Module())


def test_pass_is_noop_without_tp(monkeypatch):
    """A world of one has no peers to scatter across, so the pass marks nothing."""
    monkeypatch.setenv("RAPID_LLM_SEQUENCE_PARALLEL", "1")
    model = nn.ModuleList([_tiny_decoder_layer()])
    # TP world size is 1 here: apply() marks 0 even though the seam exists, and
    # the block stays on the all-reduce path.
    assert SequenceParallelPass().apply(model) == 0
    assert not is_sequence_parallel(model[0])


def test_pass_disabled_by_env(monkeypatch):
    """``RAPID_LLM_SEQUENCE_PARALLEL=0`` switches the pass off."""
    monkeypatch.setenv("RAPID_LLM_SEQUENCE_PARALLEL", "0")
    assert not sequence_parallel_enabled()
    assert SequenceParallelPass().enabled is False


def test_pass_enabled_by_default(monkeypatch):
    """The pass is on unless explicitly disabled."""
    monkeypatch.delenv("RAPID_LLM_SEQUENCE_PARALLEL", raising=False)
    assert sequence_parallel_enabled()
    assert SequenceParallelPass().enabled is True


# --------------------------------------------------------------------------- #
# Degenerate path: a world of one falls back to the plain fused norm
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_sequence_parallel_degenerate_world_of_one():
    """With TP off the decomposition is exactly ``fused_add_rmsnorm``."""
    T, H = 8, 128
    partial = torch.randn(T, H, device="cuda", dtype=torch.float16)
    residual = torch.randn(T, H, device="cuda", dtype=torch.float16)
    weight = torch.randn(H, device="cuda", dtype=torch.float16).abs() + 0.5

    sp_normed, sp_residual = sequence_parallel_allreduce_rmsnorm(
        partial.clone(), residual.clone(), weight, 1e-5
    )
    ref_normed, ref_residual = fused_add_rmsnorm(partial.clone(), residual.clone(), weight, 1e-5)

    torch.testing.assert_close(sp_normed, ref_normed)
    torch.testing.assert_close(sp_residual, ref_residual)


# --------------------------------------------------------------------------- #
# Decomposition math: RS->local-norm->AG == AR->norm (simulated on one device)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_sequence_parallel_decomposition_math():
    """The decomposition equals all-reduce + norm, simulated without a process group.

    Proves the core correctness claim on a single device: RMSNorm and the residual
    add are per-row, so reduce-scattering the partial, norming each rank's token
    segment, and all-gathering reconstructs exactly what norming the fully reduced
    tensor gives. The real collectives (``reduce_scatter``, ``all_gather``) are
    exercised by ``tests/distributed/test_parallel_state.py``; this isolates the
    decomposition arithmetic from the transport.
    """
    T, H, world = 8, 128, 2
    torch.manual_seed(0)
    # Two ranks' row-parallel partial sums (each rank holds its own).
    partials = [torch.randn(T, H, device="cuda", dtype=torch.float16) for _ in range(world)]
    residual = torch.randn(T, H, device="cuda", dtype=torch.float16)
    weight = torch.randn(H, device="cuda", dtype=torch.float16).abs() + 0.5
    eps = 1e-5

    # Reference: all-reduce (sum the partials) then fused add+norm on the full tensor.
    full = partials[0] + partials[1]
    ref_normed, _ = fused_add_rmsnorm(full.clone(), residual.clone(), weight, eps)

    # Simulated SP: each rank reduce-scatters (sums its token shard), norms only
    # that shard, then the shards are all-gathered back in rank order.
    local_len = T // world
    normed_shards = []
    for rank in range(world):
        lo, hi = rank * local_len, (rank + 1) * local_len
        local_partial = (partials[0][lo:hi] + partials[1][lo:hi]).contiguous()
        local_residual = residual[lo:hi].contiguous()
        shard, _ = fused_add_rmsnorm(local_partial, local_residual, weight, eps)
        normed_shards.append(shard)
    sp_normed = torch.cat(normed_shards, dim=0)  # all_gather, rank order

    # Per-row arithmetic is identical, so the shards reconstruct the full norm
    # exactly (within fp16 rounding of the two summation orderings).
    torch.testing.assert_close(sp_normed, ref_normed, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# Correctness on a real two-rank grid: SP == all-reduce + fused norm
# --------------------------------------------------------------------------- #
def _sp_matches_reference_payload(rank: int) -> dict:
    """One rank: the SP decomposition vs the all-reduce + fused-norm reference.

    Each rank holds a *different* row-parallel partial sum (as it would after a
    column-parallel projection), while the residual and the norm weight are
    replicated across ranks (as they are in the model). The reference all-reduces
    the partial and norms the full tensor; the SP path reduce-scatters, norms only
    this rank's token segment, and all-gathers. The two must agree element-wise.

    The reference uses a plain NCCL ``dist.all_reduce`` rather than
    ``tensor_model_parallel_all_reduce``: the latter routes a small payload through
    the TP=2 point-to-point fast path, whose blocking send/recv pair is a separate
    concern from the decomposition under test here.
    """
    import torch.distributed as dist

    from rapid_llm.distributed.parallel_state import get_tensor_model_parallel_group

    device = f"cuda:{rank}"
    T, H = 8, 256
    torch.manual_seed(1000 + rank)  # partial differs per rank
    partial = torch.randn(T, H, device=device, dtype=torch.float16)
    torch.manual_seed(7)  # residual + weight replicated across ranks
    residual = torch.randn(T, H, device=device, dtype=torch.float16)
    weight = torch.randn(H, device=device, dtype=torch.float16).abs() + 0.5
    eps = 1e-5

    full = partial.clone()
    dist.all_reduce(full, op=dist.ReduceOp.SUM, group=get_tensor_model_parallel_group())
    ref_normed, ref_residual = fused_add_rmsnorm(full, residual.clone(), weight, eps)

    sp_normed, sp_residual = sequence_parallel_allreduce_rmsnorm(
        partial.clone(), residual.clone(), weight, eps
    )

    return {
        "normed_close": torch.allclose(sp_normed, ref_normed, rtol=2e-2, atol=2e-2),
        "residual_close": torch.allclose(sp_residual, ref_residual, rtol=2e-2, atol=2e-2),
        "normed_shape": tuple(sp_normed.shape),
        "ref_shape": tuple(ref_normed.shape),
        "normed_max_diff": (sp_normed - ref_normed).abs().max().item(),
    }


@needs_gpus(2)
def test_sequence_parallel_matches_allreduce_reference():
    """``RS->local-norm->AG`` reproduces ``all-reduce + fused-norm`` on every rank.

    This is the core correctness claim of the pass: splitting the all-reduce into
    a reduce-scatter, norming only the local token segment, and all-gathering must
    give bit-for-bit (within fp16 rounding) the same result as norming the fully
    reduced tensor — because RMSNorm and the residual add are per-row.
    """
    results = run_on_tp_ranks(_sp_matches_reference_payload, tp_size=2)
    for rank, r in enumerate(results):
        assert r["normed_shape"] == r["ref_shape"] == (8, 256), rank
        assert r["normed_close"], f"rank {rank}: normed diverged (max diff {r['normed_max_diff']})"
        assert r["residual_close"], f"rank {rank}: residual diverged"


def _pass_marks_under_tp_payload(rank: int) -> int:
    """Build a two-layer model under TP and run the pass; return the seam count."""
    model = nn.ModuleList([_tiny_decoder_layer() for _ in range(2)])
    return SequenceParallelPass().apply(model)


@needs_gpus(2)
def test_pass_marks_seams_under_tp():
    """Under TP=2 the pass marks every decoder seam it finds."""
    counts = run_on_tp_ranks(_pass_marks_under_tp_payload, tp_size=2)
    # Both ranks build the same two-layer model, so both mark two seams.
    assert counts == [2, 2]
