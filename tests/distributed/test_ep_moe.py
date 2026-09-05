"""Expert-parallel MoE over a real two-rank group.

Two tiers, split by what they need:

* **gloo / CPU** — the EP-specific plumbing that is pure tensor shuffling:
  the ``expert_map`` weight loader (each rank keeps whole experts
  ``[offset, offset+local)`` and skips the rest) and the ``all_to_all``
  dispatch/combine exchange (tokens routed to the owning rank, results routed
  back, sender-side weights applied). No Triton, no device.
* **nccl / 2 GPUs** — the numerics that ride the fused grouped GEMM: a full
  :class:`SparseMoeBlock` forward under EP equals the same routing run through
  ``fused_moe`` with every expert local, and the EP+TBO op stream driven by
  :func:`execute_overlapped_operations` equals two independent sequential forwards (the
  collective-ordering safety the two-batch overlap depends on).

Usage:
    pytest tests/distributed/test_ep_moe.py
"""

from __future__ import annotations

import json
import os
import tempfile

import torch

from rapid_llm.distributed import parallel_state as ps
from tests.distributed.tp_harness import needs_gpus, run_on_tp_ranks

_NUM_EXPERTS = 6
_HIDDEN = 64
_INTER = 32
_TOP_K = 2

#: A minimal DeepSeek-V2 MoE config: no dense layers, no shared expert, six
#: routed experts (EP=2 ⇒ three per rank). Only the MoE fields matter here —
#: the block is built and fed by hand, never through a checkpoint.
_CONFIG_BODY = {
    "model_type": "deepseek_v2",
    "torch_dtype": "bfloat16",
    "hidden_size": _HIDDEN,
    "intermediate_size": 128,
    "moe_intermediate_size": _INTER,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "n_shared_experts": 0,
    "n_routed_experts": _NUM_EXPERTS,
    "num_experts_per_tok": _TOP_K,
    "routed_scaling_factor": 2.5,
    "first_k_dense_replace": 0,
    "norm_topk_prob": False,
    "kv_lora_rank": 16,
    "q_lora_rank": 32,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 64,
    "v_head_dim": 32,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


def _make_config():
    """A ``ModelConfig`` for the MoE body above, from a throwaway config.json."""
    from rapid_llm.models.config import ModelConfig

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as fh:
            json.dump(_CONFIG_BODY, fh)
        return ModelConfig.from_pretrained(d, max_seq_len=128)


#: Realistic weight scale. ``randn`` at std 1.0 chained through the two expert
#: GEMMs blows the outputs up to ~1e4, where bf16 rounding breaches even a 2%
#: relative tolerance on the largest elements; real expert weights sit near 0.05
#: (the repo's own test-init convention), keeping outputs O(0.1) so the parity
#: check measures the EP data movement, not the accumulator's last bits.
_W_STD = 0.05
#: A softer router: std 1.0 gate weights over hidden=64 saturate the softmax to
#: near one-hot (only the top slot contributes), which would leave the k>1
#: weighted sum — the part EP spreads across ranks — untested.
_GATE_STD = 0.1


def _global_expert_weights(dtype, device, seed: int = 1234):
    """Full stacked expert weights + router, identical on every rank (they are the model)."""
    gen = torch.Generator().manual_seed(seed)
    gate_up = (
        torch.randn(_NUM_EXPERTS, 2 * _INTER, _HIDDEN, generator=gen, dtype=torch.float32) * _W_STD
    ).to(dtype=dtype, device=device)
    down = (
        torch.randn(_NUM_EXPERTS, _HIDDEN, _INTER, generator=gen, dtype=torch.float32) * _W_STD
    ).to(dtype=dtype, device=device)
    gate_w = (
        torch.randn(_NUM_EXPERTS, _HIDDEN, generator=gen, dtype=torch.float32) * _GATE_STD
    ).to(dtype=dtype, device=device)
    return gate_up, down, gate_w


# --------------------------------------------------------------------------- #
# gloo / CPU: expert_map weight loading + the a2a exchange
# --------------------------------------------------------------------------- #
def _weight_loader_partitions_experts(rank: int) -> bool:
    """Each rank's stacked experts == its ``[offset, offset+local)`` slice of the global set.

    Non-local experts are skipped (the loader returns an empty view), local ones
    are copied whole — no TP narrow, because EP and TP expert splits are exclusive.
    """
    from rapid_llm.modules.moe import SparseMoeBlock

    config = _make_config()
    block = SparseMoeBlock(config)  # EP active: harness set enable_expert_parallel
    off, nl = block.expert_offset, block.num_local_experts
    assert nl == _NUM_EXPERTS // ps.get_ep_world_size()
    assert off == ps.get_ep_rank() * nl

    dtype = block.experts["gate_up_proj"].dtype
    gate, up = (
        torch.randn(_NUM_EXPERTS, _INTER, _HIDDEN, dtype=dtype),
        torch.randn(_NUM_EXPERTS, _INTER, _HIDDEN, dtype=dtype),
    )
    down = torch.randn(_NUM_EXPERTS, _HIDDEN, _INTER, dtype=dtype)

    gu = block.experts["gate_up_proj"]
    dp = block.experts["down_proj"]
    for e in range(_NUM_EXPERTS):
        block._expert_loader(gu, gate[e], (e, 0))  # gate half
        block._expert_loader(gu, up[e], (e, 1))  # up half
        block._expert_loader(dp, down[e], (e, 2))  # down, whole

    expected_gu = torch.cat([gate[off : off + nl], up[off : off + nl]], dim=1)
    expected_dp = down[off : off + nl]
    return bool(torch.equal(gu.data, expected_gu)) and bool(torch.equal(dp.data, expected_dp))


def _dispatch_combine_routes_across_ranks(rank: int) -> bool:
    """The a2a exchange: tokens reach the owning rank, results return, weights apply.

    Each rank has its *own* tokens (seeded per rank) but the *same* global expert
    matrices. A pure-torch per-expert transform stands in for the fused GEMM (no
    device here). After dispatch→expert→combine every rank's output must equal the
    full weighted sum over its own tokens' slots — the combine lands the complete
    routed result on the origin rank, so no all-reduce follows.
    """
    from rapid_llm.modules.moe import AllToAllDispatcher

    ep = ps.get_ep_world_size()
    r = ps.get_ep_rank()
    nl, off = _NUM_EXPERTS // ep, r * (_NUM_EXPERTS // ep)
    rows, dout = 5, 4

    torch.manual_seed(999)  # shared global expert matrices
    weight = torch.randn(_NUM_EXPERTS, _HIDDEN, dout)
    torch.manual_seed(7 + rank)  # per-rank tokens: catches send/recv rank mixups
    x = torch.randn(rows, _HIDDEN)
    ids = torch.randint(0, _NUM_EXPERTS, (rows, _TOP_K))
    w = torch.rand(rows, _TOP_K)

    dispatcher = AllToAllDispatcher(_NUM_EXPERTS, nl, off)
    handle, local_x, local_ids, _ = dispatcher.dispatch(x, ids, w)
    local_weight = weight[off : off + nl]
    local_out = torch.einsum("nh,nhe->ne", local_x, local_weight[local_ids.reshape(-1)])
    out = dispatcher.combine(handle, local_out)

    ref = torch.zeros(rows, dout)
    for t in range(rows):
        for j in range(_TOP_K):
            ref[t] += w[t, j] * (x[t] @ weight[ids[t, j]])
    return bool(torch.allclose(out, ref, atol=1e-4, rtol=1e-4))


class TestExpertParallelGloo:
    """EP plumbing over a real two-rank gloo group — no device, no Triton."""

    def test_weight_loader_partitions_experts(self):
        both = run_on_tp_ranks(
            _weight_loader_partitions_experts,
            tp_size=2,
            backend="gloo",
            enable_expert_parallel=True,
        )
        assert both == [True, True]

    def test_dispatch_combine_routes_across_ranks(self):
        both = run_on_tp_ranks(
            _dispatch_combine_routes_across_ranks,
            tp_size=2,
            backend="gloo",
            enable_expert_parallel=True,
        )
        assert both == [True, True]


# --------------------------------------------------------------------------- #
# nccl / 2 GPUs: fused-MoE numerics + the TBO op stream
# --------------------------------------------------------------------------- #
def _build_ep_block(device):
    """A SparseMoeBlock under EP with global-seeded weights; returns it plus the globals."""
    from rapid_llm.modules.moe import SparseMoeBlock

    config = _make_config()
    block = SparseMoeBlock(config).to(device)
    dtype = block.experts["gate_up_proj"].dtype
    gate_up, down, gate_w = _global_expert_weights(dtype, device)
    off, nl = block.expert_offset, block.num_local_experts
    block.gate_weight.data.copy_(gate_w)
    block.experts["gate_up_proj"].data.copy_(gate_up[off : off + nl])
    block.experts["down_proj"].data.copy_(down[off : off + nl])
    torch.manual_seed(2024)  # replicated tokens: every rank decodes the same batch
    x = torch.randn(8, _HIDDEN, dtype=dtype, device=device)
    return block.eval(), gate_up, down, x


def _ep_forward_matches_full_experts(rank: int) -> bool:
    """EP forward == the same routing through ``fused_moe`` with every expert local."""
    from rapid_llm.kernels import fused_moe

    device = f"cuda:{rank}"
    block, gate_up, down, x = _build_ep_block(device)
    with torch.no_grad():
        out = block(x.clone())
        weights, ids = block._route(x)
        ref = fused_moe(x.clone(), gate_up, down, weights, ids)
    return bool(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))


def _ep_tbo_op_stream_matches_sequential(rank: int) -> bool:
    """Two interleaved EP op streams == two independent sequential forwards.

    The safety the two-batch overlap rests on: both ranks drive
    :func:`execute_overlapped_operations` through the *same* deterministic
    interleave, so half A's and half B's all-to-all exchanges enter the shared
    EP group in the same stage order on every rank and never cross-pair. Each
    half keeps its own :class:`MoEOpContext` and dispatch handle.
    """
    from rapid_llm.batch_overlap.operations import (
        StateDict,
        YieldOperation,
        execute_overlapped_operations,
    )
    from rapid_llm.modules.moe import MoEOpContext

    device = f"cuda:{rank}"
    block, _gu, _dn, x = _build_ep_block(device)
    mid = x.shape[0] // 2
    xa, xb = x[:mid].contiguous(), x[mid:].contiguous()
    with torch.no_grad():
        ref_a = block(xa.clone())
        ref_b = block(xb.clone())

        def stream(xh):
            ctx = MoEOpContext()

            def op(fn):
                """Consume the stream's tensor, write the result back."""

                def run(state):
                    state.mlp_input = fn(state.pop("mlp_input"), state.moe_ctx)

                return run

            return [
                op(block.op_gate),
                YieldOperation(),
                op(block.op_dispatch_a),
                op(block.op_shared_experts),
                YieldOperation(),
                op(block.op_dispatch_b),
                op(block.op_experts),
                op(block.op_combine_a),
                YieldOperation(),
                op(block.op_combine_b),
            ], StateDict({"mlp_input": xh.clone(), "moe_ctx": ctx})

        ops_a, state_a = stream(xa)
        ops_b, state_b = stream(xb)
        out_a, out_b = execute_overlapped_operations(
            [state_a, state_b], [ops_a, ops_b], delta_stages=[0, 2]
        )

    ok_a = torch.allclose(out_a.mlp_input.float(), ref_a.float(), atol=2e-2, rtol=2e-2)
    ok_b = torch.allclose(out_b.mlp_input.float(), ref_b.float(), atol=2e-2, rtol=2e-2)
    return bool(ok_a and ok_b)


@needs_gpus(2)
class TestExpertParallelNumerics:
    """EP numerics over a two-rank nccl group, on the fused grouped GEMM."""

    def test_forward_matches_full_experts(self):
        both = run_on_tp_ranks(
            _ep_forward_matches_full_experts,
            tp_size=2,
            backend="nccl",
            enable_expert_parallel=True,
        )
        assert both == [True, True]

    def test_tbo_op_stream_matches_sequential(self):
        both = run_on_tp_ranks(
            _ep_tbo_op_stream_matches_sequential,
            tp_size=2,
            backend="nccl",
            enable_expert_parallel=True,
        )
        assert both == [True, True]
