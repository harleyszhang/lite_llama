"""Diagnose the b8@64 TBO-graph replay drift the two-rank test exposes.

The engine probe (V2-Lite EP, b128@512) shows parity 0.0 with the same
fork/join capture; this payload reproduces the failing test's shape
(Qwen3-0.6B dense TP, b8@64) outside the gate, then bisects: eager-TBO
determinism, the plain-vs-TBO decomposition gap, and replay-vs-eager with a
per-row pattern.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lite_llama.batch_overlap.comm_overlap import CommStreamPool
from lite_llama.batch_overlap.two_batch_overlap import (
    TBO_ENV,
    TBO_MIN_ROWS_ENV,
    reset_tbo_policy,
)
from lite_llama.executor.cuda_graph import CUDAGraphManager, _GraphKey
from lite_llama.executor.model_runner import ModelRunner
from lite_llama.executor.slot_batch import SlotBatch
from tests.distributed.tp_harness import run_on_tp_ranks

ROOT = Path(__file__).resolve().parent
QWEN = str(ROOT / "my_weight" / "Qwen3-0.6B")
ROWS = 8
SEQ_LENS = [5, 9, 13, 7, 11, 15, 19, 23]


def payload(rank: int) -> dict:
    os.environ[TBO_ENV] = "1"
    os.environ[TBO_MIN_ROWS_ENV] = "2"
    os.environ["LITE_LLAMA_OVERLAP_TIMELINE"] = "0"
    reset_tbo_policy()
    CommStreamPool.reset()

    runner = ModelRunner.build(
        QWEN, max_seq_len=128, max_gpu_num_blocks=4096, use_cuda_graph=True
    )
    slot_batch = SlotBatch(runner)
    device = torch.device("cuda", rank)

    slots = list(range(ROWS))
    table = runner.atten_info.b_req_tokens_table
    for slot in slots:
        table[slot, :64] = torch.arange(64, dtype=table.dtype, device=device) + slot * 64
    for layer_buf in runner.atten_info.kv_buffer:
        layer_buf.fill_(0.25)

    # Manual capture, bypassing the gate's discard.
    manager = CUDAGraphManager(
        runner.model,
        kv_buffer=runner.atten_info.kv_buffer,
        b_req_tokens_table=runner.atten_info.b_req_tokens_table,
        batch_sizes=(ROWS,),
        seq_len_buckets=(64,),
        device=runner.device,
        step_factory=lambda bs: runner._tbo_step(),
    )
    manager.capture_all()
    graph_runner = manager._runners[_GraphKey(ROWS, 64)]

    def step_inputs(token: int):
        slot_batch.begin_decode(slots, SEQ_LENS)
        ids = torch.full((ROWS, 1), token, dtype=torch.long, device=runner.device)
        positions = slot_batch.seq_lens.view(-1, 1) - 1
        return ids, positions

    ids, positions = step_inputs(1000)
    with torch.no_grad():
        e1 = runner.forward_tbo(ids, positions)
        e2 = runner.forward_tbo(ids, positions)
    torch.cuda.synchronize()
    eager_dup = (e1 - e2).abs().max().item()

    with torch.no_grad():
        plain = runner.forward(ids, positions, None)
    torch.cuda.synchronize()
    plain_vs_tbo = (e1 - plain).abs().max().item()

    ai = runner.atten_info
    out = graph_runner.replay(ids, positions, ai.cur_select_index, ai.b_seq_len, ai.b_req_idx)
    torch.cuda.synchronize()
    replay_vs_eager = (e1.float() - out.float()).abs().amax(dim=-1).squeeze(-1)

    from lite_llama.batch_overlap.comm_overlap import CAPTURE_FENCE_STATS
    capture_counts = dict(CAPTURE_FENCE_STATS)

    vocab = runner.model.config.vocab_size
    CAPTURE_FENCE_STATS.update(deferred=0, fenced=0)
    gate_parity = graph_runner.parity_error(vocab)
    gate_counts = dict(CAPTURE_FENCE_STATS)
    CAPTURE_FENCE_STATS.update(deferred=0, fenced=0)

    # Control: a plain-shape graph over the same synthetic inputs. The plain
    # path never forks (its AR rides the capture stream), so a drift here is
    # the attention kernel's own behaviour on the probe grid, not the
    # fork/join capture.
    def capture_in_mode(mode: str) -> float:
        os.environ["LITE_LLAMA_CAPTURE_MODE"] = mode
        try:
            mgr = CUDAGraphManager(
                runner.model,
                kv_buffer=runner.atten_info.kv_buffer,
                b_req_tokens_table=runner.atten_info.b_req_tokens_table,
                batch_sizes=(ROWS,),
                seq_len_buckets=(64,),
                device=runner.device,
                step_factory=lambda bs: runner._tbo_step(),
            )
            mgr.capture_all()
            return mgr._runners[_GraphKey(ROWS, 64)].parity_error(vocab)
        finally:
            os.environ["LITE_LLAMA_CAPTURE_MODE"] = "fork"

    flatten_parity = capture_in_mode("flatten")
    immediate_join_parity = capture_in_mode("immediate_join")
    bcast_parity = capture_in_mode("bcast")

    return {
        "eager_dup": round(eager_dup, 6),
        "plain_vs_tbo": round(plain_vs_tbo, 6),
        "replay_vs_eager_max": round(replay_vs_eager.max().item(), 6),
        "gate_parity": round(gate_parity, 6),
        "capture_counts": capture_counts,
        "gate_counts": gate_counts,
        "flatten_parity": round(flatten_parity, 6),
        "immediate_join_parity": round(immediate_join_parity, 6),
        "bcast_parity": round(bcast_parity, 6),
    }


if __name__ == "__main__":
    print({"results": run_on_tp_ranks(payload, tp_size=2)})
