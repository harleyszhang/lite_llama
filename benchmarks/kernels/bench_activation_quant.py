"""Activation quantisation: the per-token-group kernel vs its alternatives.

Rows: the new per-group kernel (int8 and fp8-e4m3 bytes, one pass), the
per-token fp8 quantiser the W8A8 GEMMs embed (its two-pass walk of the row),
the eager torch chain the kernel replaced, and the ``silu·mul`` fusion vs the
two-kernel spelling. ``_check`` dequantises every implementation and diffs it
against the reference before anything is timed.

Usage:
    python benchmarks/kernels/bench_activation_quant.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from microbench import Row, Work, bench, metadata, report, require_cuda

from lite_llama.kernels.ops.quantization import fp8_quantize_per_token, per_token_group_quant

#: Scale granularity under test — the ``block_shape=[128, 128]`` convention.
GROUP_SIZE = 128


def _silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """Eager ``silu(gate) * up`` reference for the fused rows."""
    h = gate_up.shape[-1] // 2
    gate, up = gate_up[:, :h].float(), gate_up[:, h:].float()
    return gate * torch.sigmoid(gate) * up


def torch_group_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The eager chain ``per_token_group_quant`` replaced.

    One op per step — reshape, abs, amax, clamp, div, clamp, cast — and at
    decode shapes the chain's launch overhead *is* the cost, which is exactly
    the decomposition the fused per-token quantiser was built to remove.
    """
    k = x.shape[-1]
    g = k // GROUP_SIZE
    grouped = x.reshape(-1, k).float().view(-1, g, GROUP_SIZE)
    scale = grouped.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    q = (grouped / scale[:, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).reshape(-1, k), scale


def _dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``q`` back to fp32 through ``s`` — per-token ``[T, 1]`` or per-group ``[T, G]``."""
    deq = q.view(torch.float8_e4m3fn).float() if q.dtype is torch.uint8 else q.float()
    if s.shape[-1] == 1:
        return deq * s
    return deq * s.repeat_interleave(GROUP_SIZE, dim=-1)


def _check(name: str, q: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> None:
    """Assert the dequantised result lands within one quantisation step of ``ref``."""
    deq = _dequant(q, s)
    if q.dtype is torch.int8:
        torch.testing.assert_close(deq, ref, rtol=0.0, atol=0.51 * s.max().item())
    else:
        torch.testing.assert_close(deq, ref, rtol=1.0 / 16, atol=2 * s.max().item())
    print(f"  {name}: dequant within one step of reference")


#: Quantisers of a ``[T, K]`` activation. Add a row to register a new one.
QUANT_PROVIDERS: dict[str, Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = {
    "per-group int8 (1-pass)": lambda x: per_token_group_quant(x, GROUP_SIZE, out_dtype=torch.int8),
    "per-group fp8 (1-pass)": lambda x: per_token_group_quant(x, GROUP_SIZE, out_dtype=torch.uint8),
    "per-token fp8 (2-pass)": fp8_quantize_per_token,
    "torch chain (per-group fp8)": torch_group_quant_fp8,
}

#: The ``silu·mul`` fusion on a ``[T, 2H]`` gate/up buffer, vs the spelling
#: that round-trips the intermediate through HBM first.
FUSED_PROVIDERS: dict[str, Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = {
    "fused silu·mul + group fp8": lambda g: per_token_group_quant(
        g, GROUP_SIZE, out_dtype=torch.uint8, fuse_silu_and_mul=True
    ),
    "silu·mul → group fp8": lambda g: per_token_group_quant(
        _silu_and_mul(g).to(g.dtype), GROUP_SIZE, out_dtype=torch.uint8
    ),
}

#: (label, tokens) — a decode step, a short prefill, and a full one, at Llama-3-8B
#: hidden width (4096); the fused rows run on the FFN gate/up buffer (2 * 4096).
CASES = [("t1 h4096", 1), ("t128 h4096", 128), ("t2048 h4096", 2048)]


def main() -> None:
    require_cuda()
    print(metadata())

    rows: list[Row] = []
    torch.manual_seed(0)
    for label, tokens in CASES:
        x = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16) * 3.0
        gate_up = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16) * 3.0

        print(f"\n{label}: verifying dequantised output before timing anything")
        for name, fn in QUANT_PROVIDERS.items():
            q, s = fn(x)
            _check(name, q, s, x.float())
        ref = _silu_and_mul(gate_up)
        for name, fn in FUSED_PROVIDERS.items():
            q, s = fn(gate_up)
            _check(name, q, s, ref)

        # Theoretical minimum traffic for each op: every input byte read once,
        # every byte written once. Both rows share it within their group, so
        # only the measured time separates an implementation from its peers.
        # 4 bytes/element for the scale grid at group_size=128.
        work_quant = Work(moved=tokens * 4096 * 3 + tokens * 32 * 4)
        work_fused = Work(moved=tokens * 8192 * 2 + tokens * 4096 + tokens * 32 * 4)
        for name, fn in QUANT_PROVIDERS.items():
            rows.append(Row(name, f"{label} quant", bench(lambda fn=fn, x=x: fn(x)), work_quant))
        for name, fn in FUSED_PROVIDERS.items():
            rows.append(
                Row(
                    name,
                    f"{label} fused",
                    bench(lambda fn=fn, gate_up=gate_up: fn(gate_up)),
                    work_fused,
                )
            )

    print()
    report(rows)


if __name__ == "__main__":
    main()
