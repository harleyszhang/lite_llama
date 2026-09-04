"""Data-parallel scaffolding.

The data-parallel benchmarks share a row shape, a table format, an argument set
and a measurement path, so those live here instead of twice.

Usage:
    from benchmarks.lib import measure_dp, add_dp_args
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .utils import free_gpu, measure_generate


@dataclass
class TimedRow:
    """One measured configuration: wall time plus the tokens produced in it.

    Every data-parallel row is judged on ``tps``; deriving it here keeps the two
    DP scripts' tables on one definition instead of two copies of the same formula.
    """

    latency_s: float
    gen_tokens: int

    @property
    def tps(self) -> float:
        return self.gen_tokens / self.latency_s if self.latency_s else 0.0

    def as_dict(self) -> dict:
        """The row's fields plus ``tps``, for the JSON log."""
        return {**asdict(self), "tps": round(self.tps, 1)}


def add_dp_args(
    parser,
    *,
    default_model: str = "my_weight/Qwen2.5-0.5B",
    default_gen_len: int = 128,
    default_iters: int = 2,
    default_max_num_seqs: int = 0,
    default_max_seq_len: int = 1024,
    default_max_gpu_num_blocks: int | None = None,
    dp_help: str = "Replica count",
    gen_len_help: str = "Tokens per request",
    blocks_help: str = "KV cache tokens per replica; profiled when omitted",
) -> None:
    """The knobs every data-parallel benchmark shares; keyword arguments move defaults.

    A workload that wants a different ``gen_len`` or a *stated* KV pool overrides
    those defaults rather than re-declaring the other six arguments.
    """
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--dp", type=int, default=2, help=dp_help)
    parser.add_argument("--gen-len", type=int, default=default_gen_len, help=gen_len_help)
    parser.add_argument(
        "--iters", type=int, default=default_iters, help="Timed repeats (median reported)"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=default_max_num_seqs,
        help="Replica concurrency ceiling; 0 sizes it to the per-replica batch",
    )
    parser.add_argument("--max-seq-len", type=int, default=default_max_seq_len)
    parser.add_argument(
        "--max-gpu-num-blocks", type=int, default=default_max_gpu_num_blocks, help=blocks_help
    )
    parser.add_argument("--log-dir", default=None, help="Write a JSON log here")


def print_run_header(title: str, fields: dict[str, object], *, width: int = 91) -> None:
    """The banner every run opens with: model, workload knobs, then the device."""
    print(f"\n{'=' * width}")
    print(f"{title}  |  " + "  ".join(f"{k}={v}" for k, v in fields.items()))
    print(f"gpu={torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    print(f"{'=' * width}")


def measure_dp(
    model: str,
    prompts: list[str],
    *,
    dp: int,
    gen_len: int,
    iters: int,
    max_num_seqs: int,
    warmup_prompts: list[str] | None = None,
    **engine_kwargs,
) -> tuple[float, int, list[str], object]:
    """Time one workload through the DP coordinator.

    The engine is built and torn down per row: rows sharing a process would contend
    for KV, which prices the later ones differently from the earlier ones.

    Returns:
        ``(median latency, median output tokens, last round's texts, tokenizer)`` —
        the tokenizer lets a caller replay routing decisions on exactly the ids the
        balancer saw.
    """
    from lite_llama import DataParallelEngine

    with DataParallelEngine(
        model=model, data_parallel_size=dp, max_num_seqs=max_num_seqs, **engine_kwargs
    ) as engine:
        tokenizer = engine.tokenizer
        latency, tokens, texts = measure_generate(
            engine.generate,
            prompts,
            gen_len=gen_len,
            iters=iters,
            tokenizer=tokenizer,
            warmup_prompts=warmup_prompts,
        )
    free_gpu()
    return latency, tokens, texts, tokenizer
