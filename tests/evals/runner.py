"""Offline execution layer shared by every benchmark in :mod:`tests.evals`.

``build_llm`` + ``generate_completions`` run one model over one prompt
set greedily and truncate at stop markers, returning :class:`EvalResult`
rows that the benchmarks score.

Usage:
    llm = build_llm(model_dir); outs = generate_completions(llm, prompts)
"""

from __future__ import annotations

import gc
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from lite_llama import LLM, SamplingParams

# tests/evals/runner.py -> tests/evals -> tests -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_model_dir(model_dir: str) -> Path:
    """Interpret a config's ``model_dir`` relative to the repository root."""
    path = Path(model_dir).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def kv_cache_tokens(batch_size: int, max_seq_len: int) -> int:
    """KV rows one eval batch can possibly occupy, and therefore all it should get.

    Left to itself the engine profiles the cache against free memory and takes
    90% of the device, which is the right answer for a server: unused cache is
    wasted capacity. A benchmark is the opposite workload — one fixed batch at a
    time, and the memory not spent on cache is needed for the *prefill logits*,
    which are ``batch x prompt_len x vocab``. At batch 32 over a 150k-token
    vocabulary that single tensor is 7 GB, so a profiled cache leaves the
    ``lm_head`` projection nothing to allocate into and prefill dies with an OOM.

    One row per token and at most ``max_seq_len`` tokens per sequence, so this
    bound is exact rather than a heuristic. Pinning it also makes a run
    reproducible across machines, since a profiled size depends on whatever else
    happened to be resident on the GPU.
    """
    return batch_size * max_seq_len


@dataclass(frozen=True)
class Generation:
    """What one pass over a prompt list produced.

    ``generated_tokens`` counts the tokens the engine actually decoded, i.e. it
    is measured before :func:`truncate_at_stop` throws the tail away. That is the
    honest denominator for a throughput figure: the tokens past the stop marker
    cost exactly as much to produce as the ones that get scored.
    """

    completions: list[str]
    latency_s: float
    generated_tokens: int


@dataclass(frozen=True)
class EvalResult:
    """One benchmark run against one checkpoint.

    ``accuracy`` is the headline number the threshold is checked against;
    ``invalid_rate`` is the share of completions no answer could be parsed out
    of, which separates "the model is wrong" from "the harness never saw an
    answer" (a truncated generation, or a prompt format the model ignored).
    """

    model_dir: str
    num_questions: int
    num_shots: int
    max_gen_len: int
    batch_size: int
    accuracy: float
    invalid_rate: float
    latency_s: float
    generated_tokens: int
    extra: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def questions_per_second(self) -> float:
        return self.num_questions / self.latency_s if self.latency_s else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.latency_s if self.latency_s else 0.0

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "questions_per_second": self.questions_per_second,
            "tokens_per_second": self.tokens_per_second,
        }

    def report(self) -> str:
        return (
            f"accuracy      {self.accuracy:.4f}\n"
            f"invalid_rate  {self.invalid_rate:.4f}\n"
            f"questions     {self.num_questions} ({self.num_shots}-shot, "
            f"batch {self.batch_size}, max_gen_len {self.max_gen_len})\n"
            f"latency       {self.latency_s:.1f} s "
            f"({self.questions_per_second:.2f} q/s, {self.tokens_per_second:.1f} tok/s)"
        )


def truncate_at_stop(text: str, stop: Sequence[str]) -> str:
    """Cut ``text`` at the earliest stop marker, emulating a server-side stop.

    Returns the text unchanged when no marker occurs. The marker itself is
    dropped, matching the OpenAI semantics vLLM's harness relies on.
    """
    end = len(text)
    for marker in stop:
        found = text.find(marker)
        if found != -1:
            end = min(end, found)
    return text[:end]


@contextmanager
def build_llm(
    model_dir: str | Path,
    *,
    max_seq_len: int = 2048,
    max_gpu_num_blocks: int | None = None,
    use_cuda_graph: bool | None = None,
    device: str = "cuda",
) -> Iterator[LLM]:
    """Build an :class:`~lite_llama.LLM` and free the device on the way out.

    The engine, executor and KV manager reference each other, so dropping the
    last name is not enough to release the weights; without the explicit
    collection a second checkpoint built in the same process profiles a KV cache
    against memory the first one still holds.
    """
    llm = LLM(
        model=str(model_dir),
        max_seq_len=max_seq_len,
        max_gpu_num_blocks=max_gpu_num_blocks,
        use_cuda_graph=use_cuda_graph,
        device=device,
    )
    try:
        yield llm
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def greedy_params(max_gen_len: int) -> SamplingParams:
    """Sampling settings a benchmark should be scored under.

    Greedy, and with both of lite_llama's chat-facing defaults switched off:
    ``repetition_penalty`` rescales logits and ``stop_on_repeat`` can cut a
    sequence short, so leaving either on would score a decoding policy rather
    than the model. They exist to keep small models out of loops in interactive
    use; a benchmark wants the raw argmax.
    """
    return SamplingParams(
        temperature=0.0,
        max_gen_len=max_gen_len,
        repetition_penalty=1.0,
        stop_on_repeat=False,
    )


def as_user_turn(llm: LLM, prompt: str) -> str:
    """Wrap ``prompt`` in the checkpoint's chat template as a single user turn.

    The counterpart of vLLM's ``use_chat_completions``: instruction-tuned
    checkpoints answer a bare completion prompt poorly, because the format they
    were tuned on is the template, not raw text. Base checkpoints have no
    template and must not go through here.
    """
    return llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def generate_completions(
    llm: LLM,
    prompts: Sequence[str],
    *,
    max_gen_len: int,
    batch_size: int,
    stop: Sequence[str] = (),
    progress: bool = True,
) -> Generation:
    """Greedily complete every prompt, in input order.

    Args:
        llm: A built engine; reused across all batches.
        prompts: One prompt per question.
        max_gen_len: Decode steps per batch. Every sequence runs the full count
            unless it hits EOS, so this is the dominant cost term.
        batch_size: Prompts per :meth:`~lite_llama.LLM.generate` call.
        stop: Markers to truncate each completion at.
        progress: Show a per-batch progress bar.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    params = greedy_params(max_gen_len)
    chunks = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

    completions: list[str] = []
    generated_tokens = 0
    start = time.perf_counter()
    for chunk in tqdm(chunks, desc="generating", unit="batch", disable=not progress):
        for out in llm.generate(list(chunk), params):
            raw = out.outputs[0].text
            generated_tokens += len(llm.tokenizer.encode(raw, add_special_tokens=False))
            completions.append(truncate_at_stop(raw, stop))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - start

    return Generation(completions, latency, generated_tokens)
