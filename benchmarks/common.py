"""Shared benchmark layer: one metrics vocabulary, one timing discipline.

``BenchResult`` plus ``steps_to_result`` define every script's numbers, the
:class:`Backend` ABC adapts engines into one interface, and the helpers
(``expand_prompts``, ``sampling_params``) keep workload shapes reproducible
across scripts.

Usage:
    from benchmarks.common import BenchResult, print_table
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Workload definitions: what to generate, and how to sample
# --------------------------------------------------------------------------- #
PROMPTS = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
]

#: Non-greedy defaults, matching lite_llama's sampling branch.
SAMPLE_KW = {"temperature": 0.7, "top_p": 0.8}

#: Greedy, with repetition penalty and early exit off. A benchmark's token count
#: must not depend on a heuristic that fires for some rows and not others — that
#: would give the two columns different denominators.
GREEDY_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "stop_on_repeat": False,
}


def expand_prompts(prompts: list[str], batch: int) -> list[str]:
    """Cycle ``prompts`` up to ``batch`` entries."""
    return (prompts * ((batch // len(prompts)) + 1))[:batch]


def gpu_tag() -> str:
    """Filename-safe GPU tag: ``NVIDIA H100 80GB HBM3`` -> ``h100``.

    Vendor and memory words (``80gb``, ``hbm3``) are dropped, the rest is
    lowercased and joined. ``cpu`` without CUDA, ``gpu`` if nothing survives.
    """
    if not torch.cuda.is_available():
        return "cpu"

    vendor_words = {"nvidia", "geforce", "tesla", "quadro"}

    def is_model_word(word: str) -> bool:
        return (
            word not in vendor_words  # vendor / product line
            and not word.endswith("gb")  # VRAM capacity, e.g. 80gb
            and "hbm" not in word  # VRAM type, e.g. hbm3
        )

    # "-" is a separator too: "A100-SXM4-80GB" drops only its memory segment.
    words = torch.cuda.get_device_name(0).lower().replace("-", " ").split()
    return "".join(word for word in words if is_model_word(word)) or "gpu"


def sampling_params(max_gen_len: int, greedy: bool = True):
    """The benchmark's ``SamplingParams``: :data:`GREEDY_PARAMS` or :data:`SAMPLE_KW`."""
    from lite_llama import SamplingParams

    return SamplingParams(max_gen_len=max_gen_len, **(GREEDY_PARAMS if greedy else SAMPLE_KW))


# --------------------------------------------------------------------------- #
# Metric value object
# --------------------------------------------------------------------------- #
@dataclass
class BenchResult:
    """One benchmark measurement. ``gen_tokens`` is the throughput denominator."""

    ttft_ms: float
    tpot_ms: float
    total_s: float
    steps: int
    batch: int
    gen_tokens: int
    tpot_p50_ms: float = 0.0  # only backends that time every step can supply this

    @property
    def tps(self) -> float:
        return self.gen_tokens / self.total_s if self.total_s else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "tps": self.tps}

    def row(self, label: str) -> str:
        return (
            f"{label:18s} TTFT {self.ttft_ms:7.1f} ms | "
            f"TPOT {self.tpot_ms:6.2f} ms | "
            f"TPS {self.tps:7.1f} tok/s | "
            f"{self.gen_tokens} tok in {self.total_s:.2f}s"
        )


def steps_to_result(
    step_ends: list[float],
    *,
    t_start: float,
    total_s: float,
    batch: int,
    gen_tokens: int | None = None,
) -> BenchResult:
    """Fold per-step completion timestamps into a :class:`BenchResult`.

    TTFT is the first step's end minus submission time; TPOT is the mean interval
    of the steps after it. Every step-driven backend goes through this function,
    which is what makes their numbers comparable.

    Args:
        step_ends: ``perf_counter()`` at the end of each step.
        t_start: Submission time (taken after ``torch.cuda.synchronize()``).
        total_s: Whole-run wall clock, computed by the caller after the final sync.
        batch: Concurrent request count.
        gen_tokens: Tokens actually produced; omitted means lockstep advance
            (``batch`` per step).
    """
    deltas = [b - a for a, b in itertools.pairwise(step_ends)]
    return BenchResult(
        ttft_ms=(step_ends[0] - t_start) * 1000 if step_ends else 0.0,
        tpot_ms=(statistics.mean(deltas) * 1000) if deltas else 0.0,
        tpot_p50_ms=(statistics.median(deltas) * 1000) if deltas else 0.0,
        total_s=total_s,
        steps=len(step_ends),
        batch=batch,
        gen_tokens=len(step_ends) * batch if gen_tokens is None else gen_tokens,
    )


def print_table(results: dict[str, BenchResult]) -> None:
    for label, r in results.items():
        print(r.row(label))


# --------------------------------------------------------------------------- #
# Measurement strategies
# --------------------------------------------------------------------------- #
class Backend(ABC):
    """One measured system.

    A subclass answers two questions: how the same prompt set becomes a
    :class:`BenchResult`, and how to tear itself down. ``texts()`` serves the
    accuracy comparisons — the same run's output, so nothing is measured twice.
    """

    @abstractmethod
    def measure(self, prompts: list[str], max_gen_len: int, greedy: bool) -> BenchResult:
        """Run the whole workload and return its metrics (the implementation warms up)."""

    @property
    def runner(self):
        """This backend's ``ModelRunner`` (memory and KV capacity come from it)."""
        raise NotImplementedError(f"{type(self).__name__} holds no ModelRunner")

    def texts(self) -> list[str]:
        """The completions of the last :meth:`measure`; empty for text-less backends."""
        return []

    def close(self) -> None:
        """Return the GPU memory. Required before building a second backend in the
        same process, or its KV budget profiles as zero."""
        free_gpu()


class LiteBackend(Backend):
    """Single-process lite_llama: a stream callback per step splits TTFT from TPOT.

    The batch advances in lockstep, so ``gen_tokens = steps * batch``. Constructor
    arguments pass straight through to ``TextGenerator`` (no defaults here, so this
    never overrides its own).
    """

    def __init__(self, model_dir: str, use_cuda_graph: bool, **gen_kwargs):
        from lite_llama import TextGenerator

        self._gen = TextGenerator(
            checkpoints_dir=model_dir,
            use_cuda_graph=use_cuda_graph,
            **gen_kwargs,
        )
        self._texts: list[str] = []

    @property
    def generator(self):
        """The bare generator: ``bench_e2e --verify`` calls ``generate()`` directly."""
        return self._gen

    @property
    def runner(self):
        """``ModelRunner``: memory footprint and KV pool capacity are read from it."""
        return self._gen.engine.model_runner

    def measure(self, prompts: list[str], max_gen_len: int, greedy: bool) -> BenchResult:
        # Warm up autotune + allocator so the measured run is steady state.
        for _ in range(2):
            list(self._gen.stream(prompts, sampling_params(8)))

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        step_texts: list[list[str]] = []
        step_ends: list[float] = []
        for deltas in self._gen.stream(prompts, sampling_params(max_gen_len, greedy)):
            step_ends.append(time.perf_counter())
            step_texts.append(list(deltas))
        torch.cuda.synchronize()
        total = time.perf_counter() - t_start

        self._texts = ["".join(step[i] for step in step_texts) for i in range(len(prompts))]
        return steps_to_result(step_ends, t_start=t_start, total_s=total, batch=len(prompts))

    def texts(self) -> list[str]:
        return self._texts

    def close(self) -> None:
        del self._gen
        super().close()


class EngineBackend(Backend):
    """Continuous-batching engine: submit the whole batch, then time each step.

    The only path that can drive ``tp > 1`` — its executor broadcasts each step's
    plan to the follower ranks.
    """

    def __init__(self, model_dir: str, *, tensor_parallel_size: int = 1, **engine_kwargs):
        from lite_llama.engine import ContinuousBatchingEngine

        self._engine = ContinuousBatchingEngine.from_pretrained(
            model_dir, tensor_parallel_size=tensor_parallel_size, **engine_kwargs
        )
        self.tensor_parallel_size = tensor_parallel_size
        self._texts: list[str] = []

    @property
    def runner(self):
        return self._engine.engine.model_runner

    def measure(self, prompts: list[str], max_gen_len: int, greedy: bool) -> BenchResult:
        self._engine.generate(prompts, sampling_params(8))  # warm up autotune + allocator

        params = sampling_params(max_gen_len, greedy)
        requests = [self._engine.add_request(prompt, params) for prompt in prompts]
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        step_ends: list[float] = []
        while self._engine.has_unfinished_requests():
            self._engine.step()
            step_ends.append(time.perf_counter())
        torch.cuda.synchronize()
        total = time.perf_counter() - t_start

        self._texts = [r.text for r in requests]
        return steps_to_result(
            step_ends,
            t_start=t_start,
            total_s=total,
            batch=len(prompts),
            # Requests leave on their own EOS, so ``steps * batch`` would overcount:
            # sum the tokens each request actually produced.
            gen_tokens=sum(len(r.output_token_ids) for r in requests),
        )

    def texts(self) -> list[str]:
        return self._texts

    def close(self) -> None:
        self._engine.shutdown()
        del self._engine
        super().close()


class VisionBackend(Backend):
    """Multimodal measurement: ``VisionGenerator`` serves one request at a time.

    lite_llama's multimodal path is serial (the processor takes one request), so
    ``--batch`` becomes the number of serial requests: TTFT is the mean of each
    request's first-token latency, TPOT the mean of all decode-step intervals, TPS
    the aggregate throughput of the whole serial loop. Images are resized to a
    fixed 672x672, which pins the visual-token count of a dynamic-resolution tower
    (Qwen3-VL). Decode steps still replay a CUDA graph — the visual tokens are
    already in the KV cache by then, so what is captured and replayed is a plain
    text step.
    """

    def __init__(self, model_dir: str, use_cuda_graph: bool, image_path: str, **gen_kwargs):
        from PIL import Image

        from lite_llama import VisionGenerator
        from lite_llama.models.config import read_model_type

        self._image = Image.open(image_path).convert("RGB").resize((672, 672), Image.BICUBIC)
        self._gen = VisionGenerator(
            checkpoints_dir=model_dir, use_cuda_graph=use_cuda_graph, **gen_kwargs
        )
        # llava wants an explicit <image> marker plus vicuna turns; Qwen3-VL's
        # preparer (like HF's chat template) inserts the visual placeholder itself,
        # so a plain question goes straight in.
        self._is_llava = read_model_type(model_dir) == "llava"
        self._texts: list[str] = []

    @property
    def runner(self):
        return self._gen.engine.model_runner

    def _wrap(self, prompt: str) -> str:
        return f"USER: <image>\n{prompt} ASSISTANT:" if self._is_llava else prompt

    def measure(self, prompts: list[str], max_gen_len: int, greedy: bool) -> BenchResult:
        params = sampling_params(max_gen_len, greedy)

        # Warm up autotune + graph capture so the measured run is steady state.
        for _ in range(2):
            list(self._gen.stream(self._wrap(prompts[0]), [self._image], sampling_params(8)))

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        req_ttfts: list[float] = []
        step_deltas: list[float] = []
        texts: list[str] = []
        for prompt in prompts:
            req_start = time.perf_counter()
            first = True
            prev = 0.0
            pieces: list[str] = []
            for delta in self._gen.stream(self._wrap(prompt), [self._image], params):
                now = time.perf_counter()
                if first:
                    req_ttfts.append(now - req_start)
                    first = False
                else:
                    step_deltas.append(now - prev)
                prev = now
                pieces.append(delta)
            texts.append("".join(pieces))
        torch.cuda.synchronize()
        total = time.perf_counter() - t_start

        self._texts = texts
        # Every request contributed its first token plus len(step deltas per
        # request) decode tokens; deltas were only collected after each first.
        return BenchResult(
            ttft_ms=(statistics.mean(req_ttfts) if req_ttfts else 0.0) * 1000,
            tpot_ms=(statistics.mean(step_deltas) * 1000) if step_deltas else 0.0,
            tpot_p50_ms=(statistics.median(step_deltas) * 1000) if step_deltas else 0.0,
            total_s=total,
            steps=len(req_ttfts) + len(step_deltas),
            batch=len(prompts),
            gen_tokens=len(req_ttfts) + len(step_deltas),
        )

    def texts(self) -> list[str]:
        return self._texts

    def close(self) -> None:
        del self._gen
        super().close()


def checkpoint_dtype(model_dir: str) -> torch.dtype:
    """The dtype the checkpoint's ``config.json`` declares (``torch_dtype``/``dtype``).

    Both engines load weights at the config's dtype, so the HF baseline must too:
    running HF in fp16 against a bf16 checkpoint measures "dtype change + engine
    change" together, not the engine difference. Falls back to fp16 when the field
    is absent (transformers' own historical default).
    """
    from transformers import AutoConfig

    declared = getattr(AutoConfig.from_pretrained(model_dir), "dtype", None)
    if isinstance(declared, str):  # older transformers returns a string
        declared = getattr(torch, declared, None)
    return declared if isinstance(declared, torch.dtype) else torch.float16


def dtype_tag(dtype: torch.dtype) -> str:
    """Short dtype name for row labels: ``torch.bfloat16`` -> ``bf16``."""
    return {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype, str(dtype))


class HFBackend(Backend):
    """HF transformers: ``generate`` has no per-step callback, so TTFT is a separate run.

    Weights load at the checkpoint's declared dtype (see :func:`checkpoint_dtype`),
    the same precision lite_llama uses. Under greedy, ``min_new_tokens ==
    max_gen_len`` forbids early EOS so the batch runs exactly ``max_gen_len`` steps
    (matching lite_llama's lockstep); under sampling, early EOS is allowed, ``steps``
    is the longest sequence's, and ``gen_tokens`` counts non-pad tokens.
    """

    def __init__(self, model_dir: str, attn: str = "sdpa"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dtype = checkpoint_dtype(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=self.dtype, attn_implementation=attn
            )
            .cuda()
            .eval()
        )
        self._last_gen: torch.Tensor | None = None

    def measure(self, prompts: list[str], max_gen_len: int, greedy: bool) -> BenchResult:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        gen_cfg = {"pad_token_id": self.tokenizer.pad_token_id}
        if greedy:
            gen_cfg["do_sample"] = False
        else:
            gen_cfg.update(do_sample=True, **SAMPLE_KW)

        # Warm up cudnn/autotune so the measured run is steady state.
        for _ in range(2):
            self.model.generate(
                **inputs,
                min_new_tokens=8,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # TTFT: one-token run, prefill + first sampled token.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.model.generate(**inputs, **gen_cfg, min_new_tokens=1, max_new_tokens=1)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        # Full run; greedy locks every sequence to exactly max_gen_len steps.
        full_cfg = dict(gen_cfg)
        if greedy:
            full_cfg["min_new_tokens"] = max_gen_len
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, **full_cfg, max_new_tokens=max_gen_len)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0

        prompt_len = inputs["input_ids"].shape[1]
        gen = out[:, prompt_len:]
        self._last_gen = gen
        steps = gen.shape[1]
        gen_tokens = int((gen != self.tokenizer.pad_token_id).sum())
        return BenchResult(
            ttft_ms=ttft * 1000,
            tpot_ms=(total - ttft) / (steps - 1) * 1000 if steps > 1 else 0.0,
            total_s=total,
            steps=steps,
            batch=len(prompts),
            gen_tokens=gen_tokens,
        )

    def texts(self) -> list[str]:
        if self._last_gen is None:
            return []
        return [self.tokenizer.decode(row, skip_special_tokens=True) for row in self._last_gen]

    def sample_text(self, limit: int = 120) -> str:
        """Decode the first row of the last run, for eyeball-checking output."""
        rows = self.texts()
        return rows[0][:limit] if rows else ""

    def close(self) -> None:
        del self.model
        super().close()


class VLLMBackend(Backend):
    """vllm offline ``LLM``: batch API with no per-step callback, so TTFT is a
    separate one-token run — the same pattern :class:`HFBackend` uses, which is
    what keeps the two external baselines' TTFT columns comparable.

    Weights load at the checkpoint's declared dtype (the precision lite_llama
    uses too). vllm's own CUDA-graph capture stays on (its default), matching
    lite_llama's graph rows. Under greedy, ``min_tokens == max_gen_len`` with
    ``ignore_eos`` locks every sequence to exactly ``max_gen_len`` steps,
    mirroring HF's ``min_new_tokens``; under sampling, early EOS is allowed and
    ``gen_tokens`` counts what each sequence actually produced.
    """

    def __init__(self, model_dir: str, max_model_len: int = 4096, gpu_util: float = 0.85):
        from vllm import LLM

        self.dtype = checkpoint_dtype(model_dir)
        self.llm = LLM(
            model=model_dir,
            dtype=self.dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_util,
        )
        self._last_texts: list[str] = []

    def measure(self, prompts: list[str], max_gen_len: int, greedy: bool) -> BenchResult:
        from vllm import SamplingParams

        base = {"temperature": 0.0} if greedy else {"temperature": 0.7, "top_p": 0.9}

        # Warm up capture/autotune so the measured run is steady state.
        for _ in range(2):
            self.llm.generate(
                prompts, SamplingParams(max_tokens=8, temperature=0.0), use_tqdm=False
            )

        # TTFT: one-token run — prefill plus the first sampled token.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.llm.generate(prompts, SamplingParams(max_tokens=1, **base), use_tqdm=False)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        # Full run. Greedy locks the batch to exactly max_gen_len steps, the
        # lockstep the lite_llama rows are measured under.
        full = SamplingParams(
            max_tokens=max_gen_len,
            **base,
            **({"min_tokens": max_gen_len, "ignore_eos": True} if greedy else {}),
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = self.llm.generate(prompts, full, use_tqdm=False)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0

        self._last_texts = [o.outputs[0].text for o in outputs]
        gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        steps = max(len(o.outputs[0].token_ids) for o in outputs)
        return BenchResult(
            ttft_ms=ttft * 1000,
            tpot_ms=(total - ttft) / (steps - 1) * 1000 if steps > 1 else 0.0,
            total_s=total,
            steps=steps,
            batch=len(prompts),
            gen_tokens=gen_tokens,
        )

    def sample_text(self, limit: int = 120) -> str:
        return self._last_texts[0][:limit] if self._last_texts else ""

    def close(self) -> None:
        del self.llm
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        super().close()


def make_backend(
    model_dir: str,
    *,
    use_cuda_graph: bool = True,
    tensor_parallel_size: int = 1,
    image_path: str | None = None,
    max_gpu_num_blocks: int | None = None,
    **engine_kwargs,
) -> Backend:
    """Pick the measurement strategy for a checkpoint and parallelism.

    ``tp > 1`` goes to :class:`EngineBackend` (only the continuous-batching path
    broadcasts each step's plan); a multimodal checkpoint goes to
    :class:`VisionBackend`; everything else to :class:`LiteBackend`.

    ``max_gpu_num_blocks`` is not forwarded to the multimodal path: visual tokens
    make each request's KV demand a function of image resolution, so reusing a text
    workload's pool size there only OOMs — let the engine profile it.
    """
    from lite_llama.models.config import read_model_type
    from lite_llama.models.registry import ModelRegistry

    if ModelRegistry.resolve(read_model_type(model_dir)).is_multimodal:
        if not image_path:
            raise ValueError(f"{model_dir} is a multimodal checkpoint and needs image_path")
        return VisionBackend(model_dir, use_cuda_graph, image_path, **engine_kwargs)
    if tensor_parallel_size > 1:
        return EngineBackend(
            model_dir,
            tensor_parallel_size=tensor_parallel_size,
            use_cuda_graph=use_cuda_graph,
            max_gpu_num_blocks=max_gpu_num_blocks,
            **engine_kwargs,
        )
    return LiteBackend(
        model_dir,
        use_cuda_graph=use_cuda_graph,
        max_gpu_num_blocks=max_gpu_num_blocks,
        **engine_kwargs,
    )


# --------------------------------------------------------------------------- #
# Runtime helpers shared by every bench script
# --------------------------------------------------------------------------- #
def free_gpu() -> None:
    """Release the CUDA caching allocator's view of a torn-down engine.

    Engine/generator/executor/KV manager hold mutual references, so without
    an explicit gc pass the memory is not returned: a second backend built in
    the same process then profiles a KV budget of zero tokens.
    """
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def reset_peak_mem() -> None:
    """Start a new peak-memory window (call before building the thing under test)."""
    torch.cuda.reset_peak_memory_stats()


def peak_mem_gb() -> float:
    """Peak allocated bytes since :func:`reset_peak_mem`, in GiB."""
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**3)


def describe_footprint(runner, replicas: int = 1) -> tuple[float, int]:
    """``(weight GiB, KV pool capacity in tokens)``, read from ``ModelRunner``'s tensors.

    ``replicas`` is the TP rank count: the runner holds only this rank's shard, so
    a whole replica's weights are ``replicas`` times that.
    """
    weight_bytes = sum(p.numel() * p.element_size() for p in runner.model.parameters())
    kv_tokens = runner.kv_cache_manager.gpu_kv_buffer[0].shape[0]
    return weight_bytes * replicas / (1024**3), kv_tokens


def _timed_runs(run, iters: int) -> tuple[float, list]:
    """Median wall time of ``iters`` ``run()`` calls, sync-bounded on both sides.

    Returns the median and every round's result, so callers can take the median of
    the per-round token counts.
    """
    latencies: list[float] = []
    results: list = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        results.append(run())
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    return statistics.median(latencies), results


def measure_generate(
    generate,
    prompts: list[str],
    *,
    gen_len: int,
    iters: int,
    tokenizer,
    warmup_prompts: list[str] | None = None,
) -> tuple[float, int, list[str]]:
    """Measure a one-shot ``generate`` path: warm up, time ``iters`` rounds, count tokens.

    Args:
        generate: ``(prompts, params) -> [output]``, each output having ``.text``.
        warmup_prompts: Prompts for the warm-up round; the measured ones by default.
            A script measuring cache hits must pass prompts from *outside* the
            workload, or the warm-up has already written the prefixes under test
            into the cache and every row reports a hit rate it did not earn.

    Returns:
        ``(median wall clock in seconds, median output tokens, last round's texts)``.
    """
    generate(warmup_prompts or prompts, sampling_params(8))
    median, outputs_per_iter = _timed_runs(
        lambda: generate(prompts, sampling_params(gen_len)), iters
    )
    texts_per_iter = [[out.text for out in outputs] for outputs in outputs_per_iter]
    counts = [count_gen_tokens(texts, tokenizer) for texts in texts_per_iter]
    return median, round(statistics.median(counts)), texts_per_iter[-1]


def count_gen_tokens(texts: list[str], tokenizer) -> int:
    """Re-tokenise generated text to count output tokens (vLLM's own method)."""
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)


def report_agreement(reference: list[str], rows: list[tuple[str, list[str]]]) -> None:
    """Every configuration must return the same completions; a low rate is a bug flag.

    Greedy sampling must be routing-independent: a shared prefix that hits the
    cache is *copied* K/V, not recomputed, so it can differ from a fresh prefill
    in the last bits — and an fp16 greedy tie can flip on that. The agreement
    rate is the flag that says the reuse is not merely inexact but wrong.
    """
    for label, texts in rows:
        if len(texts) != len(reference):
            continue
        same = sum(a == b for a, b in zip(reference, texts, strict=True))
        empty = sum(not text for text in texts)
        print(
            f"{label}: {same}/{len(reference)} completions identical to the baseline, {empty} empty"
        )


def require_gpus(min_count: int = 1) -> int:
    """Exit unless CUDA exposes ``min_count`` devices; returns the visible count."""
    visible = torch.cuda.device_count()
    if visible < min_count:
        print(
            f"requires {min_count} CUDA device(s), found {visible}",
            file=sys.stderr,
        )
        sys.exit(1)
    return visible


def timestamped_log_path(log_dir: str | Path, prefix: str) -> Path:
    """``<log_dir>/<prefix>_<stamp>.json`` — the --log-dir naming convention."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"{prefix}_{stamp}.json"


def write_json_log(path: str | Path, config: dict, results) -> None:
    """One JSON shape for every benchmark: {"config": ..., "results": ...}.

    A ``timestamp`` is stamped into the config unless the caller supplied one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {**config, "timestamp": datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps({"config": config, "results": results}, indent=2, default=str))
    print(f"-> {path}")


# --------------------------------------------------------------------------- #
# Data-parallel scaffolding
#
# The data-parallel benchmarks share a row shape, a table format, an argument set
# and a measurement path, so those live here instead of twice.
# --------------------------------------------------------------------------- #
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


def print_row_table(headers: list[str], widths: list[int], rows: list[list[str]]) -> None:
    """Aligned rows between two rules: first column left-aligned, the rest right.

    The caller formats each cell, so a column can hold a number, a ratio or ``—``
    without this function knowing which.
    """
    fmt = "".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
    rule = "─" * sum(widths)
    print(f"\n{rule}")
    print(fmt.format(*headers))
    print(rule)
    for row in rows:
        print(fmt.format(*row))
    print(rule)
