"""The measured systems: one ABC, one factory.

Every engine the benchmarks compare — rapid_llm's three entry points plus the
HF / vllm baselines — adapts into a :class:`Backend` here, so a scenario script
measures systems without knowing their constructors. ``make_backend`` picks the
strategy from the checkpoint and the parallelism.

Usage:
    from benchmarks.lib import make_backend, LiteBackend
"""

from __future__ import annotations

import statistics
import time
from abc import ABC, abstractmethod

import torch

from .metrics import BenchResult, run_requests, steps_to_result
from .utils import free_gpu
from .workloads import SAMPLE_KW, sampling_params


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

    def timeline_summary(self) -> str:
        """The engine's CUDA-event region table, the overlap benches' evidence.

        Empty for backends with no engine timeline (HF, vLLM).
        """
        return ""

    def close(self) -> None:
        """Return the GPU memory. Required before building a second backend in the
        same process, or its KV budget profiles as zero."""
        free_gpu()


class LiteBackend(Backend):
    """Single-process rapid_llm: a stream callback per step splits TTFT from TPOT.

    The batch advances in lockstep, so ``gen_tokens = steps * batch``. Constructor
    arguments pass straight through to ``TextGenerator`` (no defaults here, so this
    never overrides its own).
    """

    def __init__(self, model_dir: str, use_cuda_graph: bool, **gen_kwargs):
        from rapid_llm import TextGenerator

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
        from rapid_llm.engine import ContinuousBatchingEngine

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
        run = run_requests(self._engine, prompts, sampling_params(max_gen_len, greedy))
        self._texts = run.texts
        return run.result(len(prompts))

    def texts(self) -> list[str]:
        return self._texts

    def timeline_summary(self) -> str:
        return self._engine.timeline_summary()

    def close(self) -> None:
        self._engine.shutdown()
        del self._engine
        super().close()


class VisionBackend(Backend):
    """Multimodal measurement: ``VisionGenerator`` serves one request at a time.

    rapid_llm's multimodal path is serial (the processor takes one request), so
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

        from rapid_llm import VisionGenerator
        from rapid_llm.models.config import read_model_type

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
    the same precision rapid_llm uses. Under greedy, ``min_new_tokens ==
    max_gen_len`` forbids early EOS so the batch runs exactly ``max_gen_len`` steps
    (matching rapid_llm's lockstep); under sampling, early EOS is allowed, ``steps``
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

    Weights load at the checkpoint's declared dtype (the precision rapid_llm
    uses too). vllm's own CUDA-graph capture stays on (its default), matching
    rapid_llm's graph rows. Under greedy, ``min_tokens == max_gen_len`` with
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
        # lockstep the rapid_llm rows are measured under.
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
    continuous: bool = False,
    image_path: str | None = None,
    max_gpu_num_blocks: int | None = None,
    **engine_kwargs,
) -> Backend:
    """Pick the measurement strategy for a checkpoint and parallelism.

    ``tp > 1`` goes to :class:`EngineBackend` (only the continuous-batching path
    broadcasts each step's plan), and so does ``continuous=True``: the overlap
    benches need the continuous engine on one GPU as well, because the
    copy-stream overlap and its CUDA-event timeline live in the worker rather
    than in ``TextGenerator``. A multimodal checkpoint goes to
    :class:`VisionBackend`; everything else to :class:`LiteBackend`.

    ``max_gpu_num_blocks`` is not forwarded to the multimodal path: visual tokens
    make each request's KV demand a function of image resolution, so reusing a text
    workload's pool size there only OOMs — let the engine profile it.
    """
    from rapid_llm.models.config import read_model_type
    from rapid_llm.models.registry import ModelRegistry

    if ModelRegistry.resolve(read_model_type(model_dir)).is_multimodal:
        if not image_path:
            raise ValueError(f"{model_dir} is a multimodal checkpoint and needs image_path")
        return VisionBackend(model_dir, use_cuda_graph, image_path, **engine_kwargs)
    if tensor_parallel_size > 1 or continuous:
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
