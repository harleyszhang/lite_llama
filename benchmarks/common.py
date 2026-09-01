"""Shared benchmark layer: one metrics vocabulary, one timing discipline.

``BenchResult`` plus ``steps_to_result`` define every script's numbers,
the :class:`Backend` ABC adapts engines into one interface, and the
helpers (``expand_prompts``, ``sampling_params``) keep workload shapes
reproducible across scripts.

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
# 口径:测什么、怎么采样
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

# 与 lite_llama 采样分支对齐的非 greedy 默认参数。
SAMPLE_KW = {"temperature": 0.7, "top_p": 0.8}

#: Greedy,且关掉重复惩罚与提前退出:基准的 token 数不能由"某些行触发、某些行
#: 不触发"的启发式决定,否则两列数字的分母都不同源。
GREEDY_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "stop_on_repeat": False,
}


def expand_prompts(prompts: list[str], batch: int) -> list[str]:
    """Cycle ``prompts`` up to ``batch`` entries."""
    return (prompts * ((batch // len(prompts)) + 1))[:batch]


def sampling_params(max_gen_len: int, greedy: bool = True):
    """基准口径的 ``SamplingParams``:greedy 走 :data:`GREEDY_PARAMS`,否则 :data:`SAMPLE_KW`。"""
    from lite_llama import SamplingParams

    return SamplingParams(max_gen_len=max_gen_len, **(GREEDY_PARAMS if greedy else SAMPLE_KW))


# --------------------------------------------------------------------------- #
# 指标值对象
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
    tpot_p50_ms: float = 0.0  # 仅流式逐步打点的后端能给出

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
    """把逐 step 的完成时刻折成一个 :class:`BenchResult`。

    TTFT 是第一步结束减去提交时刻,TPOT 是其后所有步间隔的均值——所有逐步推进的
    后端都是这套算法,写在一处才能保证它们的数字可比。

    Args:
        step_ends: 每步结束时的 ``perf_counter()``。
        t_start: 提交时刻(应在 ``torch.cuda.synchronize()`` 之后取)。
        total_s: 整轮墙钟秒数,由调用方在收尾 sync 之后算出。
        batch: 并发请求数。
        gen_tokens: 实际产出 token 数;省略时按 lockstep 推进算(每步 batch 个)。
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
# 测量策略
# --------------------------------------------------------------------------- #
class Backend(ABC):
    """一种被测系统的测量策略。

    子类只需回答两件事:同一组 prompt 怎么跑成一个 :class:`BenchResult`,以及怎么
    拆掉自己。``texts()`` 给精度对照用——同一次测量的产出,不必再跑一遍。
    """

    @abstractmethod
    def measure(self, prompts: list[str], max_gen_len: int, greedy: bool) -> BenchResult:
        """跑完整个工作负载并返回指标(实现方负责预热到稳态)。"""

    @property
    def runner(self):
        """本后端的 ``ModelRunner``(显存与 KV 容量从它读);外部对照后端没有。"""
        raise NotImplementedError(f"{type(self).__name__} 不持有 ModelRunner")

    def texts(self) -> list[str]:
        """上一次 :meth:`measure` 的完整输出;未产出文本的后端返回空列表。"""
        return []

    def close(self) -> None:
        """交还显存。同进程里建第二个后端之前必须调,否则它的 KV 预算profile 成 0。"""
        free_gpu()


class LiteBackend(Backend):
    """lite_llama 单卡策略:stream 每步回调,直接拆出 TTFT 与稳态 TPOT。

    batch 内所有序列 lockstep 推进,因此 gen_tokens = steps * batch。
    构造参数原样透传给 TextGenerator(不设默认,避免覆盖其自身的缺省语义)。
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
        """裸生成器:bench_e2e --verify 需要直接调 generate() 对拍 eager/graph 输出。"""
        return self._gen

    @property
    def runner(self):
        """ModelRunner:显存占用与 KV 池容量都从它读(见 :func:`describe_footprint`)。"""
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
    """连续批处理引擎策略:提交整批后逐 step 打点。

    TP>1 只有这条路能跑:executor 才会把每步的 plan 广播给 follower rank。
    此时 decode 走 eager——NCCL 集合通信不能进 graph 捕获。
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
        self._engine.generate(prompts, sampling_params(8))  # 预热 autotune 与分配器

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
            # 请求各自 EOS 离场,步数乘 batch 会高估:按每请求实收 token 累加。
            gen_tokens=sum(len(r.output_token_ids) for r in requests),
        )

    def texts(self) -> list[str]:
        return self._texts

    def close(self) -> None:
        self._engine.shutdown()
        del self._engine
        super().close()


class VisionBackend(Backend):
    """多模态测量策略:VisionGenerator 逐请求串行 stream 打点。

    lite_llama 的多模态路径逐请求服务(processor 单请求),因此
    ``--batch`` 语义变为串行请求数:TTFT 取每请求各自的首 token 时延均值,
    TPOT 取所有请求 decode 步间隔的均值,TPS 为整轮串行循环的聚合吞吐。
    图像统一 672x672 resize,钉死动态分辨率视觉塔(Qwen3-VL)的视觉 token 数。
    decode 步照常走 CUDA graph 重放——视觉 token 此刻已在 KV cache 里,
    捕获与重放的都是纯文本步。
    """

    def __init__(self, model_dir: str, use_cuda_graph: bool, image_path: str, **gen_kwargs):
        from PIL import Image

        from lite_llama import VisionGenerator
        from lite_llama.models.config import read_model_type

        self._image = Image.open(image_path).convert("RGB").resize((672, 672), Image.BICUBIC)
        self._gen = VisionGenerator(
            checkpoints_dir=model_dir, use_cuda_graph=use_cuda_graph, **gen_kwargs
        )
        # llava 要求显式 <image> 标记 + vicuna 轮次;Qwen3-VL 的 preparer
        # (与 HF 侧 chat template) 自己插入视觉占位符,普通问题直接进。
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


class HFBackend(Backend):
    """HF transformers 测量策略:generate 无逐步回调,用两段式拆 TTFT。

    greedy 时 min_new_tokens == max_gen_len,禁止提前 EOS 退出,
    保证 batch 恰好跑满 max_gen_len 步(与 lite_llama lockstep 对齐);
    采样时允许提前 EOS,steps 取 batch 内最长序列的步数,
    gen_tokens 按非 pad token 实数统计。
    """

    def __init__(self, model_dir: str, attn: str = "sdpa"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch.float16, attn_implementation=attn
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


def make_backend(
    model_dir: str,
    *,
    use_cuda_graph: bool = True,
    tensor_parallel_size: int = 1,
    image_path: str | None = None,
    max_gpu_num_blocks: int | None = None,
    **engine_kwargs,
) -> Backend:
    """按 checkpoint 与并行度挑测量策略,调用方不必自己分流。

    TP>1 交给 :class:`EngineBackend`(只有连续批处理路径会广播每步的 plan);
    多模态 checkpoint 交给 :class:`VisionBackend`;其余走 :class:`LiteBackend`。

    ``max_gpu_num_blocks`` 不转给多模态:视觉 token 让每请求的 KV 需求成为图像
    分辨率的函数,把文本档位的池子大小照搬过去只会 OOM,交给引擎自己 profile。
    """
    from lite_llama.models.config import read_model_type
    from lite_llama.models.registry import ModelRegistry

    if ModelRegistry.resolve(read_model_type(model_dir)).is_multimodal:
        if not image_path:
            raise ValueError(f"{model_dir} 是多模态 checkpoint,需要 image_path")
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
    """``(权重 GiB, KV 池容量 token)``,从 ``ModelRunner`` 的真实张量读。

    ``replicas`` 是 TP 的 rank 数:runner 只持有本 rank 的分片,整个副本的权重
    占用是它的 ``replicas`` 倍。
    """
    weight_bytes = sum(p.numel() * p.element_size() for p in runner.model.parameters())
    kv_tokens = runner.kv_cache_manager.gpu_kv_buffer[0].shape[0]
    return weight_bytes * replicas / (1024**3), kv_tokens


def timed_runs(run, iters: int) -> tuple[float, list]:
    """Median wall time of ``iters`` ``run()`` calls, sync-bounded on both sides.

    Returns the median and every round's result, so callers can aggregate
    token counts over all iters (median of per-iter counts) exactly as they
    did before this helper existed.
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
    """一次性 ``generate`` 路径的测量:预热、计时 ``iters`` 轮、数输出 token。

    Args:
        generate: ``(prompts, params) -> [output]``,每个 output 有 ``.text``。
        warmup_prompts: 预热用的 prompt;默认就用被测 prompts。凡是测缓存命中的
            脚本必须给一组**工作负载之外**的 prompt,否则预热已经把待测前缀写进
            缓存,每一行都会报出自己没挣到的命中率。

    Returns:
        ``(中位墙钟秒, 中位输出 token 数, 最后一轮的完整文本)``。
    """
    generate(warmup_prompts or prompts, sampling_params(8))
    median, outputs_per_iter = timed_runs(
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
