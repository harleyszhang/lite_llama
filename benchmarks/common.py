"""benchmarks 共享层:统一 prompts、指标口径、后端策略与结果渲染。

benchmarks/ 下的脚本测量同一组 prompts,指标口径对齐 vLLM/TensorRT-LLM:
    TTFT = prefill 提交到第一个 token 可见的墙钟时间
    TPOT = 稳态每步(每 token)延迟,取首 token 之后所有步间隔的均值
    TPS  = batch 聚合生成吞吐 = gen_tokens / 总时间

结构(Strategy + Factory):
    PROMPTS          测试 prompt 的唯一事实源,改 prompts 只动这里
    expand_prompts() 把 prompts 循环补齐到指定 batch
    BenchResult      指标值对象,负责把自身渲染成一行表格
    LiteBackend      lite_llama 策略:stream 逐步打点,TTFT/TPOT 直接可读
    HFBackend        HF transformers 策略:generate 无逐步回调,两段式拆 TTFT
    print_table()    按插入顺序打印对比表
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

import torch

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


def expand_prompts(prompts: list[str], batch: int) -> list[str]:
    """Cycle ``prompts`` up to ``batch`` entries."""
    return (prompts * ((batch // len(prompts)) + 1))[:batch]


@dataclass
class BenchResult:
    """One benchmark measurement. ``gen_tokens`` is the throughput denominator."""

    ttft_ms: float
    tpot_ms: float
    total_s: float
    steps: int
    batch: int
    gen_tokens: int
    tpot_p50_ms: float = 0.0  # 仅流式逐步打点的后端(LiteBackend)能给出

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


def print_table(results: dict[str, BenchResult]) -> None:
    for label, r in results.items():
        print(r.row(label))


class LiteBackend:
    """lite_llama 测量策略:stream 每步回调,直接拆出 TTFT 与稳态 TPOT。

    batch 内所有序列 lockstep 推进,因此 gen_tokens = steps * batch。
    采样口径:greedy(temperature=0)或 temperature=0.7 / top_p=0.8。
    构造参数原样透传给 TextGenerator(不设默认,避免覆盖其自身的缺省语义)。
    """

    def __init__(self, model_dir: str, use_cuda_graph: bool, **gen_kwargs):
        from lite_llama import TextGenerator

        self._gen = TextGenerator(
            checkpoints_dir=model_dir,
            use_cuda_graph=use_cuda_graph,
            **gen_kwargs,
        )

    @property
    def generator(self):
        """裸生成器:bench_cuda_graph 需要直接调 generate() 做中位延迟诊断。"""
        return self._gen

    def measure(self, prompts: list[str], max_gen_len: int, greedy: bool) -> BenchResult:
        from lite_llama import SamplingParams

        params = (
            SamplingParams(temperature=0.0, max_gen_len=max_gen_len)
            if greedy
            else SamplingParams(max_gen_len=max_gen_len, **SAMPLE_KW)
        )

        # Warm up autotune + allocator so the measured run is steady state.
        for _ in range(2):
            list(self._gen.stream(prompts, SamplingParams(temperature=0.0, max_gen_len=8)))

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        ttft = None
        step_ends: list[float] = []
        for _step_texts in self._gen.stream(prompts, params):
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t_start
            step_ends.append(now)
        torch.cuda.synchronize()
        total = time.perf_counter() - t_start

        steps = len(step_ends)
        deltas = [b - a for a, b in zip(step_ends, step_ends[1:])]
        return BenchResult(
            ttft_ms=(ttft or 0.0) * 1000,
            tpot_ms=(statistics.mean(deltas) * 1000) if deltas else 0.0,
            tpot_p50_ms=(statistics.median(deltas) * 1000) if deltas else 0.0,
            total_s=total,
            steps=steps,
            batch=len(prompts),
            # The batch advances in lockstep, so every step is batch tokens of work.
            gen_tokens=steps * len(prompts),
        )

    def close(self) -> None:
        del self._gen
        torch.cuda.empty_cache()


class HFBackend:
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
                **inputs, min_new_tokens=8, max_new_tokens=8,
                do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
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

    def sample_text(self, limit: int = 120) -> str:
        """Decode the first row of the last run, for eyeball-checking output."""
        if self._last_gen is None:
            return ""
        return self.tokenizer.decode(self._last_gen[0], skip_special_tokens=True)[:limit]
