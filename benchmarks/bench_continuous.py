"""连续批处理 vs 一次性批处理:吞吐、延迟与在线到达场景的对比基线。

两种调度策略在不同场景下的差距完全不同,所以分两个场景测,不合成单一数字:

场景 A(offline,同一批 prompt 一起提交)
    两条路径做同样的工作。一次性批处理锁步推进,某条序列 EOS 之后它的 batch 行
    仍然参与每一步计算,直到全批结束;连续批处理让它立刻离开,空出的槽位换成
    等待队列里的请求。差距来自输出长度的离散程度——所有请求一样长时两者持平。

场景 A2(offline-skew,同一批但每个请求要的长度不同)
    真实负载里 "写一句话" 和 "写一篇分析" 混在同一批。一次性批处理的
    ``generate()`` 只接受**一个** ``max_gen_len``,没法逐请求设上限,所以它只能按
    最长的那个跑,短请求多出来的 token 是用户没要的浪费。这里的 TPS 只统计
    "用户真正要的 token",并单独报出浪费量。

场景 B(online,请求陆续到达)
    一次性批处理的 ``generate()`` 在调用时就把 batch 定死了,晚到 1 毫秒的请求
    只能等下一次调用,即串行服务。连续批处理把它插进正在跑的 batch。这里量的是
    每请求延迟与排队时间,而不是峰值吞吐。

指标口径与 benchmarks/common.py 一致:
    TTFT = 请求提交到它第一个 token 可见的墙钟时间
    TPS  = 实际交付给用户的 token 数 / 总墙钟时间

用法:
    python benchmarks/bench_continuous.py --model-dir my_weight/Qwen2.5-1.5B-Instruct
    python benchmarks/bench_continuous.py --scenario online --batch 16 --json out.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field

import torch
from common import PROMPTS, expand_prompts

from lite_llama.engine.continuous_engine import ContinuousBatchingEngine
from lite_llama.engine.llm_engine import LLMEngine
from lite_llama.engine.sampler import SamplingParams

CKPT = "my_weight/Qwen2.5-1.5B-Instruct"


@dataclass
class Measurement:
    """一次测量结果;``gen_tokens`` 只算真正交付给用户的 token。"""

    label: str
    total_s: float
    gen_tokens: int
    requests: int
    ttfts_ms: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    # 用户没有要、纯粹因为整批共用一个上限而多算出来的 token
    wasted_tokens: int = 0

    @property
    def tps(self) -> float:
        return self.gen_tokens / self.total_s if self.total_s else 0.0

    def _percentile(self, values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]

    def row(self) -> str:
        ttft = statistics.mean(self.ttfts_ms) if self.ttfts_ms else 0.0
        latency = statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0
        return (
            f"{self.label:22s} {self.total_s:7.2f}s | TPS {self.tps:8.1f} tok/s | "
            f"TTFT mean {ttft:8.1f} ms p95 {self._percentile(self.ttfts_ms, 0.95):8.1f} ms | "
            f"latency mean {latency:8.1f} ms | {self.gen_tokens} tok"
        )

    def as_dict(self) -> dict:
        return {**asdict(self), "tps": self.tps}


def free() -> None:
    """两个引擎不能同时留在一张卡上:显式回收后下一个才能 profile 到 KV 预算。"""
    gc.collect()
    torch.cuda.empty_cache()


def count_tokens(tokenizer, texts: list[str]) -> int:
    return sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)


# --------------------------------------------------------------------------- #
# 场景 A:offline,整批一起提交
# --------------------------------------------------------------------------- #
def measure_static_offline(
    model_dir: str, prompts: list[str], params, kv_blocks: int, max_seq_len: int
):
    engine = LLMEngine(
        model_dir, max_seq_len=max_seq_len, max_gpu_num_blocks=kv_blocks, use_cuda_graph=True
    )
    engine.model_runner.enable_cuda_graph()
    token_ids = [engine.tokenizer.encode(p, add_special_tokens=True) for p in prompts]

    # 预热:让 Triton autotune 与 graph capture 不计入测量
    LLMEngine.generate_text(engine, token_ids, SamplingParams(temperature=0.0, max_gen_len=8))

    # 走流式接口:一趟既能拿到 TTFT,也能累积出完整文本,不必为了两个数跑两遍
    torch.cuda.synchronize()
    started = time.perf_counter()
    first_token_at = 0.0
    completions = [""] * len(prompts)
    for deltas in LLMEngine.generate(engine, token_ids, params):
        if not first_token_at and any(deltas):
            first_token_at = time.perf_counter()
        for index, delta in enumerate(deltas):
            completions[index] += delta
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    ttft = (first_token_at - started) * 1000 if first_token_at else 0.0
    result = Measurement(
        label="static (one-shot)",
        total_s=elapsed,
        gen_tokens=count_tokens(engine.tokenizer, completions),
        requests=len(prompts),
        # 锁步推进:整批同时开始,也同时被最长的那条拖住
        ttfts_ms=[ttft] * len(prompts),
        latencies_ms=[elapsed * 1000] * len(prompts),
    )
    del engine
    free()
    return result


def measure_continuous_offline(
    model_dir: str, prompts: list[str], params, kv_blocks: int, max_seq_len: int, max_num_seqs: int
):
    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=max_seq_len,
        max_num_seqs=max_num_seqs,
        max_gpu_num_blocks=kv_blocks,
        use_cuda_graph=True,
    )
    engine.generate(prompts[:2], SamplingParams(temperature=0.0, max_gen_len=8))  # 预热

    torch.cuda.synchronize()
    started = time.perf_counter()
    requests = [engine.add_request(prompt, params) for prompt in prompts]
    while engine.has_unfinished_requests():
        engine.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    result = Measurement(
        label="continuous",
        total_s=elapsed,
        gen_tokens=sum(len(r.output_token_ids) for r in requests),
        requests=len(prompts),
        ttfts_ms=[(r.first_token_time - started) * 1000 for r in requests if r.first_token_time],
        latencies_ms=[(r.finish_time - started) * 1000 for r in requests if r.finish_time],
    )
    del engine
    free()
    return result


# --------------------------------------------------------------------------- #
# 场景 A2:offline,但每个请求要的输出长度不同
# --------------------------------------------------------------------------- #
def skewed_caps(count: int, short: int, long: int, every: int = 4) -> list[int]:
    """每 ``every`` 个请求里一个要长输出,其余要短输出。"""
    return [long if index % every == 0 else short for index in range(count)]


def measure_static_skewed(
    model_dir: str, prompts: list[str], caps: list[int], kv_blocks: int, max_seq_len: int
):
    """一次性批处理只能给整批一个上限,于是按最长的跑,短请求的超额产出算浪费。"""
    engine = LLMEngine(
        model_dir, max_seq_len=max_seq_len, max_gpu_num_blocks=kv_blocks, use_cuda_graph=True
    )
    engine.model_runner.enable_cuda_graph()
    token_ids = [engine.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    LLMEngine.generate_text(engine, token_ids, SamplingParams(temperature=0.0, max_gen_len=8))

    params = SamplingParams(temperature=0.0, max_gen_len=max(caps), repetition_penalty=1.0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    completions = LLMEngine.generate_text(engine, token_ids, params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    produced = [
        len(engine.tokenizer.encode(text, add_special_tokens=False)) for text in completions
    ]
    useful = sum(min(count, cap) for count, cap in zip(produced, caps, strict=True))
    result = Measurement(
        label="static (one cap)",
        total_s=elapsed,
        gen_tokens=useful,
        requests=len(prompts),
        latencies_ms=[elapsed * 1000] * len(prompts),
    )
    result.wasted_tokens = sum(produced) - useful
    del engine
    free()
    return result


def measure_continuous_skewed(
    model_dir: str,
    prompts: list[str],
    caps: list[int],
    kv_blocks: int,
    max_seq_len: int,
    max_num_seqs: int,
):
    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=max_seq_len,
        max_num_seqs=max_num_seqs,
        max_gpu_num_blocks=kv_blocks,
        use_cuda_graph=True,
    )
    engine.generate(prompts[:2], SamplingParams(temperature=0.0, max_gen_len=8))  # 预热

    torch.cuda.synchronize()
    started = time.perf_counter()
    requests = [
        engine.add_request(
            prompt, SamplingParams(temperature=0.0, max_gen_len=cap, repetition_penalty=1.0)
        )
        for prompt, cap in zip(prompts, caps, strict=True)
    ]
    while engine.has_unfinished_requests():
        engine.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    result = Measurement(
        label="continuous (per-req)",
        total_s=elapsed,
        gen_tokens=sum(len(r.output_token_ids) for r in requests),
        requests=len(prompts),
        ttfts_ms=[(r.first_token_time - started) * 1000 for r in requests if r.first_token_time],
        latencies_ms=[(r.finish_time - started) * 1000 for r in requests if r.finish_time],
    )
    result.wasted_tokens = 0  # 每个请求都在自己的上限处停下
    del engine
    free()
    return result


# --------------------------------------------------------------------------- #
# 场景 B:online,请求按间隔陆续到达
# --------------------------------------------------------------------------- #
def measure_static_online(
    model_dir: str, prompts: list[str], params, kv_blocks: int, max_seq_len: int, interval: float
):
    """一次性批处理没有"插入正在跑的 batch"这个能力,只能一条条服务。"""
    engine = LLMEngine(
        model_dir, max_seq_len=max_seq_len, max_gpu_num_blocks=kv_blocks, use_cuda_graph=True
    )
    engine.model_runner.enable_cuda_graph()
    LLMEngine.generate_text(
        engine,
        [engine.tokenizer.encode(prompts[0])],
        SamplingParams(temperature=0.0, max_gen_len=8),
    )

    torch.cuda.synchronize()
    started = time.perf_counter()
    completions, latencies = [], []
    for index, prompt in enumerate(prompts):
        arrival = started + index * interval
        # 请求还没到就等它到;到了但引擎在忙,排队时间自然计入延迟
        now = time.perf_counter()
        if now < arrival:
            time.sleep(arrival - now)
        text = LLMEngine.generate_text(engine, [engine.tokenizer.encode(prompt)], params)[0]
        completions.append(text)
        latencies.append((time.perf_counter() - arrival) * 1000)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    result = Measurement(
        label="static (serial)",
        total_s=elapsed,
        gen_tokens=count_tokens(engine.tokenizer, completions),
        requests=len(prompts),
        latencies_ms=latencies,
    )
    del engine
    free()
    return result


def measure_continuous_online(
    model_dir: str,
    prompts: list[str],
    params,
    kv_blocks: int,
    max_seq_len: int,
    max_num_seqs: int,
    interval: float,
):
    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=max_seq_len,
        max_num_seqs=max_num_seqs,
        max_gpu_num_blocks=kv_blocks,
        use_cuda_graph=True,
    )
    engine.generate(prompts[:2], SamplingParams(temperature=0.0, max_gen_len=8))  # 预热

    torch.cuda.synchronize()
    started = time.perf_counter()
    arrivals = {}
    pending = list(enumerate(prompts))
    live = []

    while pending or engine.has_unfinished_requests():
        now = time.perf_counter()
        # 到点的请求立刻插进正在跑的 batch
        while pending and now - started >= pending[0][0] * interval:
            index, prompt = pending.pop(0)
            request = engine.add_request(prompt, params)
            arrivals[request.request_id] = started + index * interval
            live.append(request)
        if not engine.has_unfinished_requests():
            if pending:
                time.sleep(0.001)
                continue
            break
        engine.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    result = Measurement(
        label="continuous",
        total_s=elapsed,
        gen_tokens=sum(len(r.output_token_ids) for r in live),
        requests=len(prompts),
        ttfts_ms=[
            (r.first_token_time - arrivals[r.request_id]) * 1000 for r in live if r.first_token_time
        ],
        latencies_ms=[
            (r.finish_time - arrivals[r.request_id]) * 1000 for r in live if r.finish_time
        ],
    )
    del engine
    free()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=CKPT)
    ap.add_argument(
        "--scenario",
        choices=["offline", "offline-skew", "online", "both", "all"],
        default="all",
    )
    ap.add_argument("--batch", type=int, default=16, help="请求数")
    ap.add_argument("--max-gen-len", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--kv-blocks", type=int, default=40960)
    ap.add_argument("--interval", type=float, default=0.25, help="online 场景的到达间隔(秒)")
    ap.add_argument("--skew-short", type=int, default=32, help="offline-skew 里短请求的上限")
    ap.add_argument("--skew-long", type=int, default=256, help="offline-skew 里长请求的上限")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    prompts = expand_prompts(PROMPTS, args.batch)
    # greedy:两条路径的工作量口径一致,输出长度差异只来自模型自己何时 EOS
    params = SamplingParams(temperature=0.0, max_gen_len=args.max_gen_len, repetition_penalty=1.0)
    results: dict[str, Measurement] = {}

    if args.scenario in ("offline", "both", "all"):
        print(f"=== offline: {args.batch} requests submitted together ===")
        results["offline_static"] = measure_static_offline(
            args.model_dir, prompts, params, args.kv_blocks, args.max_seq_len
        )
        results["offline_continuous"] = measure_continuous_offline(
            args.model_dir, prompts, params, args.kv_blocks, args.max_seq_len, args.max_num_seqs
        )
        for key in ("offline_static", "offline_continuous"):
            print(results[key].row())
        speedup = results["offline_continuous"].tps / results["offline_static"].tps
        print(f"-> continuous batching throughput x{speedup:.2f}\n")

    if args.scenario in ("offline-skew", "all"):
        caps = skewed_caps(len(prompts), args.skew_short, args.skew_long)
        print(
            f"=== offline-skew: {caps.count(args.skew_long)} long ({args.skew_long}) + "
            f"{caps.count(args.skew_short)} short ({args.skew_short}) requests ==="
        )
        results["skew_static"] = measure_static_skewed(
            args.model_dir, prompts, caps, args.kv_blocks, args.max_seq_len
        )
        results["skew_continuous"] = measure_continuous_skewed(
            args.model_dir, prompts, caps, args.kv_blocks, args.max_seq_len, args.max_num_seqs
        )
        for key in ("skew_static", "skew_continuous"):
            print(results[key].row())
        print(
            f"-> static wasted {results['skew_static'].wasted_tokens} tokens nobody asked for; "
            f"useful throughput x{results['skew_continuous'].tps / results['skew_static'].tps:.2f}, "
            f"wall time x{results['skew_static'].total_s / results['skew_continuous'].total_s:.2f} faster\n"
        )

    if args.scenario in ("online", "both", "all"):
        print(f"=== online: {args.batch} requests, {args.interval * 1000:.0f} ms apart ===")
        results["online_static"] = measure_static_online(
            args.model_dir, prompts, params, args.kv_blocks, args.max_seq_len, args.interval
        )
        results["online_continuous"] = measure_continuous_online(
            args.model_dir,
            prompts,
            params,
            args.kv_blocks,
            args.max_seq_len,
            args.max_num_seqs,
            args.interval,
        )
        for key in ("online_static", "online_continuous"):
            print(results[key].row())
        static_latency = statistics.mean(results["online_static"].latencies_ms)
        cb_latency = statistics.mean(results["online_continuous"].latencies_ms)
        print(
            f"-> mean latency {static_latency:.0f} ms -> {cb_latency:.0f} ms "
            f"(x{static_latency / cb_latency:.2f} better)"
        )
        print(
            f"-> throughput x{results['online_continuous'].tps / results['online_static'].tps:.2f}\n"
        )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {"config": vars(args), "results": {k: v.as_dict() for k, v in results.items()}},
                handle,
                indent=2,
            )
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
