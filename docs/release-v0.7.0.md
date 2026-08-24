# Release v0.7.0 — 调度能力 (Chunked Prefill + Prefix Caching + Preemption)

**Date:** 2026-08-24
**Branch:** `prefix_caching`
**Theme:** Chunked prefill 封顶单步 prefill 工作量 + Prefix caching 复用共享前缀 KV + 抢占 (recompute)

## Summary

v0.7.0 升级调度器（参考 vLLM Scheduler v1），引入三项能力：

1. **Chunked prefill**：长 prompt 按 `max_chunk_size` 分片，封顶单 step 的 prefill 工作量，decode 尾延迟从「等一个完整 prompt」变成「等一个 chunk」。
2. **Prefix caching**：基于 block-hash 复用共享 prompt 前缀的 KV（对标 vLLM），共享 system prompt 的请求只 prefill 自己的 tail。
3. **抢占**（recompute 策略）：KV 压力超水位线时 evict 最新请求并重新排队。

## 1. Feature: Chunked Prefill

**v0.6 行为：** 一个 2000-token prompt 的 prefill 在**单个 step 内一次做完**。同 step 里的 decode 请求必须等这 2000 token 的 attention/GEMM 全部算完才能拿到自己的下一个 token，TPOT 出现 spike。

**v0.7 行为：** prefill 按 `max_chunk_size=512` 分片，单 step 承载的 prefill 工作量被封顶，decode 的等待时间由 chunk 大小决定，而不再由 prompt 长度决定。

### 可视化：同一份 scheduler 代码，只改 `max_chunk_size`

![chunked prefill](images/chunked_prefill.gif)

> GIF 由 `scripts/gen_chunked_prefill_gif.py` 驱动**真实 Scheduler** 录制，逐 step 打印调度决策。先播 `chunking OFF`，再播 `chunking ON`。

### 实测数据（真实 scheduler 输出）

| 配置 | Prefill 步数 | 单 step 峰值 prefill token | 每步 decode 请求数 |
|------|-------------|--------------------------|------------------|
| `max_chunk_size=0` (v0.6) | 1 | **2000** | 4 |
| `max_chunk_size=512` (v0.7) | 4 | **512** | 4 |
| `max_chunk_size=256` | 8 | **256** | 4 |

**关键结论：** decode 在两种模式下都在跑（prefill 与 decode 本就并存），真正的收益是
**单 step 最坏 prefill 工作量下降 3.9x**（2000 → 512 token）。decode 请求的尾延迟因此
从「等一个完整 prompt」变成「等一个 chunk」。

> 复现：`python scripts/gen_chunked_prefill_gif.py`
> 逐 step 原始输出与 benchmark 日志：[`docs/benchmark_logs/bench_chunked_prefill_v07.json`](benchmark_logs/bench_chunked_prefill_v07.json)

**使用方式：**

```python
from lite_llama.engine.scheduler import Scheduler, SchedulerConfig

config = SchedulerConfig(
    max_seq_len=4096,
    max_num_seqs=16,
    max_chunk_size=512,  # 0 = disable chunking (v0.6 behavior)
)
sched = Scheduler(config, num_slots=64)
```

## 2. Feature: Prefix Caching

**问题：** 生产负载里大量请求共享同一段 system prompt（agent 指令、few-shot 示例、
RAG 文档前缀）。每个请求都从头 prefill 这段共享前缀，是纯粹的重复计算。

**方案（对标 vLLM 的 block-hash prefix caching）：** prompt 按 16-token 一个 block
切分，每个 block 的 hash **链式**折入前一个 block 的 hash，因此一个 hash 唯一标识
一个**前缀**而非单个 block 的内容。新请求的 leading blocks 若已在缓存中，这些 token
的 KV 直接复用，prefill 从未命中处开始。

### 可视化：共享 system prompt 只 prefill 一次

![prefix caching](images/prefix_cache.gif)

> GIF 由 `scripts/gen_prefix_cache_gif.py` 驱动**真实 Scheduler**（`enable_prefix_cache=True`）
> 录制，每个数字都是真实 admission 决策的 `Request.num_cached_tokens` 与 `prefix_cache_hit_rate`。

### 实测数据（真实 scheduler 输出）

场景：4 个请求共享 768-token system prompt，各带 32-token 独立 user tail。

| 请求 | Cached (KV 复用) | 实际 prefill | 累计命中率 |
|------|----------------|-------------|-----------|
| req-0 (cold) | 0 | **800** | 0.0% |
| req-1 (shared) | 768 | **32** | 48.0% |
| req-2 (shared) | 768 | **32** | 64.0% |
| req-3 (shared) | 768 | **32** | 72.0% |

**关键结论：** 第一个请求 cold，prefill 全部 800 token 并填充缓存；之后每个共享前缀的
请求跳过 768 token，实际 prefill 工作量从 800 → 32 token，**降低 25x**。命中率随共享
请求增多持续爬升到 72%。

> 复现：`python scripts/gen_prefix_cache_gif.py`
> benchmark 日志：[`docs/benchmark_logs/bench_prefix_cache_v07.json`](benchmark_logs/bench_prefix_cache_v07.json)

**使用方式：**

```python
from lite_llama.engine.scheduler import Scheduler, SchedulerConfig

config = SchedulerConfig(
    max_seq_len=4096,
    max_num_seqs=16,
    enable_prefix_cache=True,  # 开启 block-hash 前缀复用
)
sched = Scheduler(config, num_slots=64)
# ...
print(f"prefix cache hit rate: {sched.prefix_cache_hit_rate:.1%}")
```

实现见 [`lite_llama/engine/prefix_cache.py`](../lite_llama/engine/prefix_cache.py)（`PrefixCache`
类：`query` / `register` / `release` + 引用计数驱逐）。

## 3. Feature: Preemption (Recompute Strategy)

当无空闲 slot 时，调度器 evict 最近接入的 running 请求：
- 释放其 KV slot
- 清空已生成 tokens
- 重新排到 waiting 队列头部（下次优先重新 prefill）

```python
# Automatic: scheduler preempts when no slots available
sched.add_request(new_request)  # triggers preemption if full
print(f"Total preemptions: {sched.num_preemptions}")
```

对标 vLLM 的 `PreemptionMode.RECOMPUTE`。

## 4. Feature: SchedulerOutput 增强

`SchedulerOutput` 新增字段支持 chunked prefill：
- `prefill_chunk_lens: list[int]` — 每个 prefill 请求本步处理的 token 数
- `preempted: list[Request]` — 本步被抢占的请求（用于日志/监控）
- prefill + decode 可同时非空（v0.6 互斥）

调度器新增 `advance_chunks()` 方法用于推进 chunk 进度。

## 5. SchedulerConfig 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_chunk_size` | 512 | 每步最大 prefill token 数。0=不分片 |
| `enable_prefix_cache` | False | 开启 block-hash 前缀复用（v0.7 已实现）|

## 6. 测试结果

```
12 passed   (test_prefix_cache.py, block-hash cache + scheduler integration)
25 passed   (test_scheduler.py, chunked prefill + preemption)
394 passed  (full CPU suite)
```

调度器测试更新：`test_prefill_takes_priority_over_decode` 验证 prefill 与 decode
并行运行（不再互斥）。

## 7. 设计参考 (vLLM)

| lite_llama v0.7 | vLLM 对应 | 说明 |
|-----------------|-----------|------|
| `SchedulerConfig.max_chunk_size` | `SchedulerConfig.max_num_batched_tokens` | 控制 prefill 粒度 |
| `PrefixCache` (block-hash 链式) | `BlockHashType` + `KVCacheManager` prefix cache | 共享前缀 KV 复用 |
| `Scheduler._preempt()` | `Scheduler._preempt()` | recompute 策略 |
| `SchedulerOutput.prefill_chunk_lens` | `SchedulerOutput.num_prefill_groups` | 分片元数据 |
| `advance_chunks()` | 内置在 `_schedule_running()` | 推进 chunk 状态 |

## Upgrade

```bash
git checkout prefix_caching && uv pip install -e .

# 启用 chunked prefill (默认 512 tokens/chunk)
lite-llama serve --port 8000  # scheduler 自动使用 chunked prefill

# 启用 prefix caching + 自定义 chunk 大小
# SchedulerConfig(max_chunk_size=256, enable_prefix_cache=True)
```
