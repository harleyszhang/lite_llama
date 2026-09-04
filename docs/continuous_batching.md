# 连续批处理（continuous batching）

## 为什么需要它

`LLMEngine.generate()` 在调用的那一刻就把 batch 定死了：所有序列同一步开始，整批一直保持满宽度推进，直到**最长**的那条结束。这带来两个无法在原路径里绕开的问题。

**一是算力浪费。** 一批 8 条请求里 7 条在第 20 个 token 就 EOS 了、1 条要跑到 500，那么后面 480 步里有 7 行在做纯粹的填充计算。序列一旦结束，它占的 batch 行并不会让出来。

**二是无法服务在线请求。** `generate()` 开始 1 毫秒之后到达的请求没有任何办法挤进去，只能等这一整批跑完。对一个服务端来说这等于串行。

连续批处理把"整批一次决定"换成"每步重新决定"：

```text
一次性批处理                          连续批处理
step  A B C D                        step  A B C D   等待队列
 1    ■ ■ ■ ■                         1    ■ ■ ■ ■   E F
 2    ■ ■ ■ ■                         2    ■ ■ ■ ■   E F
 3    ■ □ ■ ■   B 结束但仍占位          3    ■ E ■ ■   F     B 结束，E 立刻补位
 4    ■ □ ■ □                          4    ■ E F ■         C 结束，F 立刻补位
 5    ■ □ ■ □   ← 40% 是填充           5    ■ E F ■   ← 满载
```

## 分层结构

三个协作者各管一件事，边界按"host 决策 / device 状态"划分：

| 模块 | 职责 | 在哪一侧 |
| --- | --- | --- |
| [`Scheduler`](../lite_llama/engine/scheduler.py) | 谁 prefill、谁 decode、谁拿哪个槽位 | 纯 host，无张量 |
| [`SlotBatch`](../lite_llama/executor/slot_batch.py) | KV 布局与每步 attention 元数据 | 纯 device |
| [`ContinuousBatchingEngine`](../lite_llama/engine/continuous_engine.py) | 串起 step 循环、采样、停止判定 | 两侧 |

`Scheduler` 不持有任何张量，这是刻意的：调度策略因此可以在没有 GPU、没有权重的机器上完整单测（`tests/engine/test_scheduler.py` 25 个用例全部跑在 CPU 上）。

## 每一步做什么

```python
scheduled = scheduler.schedule()          # 1. host 侧决策
if scheduled.prefill:                     # 2a. 新请求进来，跑一格 padded prefill
    next_token = engine._prefill(scheduled.prefill)
else:                                     # 2b. 否则给所有在跑的请求走一步 decode
    next_token = engine._decode(scheduled.decode)
engine._harvest(batch, next_token)        # 3. 读回 token、detokenize、退休已结束的
```

**prefill 优先。** 队列里有请求时优先给它 prefill，这样 TTFT 不必等一整轮生成；代价是这一步 decode 停一拍。把 prefill 拼到 decode 步里（chunked prefill）需要混合阶段的 attention kernel，本框架暂时没有。

## KV 布局：固定槽位

一次性批处理路径可以用 bump 指针分配 KV 行，因为它独占整个 cache 且只追加。连续批处理不行——请求中途来去，每步都会落到 `KVCacheManager.alloc_contiguous_kvcache`，那里一次 `nonzero` 全池扫描加两次 `.item()`，等于**每个 decode 步 3 次设备同步**。

所以 `SlotBatch` 把分配器整个从 decode 路径上移走：槽位 `s` 永久拥有 `[s * max_seq_len, (s + 1) * max_seq_len)` 这段行，于是 `b_req_tokens_table` 就是恒等映射，建好一次再也不变。

- 申请一个请求的 cache = host 侧 pop 一个槽位号；
- 释放 = host 侧 push 回去；
- `update_kv_index` 这个 kernel 从 decode 路径上消失；
- 永不碎片化，且**不需要抢占**：槽位容量等于上下文窗口，准入时保证 `prompt + 生成上限 ≤ max_seq_len`，所以在跑的请求绝不可能中途 KV 不够。

代价写在明面上：一个槽位不管用不用都占满 `max_seq_len` 行，并发上限因此是 `gpu_num_blocks // max_seq_len`。分页分配器在显存密度上更优，一次性批处理路径（prompt 全部已知）继续用它。

## decode 步为什么没有主机-设备传输

`b_req_idx` 与 `b_seq_len` 只在**请求集合发生变化**时才从 host 重建。集合不变时：

```python
self._b_seq_len += 1                                    # 纯设备自增
cur_select_index = table[self._b_req_idx, self._b_seq_len - 1]   # 纯设备 gather
```

稳态下一个 decode 步的元数据只有两个 kernel，零传输。集合变化时才 `torch.tensor(...)` 上传一次——而且刻意每次**新建**张量而不是写回复用的 staging buffer：上一步的张量可能还排在流里等着被 kernel 读，从 host 覆写它会和未执行完的 kernel 竞争。

同样的"仅在集合变化时重建"策略也用在采样参数上： [`BatchedSamplingParams`](../lite_llama/engine/sampler.py) 把每个请求的 temperature / top-p / repetition_penalty 摊成 `[batch, 1]` 张量，一次采样覆盖整批混合配置，而不是按配置分组多次启动 kernel。

## CUDA Graph：把奇数 batch 补齐

decode graph 是按固定的 `(batch_size, seq_len_bucket)` 网格 capture 的，而连续批处理产生的 batch 大小是负载决定的——batch 7 直接掉回 eager，graph 的收益就没了。 `SlotBatch._pad` 把 batch 补到下一个已 capture 的尺寸，多出来的填充行指向一个 **保留槽位**，其 logits 被丢弃。

填充行的长度取整批的最大长度，这样每一行每步都恰好 +1，上面那条"集合不变就原地自增"的快路径才不会被填充行破坏。保留槽位的 KV 行在初始化时清零：填充行读它不影响任何真实行（没有 kernel 跨 batch 行做归约），但一池 NaN 会污染之后每一次 debug 的读数。

## 一个真实的 bug：按 batch 位置索引槽位

`flash_decoding` 原本这样取一行的 KV 历史：

```python
k_loc = tl.load(b_req_tokens_table + stride_req_to_tokens_b * batch_pid + offs_n_new, ...)
```

即**用 batch 内位置当请求号**。一次性批处理路径永远传 `b_req_idx = arange(batch)`，位置恰好等于槽位号，所以这个假设一直成立、从未暴露。

连续批处理一旦有请求中途结束，后面的请求在 batch 里前移一位，于是每一个都开始读 **邻居的 KV**。症状不是崩溃或乱码，而是：

```text
请求 3 期望： "...we need to identify numbers that are only divisible by 1 and
              themselves. Let's go through the process step by step."
请求 3 实际： "...we need to identify numbers that are only divisible by 1 and
              themselves. Let's go through using electromagnetic induction."
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                            这是请求 2 的结尾，逐 token 完全一致
```

修法是把 `b_req_idx` 传进 kernel，用 `cur_req_idx = tl.load(B_Req_Idx + batch_pid)` 转译一次再索引。回归测试见 `tests/kernels/test_flash_decoding.py::test_batch_row_reads_the_slot_named_by_b_req_idx` （把映射反转，逐位置索引必然读错）与 `tests/engine/test_continuous_batching.py::test_survivors_are_byte_identical_when_a_neighbour_leaves` （引擎级，退回旧 kernel 立即失败）。

## 实测数据

Qwen2.5-1.5B-Instruct，单卡 A10（23 GB），greedy，`max_gen_len=256`，16 个请求， `max_num_seqs=16`。原始日志：`docs/benchmark_logs/continuous_Qwen2.5-1.5B-Instruct_b16.json`。

| 场景 | 策略 | 墙钟 | 有效吞吐 | 平均延迟 |
| --- | --- | ---: | ---: | ---: |
| offline，整批同时提交 | 一次性 | 2.36 s | 1650 tok/s | 2355 ms |
| offline，整批同时提交 | 连续 | 2.34 s | **1661 tok/s** | **2221 ms** |
| offline，长短混合（4×256 + 12×32） | 一次性（只能给整批一个上限） | 2.34 s | 507 tok/s | 2337 ms |
| offline，长短混合 | 连续（逐请求上限） | 2.18 s | **550 tok/s** | **657 ms** |
| online，每 250 ms 到达一个 | 一次性（只能串行） | 41.6 s | 93 tok/s | 19145 ms |
| online，每 250 ms 到达一个 | 连续 | **6.04 s** | **644 tok/s** | **2309 ms** |

三条结论：

1. **在线到达是主战场：吞吐 ×6.9，平均延迟 ×8.3。** 这就是 online batch inference 与 continuous batching 叠加的收益。
2. **offline 且所有请求长度相近时，两者持平（×1.01）。** 请求同时结束，一次性批处理本来就没浪费，这时连续批处理不该有收益——有的话反而说明测量有问题。
3. **offline 长短混合时，墙钟只快 ×1.07，但平均延迟好 ×3.6，且省下 2688 个没人要的 token（一次性路径 69% 的产出是多余的）。** 墙钟提升有限是因为在 A10 上这个 batch 宽度下 decode 受权重带宽支配，batch 从 16 缩到 4 每步耗时几乎不变；收益体现在短请求早 3.6 倍拿到结果、以及槽位提前释放、承接新请求。

复现：

```bash
python benchmarks/bench_continuous.py --model-dir my_weight/Qwen2.5-1.5B-Instruct \
    --scenario all --batch 16 --max-num-seqs 16 --interval 0.25
```

## 精度如何保证

"改调度不改数值"这件事需要小心表述，因为**只有算术完全一致时"文本相同"才是合理预期**。 batch 宽度是 GEMM 的 M 维，batch 3 与 batch 4 的 fp16 累加顺序不同，top-2 logits 相差 ~1e-2 的 token 就可能翻转。这是批处理固有的，vLLM 同样如此。所以测试分两档：

- **逐字节比对**只用在两侧 shape 一致的场合：单请求对静态 batch-of-1；以及借 CUDA Graph 填充，让"4 条缩到 3 条"仍在 capture 宽度 4 上执行，从而算术完全一致。
- **其余场合**用 `assert_no_foreign_tail`，直接针对真正要防的失效模式：一个请求吐出**邻居的**文本。这种污染从来不是一个 token 的翻转，而是整句易主。

测试规模：

| 文件 | 数量 | 需要 |
| --- | ---: | --- |
| `tests/engine/test_scheduler.py` | 25 | CPU |
| `tests/entrypoints/test_api_server.py` | 23 | CPU（fake engine） |
| `tests/engine/test_async_engine.py` | 11 + 1 | CPU（stub），1 个需权重 |
| `tests/engine/test_continuous_batching.py` | 15 | GPU + 权重 |
| `tests/engine/test_continuous_perf.py` | 4 | GPU + 权重，`slow` |
| `tests/kernels/test_flash_decoding.py` | +2 | GPU |

## 当前边界

- **仅文本模型。** 视觉 prefill 需要逐请求的 processor 输出，padded prefill 网格放不下；多模态 checkpoint 在构造时就报 `NotImplementedError`。
- **无前缀复用。** 槽位是独占的，共享系统提示的请求各存一份 KV。
- **无抢占 / 无 chunked prefill。** 前者靠"槽位容量 = 上下文窗口"从设计上规避，后者需要混合阶段 attention kernel。
- **`n > 1` 采样未实现**，HTTP 层显式拒绝而不是静默返回一条。
- **每步一次同步。** 读回采样 token 用于 detokenize 与停止判定。这换来精确的停止语义（EOS 的下一步就离开 batch），也正因为如此才划得来：腾出的槽位立刻给排队请求。

## 相关文档

- [在线推理服务](./online_serving.md)：`lite-llama serve` 与 OpenAI 兼容端点。
