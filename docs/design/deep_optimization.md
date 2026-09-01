# 深度优化设计方案（参照 SGLang）

v0.10 之后以性能为唯一目标的专题设计。条目编号 O1–O14，落地时映射回
[ROADMAP.md](../../ROADMAP.md) 的版本条目（F/A/P 系列、地基 0–3、L1–L5）——
ROADMAP 管"什么时候做什么"，本文管"每件事具体怎么做、为什么值"。

基准环境：2×A10（24 GB，PCIe 互联、无 NVLink）+ Qwen3-30B-A3B-Instruct-2507-FP8。
文中所有收益数字均为该口径下的机制推导估算，最终以 P0 立零点的 on/off 实测为准。

## 1 现状基准：四个结构性瓶颈

**KV 是「槽位」不是「页」，prefix 复用靠拷贝。**
`executor/slot_batch.py`：slot `s` 永久拥有行 `[s*max_seq_len, (s+1)*max_seq_len)`，
并发上限 = 槽位数；`executor/kv_cache_manager.py` 自述 `One row per token
(block_size=1)`，并有 `TODO: reshape into [blocks, block_size, ...] to support
PagedAttention`。于是 `engine/prefix_cache.py` 的命中结果是一串
`(src_slot, start, len)` 拷贝段——每 token 的 KV（fp16、TP=2、48 层）约 96 KB/rank，
一个 4K 共享前缀的命中要搬约 200 MB D2D。v0.6 的「分页 KV」实际落地的是行级
引用计数，真分页（block_size > 1 + block_table）没有落。

**一步最多三次 forward，metadata 三套。**
`engine/continuous_engine.py` 把每步拆成 `PREFILL`（padded 网格、不读 cache）、
`EXTEND`（逐 token 行、读 cache）、`DECODE` 三个 pass，各自一套注意力元数据与
kernel。根因是 prefill kernel 不读 cache，续传 chunk 只能走 EXTEND 兜底。

**CPU-GPU 串行 + TP 下无 CUDA graph。**
`engine/async_engine.py` worker 线程逐条 `step()`：调度 → 发射 kernel →
同步等采样结果回读 → 才调下一步。TP > 1 时 CUDA graph 被显式禁用，
48 层 × 2 个 all-reduce 走 NCCL ring over PCIe（batch 小 payload 小，延迟主导），
加上数百次 kernel launch 的 Python 派发——batch=1 的 TPOT 里约 30–40% 是非计算开销。

**A10 上 MoE 是带宽瓶颈，dequant 路径物化中间张量。**
sm86 无 fp8 算力，fp8 权重解包成 fp16 再进 GEMM（ROADMAP P3 指出的问题），
每步多一轮全量权重读写。

## 2 方案全景与依赖

```text
引擎循环层（不碰 kernel，先拿钱）
  O2 zero-overhead 循环        O10 tokenize 移出关键路径
调度与内存层
  O1 真分页 + Radix 零拷贝     O6.1 in-batch prefix   O6.2 两级 token budget
  O9 准入滞回 + decode 窗口
通信层
  O3.1 one-shot all-reduce     O3.2 TBO 双批重叠      O11 通信-RMSNorm 融合
kernel 层
  O4 MoE dequant 融合          O7 prefill 桶化 graph   O8 split-kv 自适应
解码算法层
  O5 ngram 投机解码            O6.3 greedy argmax
工程层
  O12 prefill/decode 双 stream 重叠   O13 graph 惰性捕获   O14 fp8 KV 端到端强化
```

依赖关系：O2 / O3.1 / O6.3 / O9 / O10 / O13 无前置，可立即做；O1 是
O5 / O6.2 / O7 的前置；O3.1 解锁 TP + CUDA graph（ROADMAP P8）；O8 / O14
可独立做，收益在 O1 之后放大。

## 3 引擎循环层

### O2 zero-overhead 引擎循环

**一句话**：decode 的输入 token 不再回 CPU，直接把 device 上的采样结果喂给
下一步的 embedding；step 拆成 launch / harvest 两半，隔一步读结果。

**现状慢在哪。** `ContinuousBatchingEngine.step()` 每步严格串行：
调度 → 发射 kernel → 等 GPU 算完 → 采样 token 读回 CPU → detokenize/判停 →
下一轮。CPU 在等 GPU 时是死的，GPU 在等 CPU 时也是死的。batch=1 decode
的时间线（数字为估算）：

```text
现状，一步 ≈ 7.5ms：
CPU:  [调度+构建 1.5ms][ 同步读回+detok 1ms ][调度...]
GPU:                     [======decode 计算 5ms======]
                                          ↑ CPU 在干等
```

**为什么现在没法提前调度下一步——卡在 token 上。** decode 的输入是上一步
采样出来的 token，`_decode_work` 用 `request.output_token_ids[-1]`（host 上的
Python int）拼输入，再上传回 GPU，一去一回强迫每步同步。sglang 的 overlap
循环（`managers/scheduler.py` 的 `event_loop_overlap`）能转起来，靠的是想通了
一件事：下一步 forward 需要的只是 token id 的**数值**，而这个数值此刻就在
显存里——直接把 device 上的采样结果 tensor 喂给下一步 embedding 查表。
真正需要 token 数值的只有 detokenize 和停止判断，晚一步做毫无损失：
读回来时 GPU 已经在算下一步了。

**改法（三处）。**

1. `ModelInput.tokens` 允许传 device Tensor：decode pass 直接引用上一步采样
   输出 buffer；prefill 的 prompt 本来就在 CPU，不动。
2. step 循环重排：

```python
result_q = deque()
while True:
    plan = scheduler.schedule()            # 纯 CPU；GPU 正算着上一步
    result_q.append(executor.launch(plan)) # 只发射，不同步
    if len(result_q) > 1:                  # 隔一步才读
        last = result_q.popleft()
        tokens = last.sampled().cpu()      # D2H 已被本步 GPU 计算藏住
        advance_requests(tokens)           # detok、停止判断、penalty 计数
```

3. 采样结果落 pinned buffer + event——`executor/overlap.py` 的 `StreamPool`
   就是干这个的，复用。

CUDA graph 不受影响：replay 的输入本来就是静态 buffer，把 device 采样结果
D2D 拷进输入 buffer，比现在「CPU 读回再上传」少绕一圈。

**三个诚实的代价**：停止判定滞后一步（请求 eos 后多占一个槽位一步，sglang
同样如此）；repetition penalty 计数滞后一步（惩罚窗口几十 token，影响可忽略，
写进文档）；`max_gen_len` 在 admit 时预扣 1，避免多吐一个 token。

**预期收益**：batch=1 TPOT -20~25%（~7.5ms → ~6ms）；并发下收益更大——CPU
组批时间随 batch 涨、GPU 时间不涨。这条也把 ROADMAP P9 的第一落点从
「独立进程 + ZMQ」降级为同进程重叠，成本 10% 拿到 90% 收益，且不再与
F2（单进程 pdb 直达）冲突。

### O10 tokenize 移出关键路径

CLI 与 serve 入口的 tokenize 在主线程串行，几十 K token 的大 prompt 一次几十
毫秒直接叠在 TTFT 上。O2 的 launch/harvest 结构天然给了挂点：tokenize 丢进
harvest 段的线程池，下一个 launch 只用「已就绪」的请求。预期：大 prompt
TTFT 减掉几十毫秒；优先级最低，顺手做。

## 4 内存与调度层

### O1 真·分页 KV + Radix 零拷贝共享（最大的一项）

**参照**：sglang `mem_cache/radix_cache.py`（chained-hash 树 + 引用计数 +
只从叶驱逐的 LRU）、`mem_cache/allocator/`（页分配器）、`mem_cache/memory_pool.py`
（buffer 直接塑成 `[pages, page_size, kv_heads, head_dim]`）、
`layers/attention/triton_backend.py`（decode kernel 直接吃 block_table）。

**设计四步。**

1. **buffer 重塑**：`gpu_kv_buffer` 从 `[max_tokens, 2*kv_heads, head_dim]` 变
   `[num_pages, page_size, 2*kv_heads, head_dim]`，page_size=16（与现有
   `PREFIX_CACHE_BLOCK_SIZE` 对齐，flashinfer 默认页也是 16）；
   `b_req_tokens_table` 从恒等映射变成真 block_table `[max_reqs, max_pages]`。
2. **PrefixCache 升级为 RadixCache**：树节点持有 page ids 而非 owner_slot。
   命中 = 页引用 +1，零拷贝；`prefix_copies` / `invalidate_slot` /
   `assign_owner` / `_pending_owners` 整套「先拷贝后认领」簿记退场——它们
   存在的唯一理由就是拷贝有时序（`_promote_pending_owners` 的 docstring
   写得很清楚）。
3. **kernel 侧改造（全部工作量的所在）**：`flash_decoding`、
   `flashattention2_nopad`、`update_kv_buffer` 从行寻址改页寻址——sglang
   triton_backend 的 decode kernel 就是带 page table 的，形态可直接参照。
   flashinfer 适配器反而变简单：`BatchPrefillWithPagedKVCacheWrapper`
   原生吃页表。
4. **写后读统一（顺带消灭三 forward 分叉）**：sglang 的 extend 语义是
   「先把本 chunk 的 KV 写进 cache，再 attend 完整 `[0, pos+1)`」。统一后
   PREFILL/EXTEND/DECODE 合成一个 varlen 契约：每行带
   `(page_table, q_len, kv_len)`，首 chunk、续传 chunk、decode 都是同一
   kernel 的不同 q_len。`PassKind` 三分支、`_prefill_work` 路由、padded
   网格全部退役；纯 decode 步继续走 CUDA graph（sglang 同款组合）。

**实例理解**：10 个请求同时到达，都带同一个 2K system prompt。现状：
第一个请求的块只进 `_pending_owners`，`assign_owner` 下一步才认领，于是
第 2~10 个请求同一步 admit 时 `copyable_tokens=0`，**各自完整 prefill 一遍
2K 前缀**（一次昂贵计算付 10 次）；即便命中认领后的 owner，还要按段拷贝
~200 MB/请求。页化后：命中 = 页引用 +1，第 2~10 个请求直接共享页，TTFT
从 `10 × prefill(2K)` 变 `1 × prefill(2K)`。

**迁移策略**：走 ROADMAP A2（KV 布局可插拔），`PagedLayout` 与现有
`SlotLayout` 并存注册、golden 矩阵双跑，随时回退。

**预期收益**：32K 上下文 + 平均 2K 对话场景，并发容量 ×5–16；多轮对话
TTFT 从「重算 2–4ms + 拷贝 ~1ms」到近零；单步 forward 3 次 → 1 次。

### O6.1 in-batch prefix（同批请求互认）

不依赖 O1 的先行版：admit 时把「同一步内、块哈希相同」的等待请求分组，
组内第一个承担计算，其余延一步 admit，下一步命中已认领的 owner，
省掉重复 prefill 的算力（拷贝仍在，O1 落地后归零）。
预期：共享前缀 burst 场景 prefill 算力 -60~90%。

### O6.2 两级 token budget（必须随 O1 同版本落地）

现状 slot 模型下 KV 容量被 `num_slots` 隐式管住，粗但安全。真分页之后容量
变成动态共享池，只看 `max_num_batched_tokens`（prefill 预算）不够：

场景走查：32K 上下文模型，KV 池剩 60K 页-token，来了一个 30K prompt，
同时 32 个 decode 各占 1K 还在涨。只看 prefill 预算会 admit 这个 30K——
它自己分块没问题，但 32 个 decode 每步各长 1 token，几步之内池子见底，
触发抢占风暴：刚算完 30K 的请求被踢回 waiting 重算。解法对齐 sglang
`ScheduleBatch` 的语义：admit 前把 `rem_total_tokens`（KV 容量账）与
`rem_input_tokens`（prefill 预算账）分开记，「本请求全程要占多少、现有
running 会涨多少」模拟完不够就不放行。页模型的灵活性没有这道账，
就是抢占风暴的燃料。

### O6.3 greedy 直接 argmax

`sampler.py` 现在所有请求统一走 `sample_top_p`：topk(候选池) → softmax →
掩码 → 重归一 → multinomial，五六个小 kernel。greedy 请求一个 argmax 就够。
对教学框架来说「greedy 就是 argmax」也比「greedy 也走 top_p 流水线」诚实。
采样固定开销 -80%，半小时的活。

### O9 准入滞回 + decode 窗口

**滞回**：`kv_cache_manager.can_admit` 的 watermark 是单边阈值，水位在阈值
附近抖动时 admit/preempt 来回横跳，每次 preempt 都付一次重算租金。sglang
的 `min_free_slots_delayer` 是对应细节：跌破水位后不立刻恢复准入，等回升到
阈值以上一段才放行。搬过来就是 `can_admit` 加恢复带。

**decode 窗口**：burst 到来时立刻打断 decode 去 prefill，所有在跑请求的
TPOT 尖刺。policy：新 prefill 最多攒 N 步再插队，用一点 TTFT 换 decode
平滑。做成 `SchedulerConfig` 开关，benchmark 出 TTFT-TPOT 权衡曲线。
预期：水位抖动期吞吐掉坑消除；decode TPOT P99 -30~50%。

## 5 通信层

### O3.1 one-shot all-reduce（解锁 TP graph）

TP 组内 one-shot all-reduce：rank 间直接 P2P 写对端 IPC buffer + flag 自举，
payload ≤ 阈值时替换 NCCL ring。参照 sglang
`distributed/device_communicators/custom_all_reduce.py`（与 `torch_symm_mem.py`
的 multimem 路径）。

实例：现在每步 48 层 × 2 次 ring all-reduce，PCIe 小消息每次 15–25μs，
纯延迟主导；one-shot P2P 写 5–8μs，一步省 ~1–1.5ms。它 graph 安全，
顺带解锁 ROADMAP P8（TP + CUDA graph 锁步 capture）——ROADMAP 自己标注的
「真实的设计空缺」。

**预期收益**：TPOT -15~20%；与 O2、TP graph 三项收益有重叠，合计后
CPU + 通信侧开销逼近零，TPOT 逼近 GPU 下限 ~4ms（合计 -40% 左右）。

### O3.2 TBO 双批重叠

decode batch 切两个 micro-batch，micro-A 做 all-reduce 时 micro-B 在算
GEMM。参照 sglang `batch_overlap/`（原 two_batch_overlap）。lite_llama 已有
`PassKind` / `ModelInput` 抽象，双 pass metadata 现成。做成 batch 超阈值才
启用的 policy。PCIe 互联通信占比高时收益真实，NVLink 上气泡更小。

### O11 通信-RMSNorm 融合

TP 下 o_proj/mlp 后的 all-reduce + RMSNorm 改成 reduce-scatter → 每 rank
只对持有的 token 段做 norm → all-gather；norm 并行化，且 reduce-scatter
完成的 token 段立刻进 norm 而不用等全量到齐——通信尾部与 norm 重叠。
flashinfer 的 fused allreduce+rmsnorm 是同类思路，lite_llama 已有
`skip_rmsnorm` kernel，融合点现成。

**实例理解**：batch=1 时 norm 读写是 8KB 级、无关痛痒——这条别指望
batch=1。batch=32 时 norm 读写省一半，通信尾部可藏。预期：batch≥16 每步
-0.3–0.5ms，batch=1 <5%，诚实标注。它是 O3.1 的增值包，不是独立项。

## 6 kernel 层

### O4 MoE dequant 融合 grouped GEMM

fp8 权重解包进 GEMM mainloop（per-group scale 在 Triton 里就是加载时
`* scale`），消除中间 fp16 权重物化。A10 上 MoE decode 是纯带宽活，
省掉的是那次额外的全量权重读。配套：router 提前发射（O2 的 launch/harvest
分离给了挂点）、`moe_align_block_size` 中间 buffer 常驻、`_launch_config`
由 autotune 冻结记录接管（机制 v0.10 已接好，缺一轮 collect）。
预期：MoE kernel 时间 -20~25%，TPOT 直接受益。

### O7 prefill 桶化 CUDA graph

O2 只解决 CPU 不等 GPU，没解决 launch 本身。decode 已有 graph
（batch 桶 + filler slot 补齐），prefill 是裸奔的——而 prefill 恰是 kernel
数最多的路径。机会：chunked prefill 的 chunk 大小本来就桶化
（`max_chunk_size=512`），PREFILL pass 的形状只有 `(batch, 512)` 有限几种，
正是 capture 的好对象。sglang 最新代码在推 FullCG（连 prefill 也进 graph），
证明是主流方向。

设计：PREFILL pass 按 `(batch桶, chunk桶)` 捕获，replay 时 D2D 拷入
token/position。EXTEND pass 先不动——O1 的写后读统一会把 EXTEND 整条路径
消灭，别在它上面花 graph 功夫，O7 排在 O1 之后最省。
预期：TTFT 里 CPU 派发占比归零，TTFT -5~10%（大 prompt 场景）。

### O8 decode attention 的 split-kv 自适应

batch=1（单请求聊天）的 decode 有结构性短板：不 split 时一个 seq 的注意力
压在少数 SM 上，A10 的 80 个 SM 大部分在围观；长上下文（>4K）时这是
TPOT 大头——8K 上下文的 KV 读 ~750MB/步，与 MoE 的 ~1.8GB 同量级。
解法：split-kv 把 KV 维切开并行算 partial softmax 再合并，**split 数按
`(batch, seq_len)` 查表**——batch 大时每行自有并行度，split=1 最好（省
合并）；batch=1 且 seq 长才开大 split。查表的生成直接挂 autotune 冻结
记录（collect → search → persist 三步现成），给 autotune 找到第二个高价值
客户。预期：长上下文 TPOT -20~30%，与 O14 乘法叠加。

## 7 解码算法层

### O5 ngram 投机解码先行

候选来自 prompt + 已生成文本的 n-gram 查表（纯 Python 先行，热了再下沉）；
verify = 一次 varlen pass——draft k 个 token 作为 q_len=k 的行 attending
既有 cache，树形掩码用 varlen kernel 的 mask 位段表达。这正是 O1 统一
varlen 契约的直接受益者：没有 O1，verify 要在 EXTEND 三分叉上再叠树形
逻辑。自适应 draft 长度（滑动接受率）对齐 ROADMAP P4 的自有设计；
MTP 等 draft 权重类策略仍按 ROADMAP 放 v0.14。sglang 的 ngram
（`speculative/`，本机源码已有 `cpp_ngram`）证明这条路在代码/改写负载上
性价比极高。
预期：mean accept 1.5–2.5 → TPS ×1.3–2；验收：重复前缀负载 mean accept ≥2。

## 8 工程层

### O12 prefill/decode 双 stream 重叠

prefill（compute-bound）与 decode（memory-bound）无数据依赖
（decode 集合本就不含本步刚 prefill 完的请求，`just_prefilled` 已排除），
可两条 stream 同时发射。风险是 SM 竞争拖慢 prefill——做成 policy：
仅当本步 prefill chunk 总 token < 阈值时并行。

实例：现状一个 512-token chunk 插进来，正在 decode 的 32 个请求这步多等
~1–2ms，TPOT 尖刺；双 stream 后只被 SM 竞争拖慢 ~20–30%，尖刺
+1.5ms → +0.4ms，代价是 prefill TTFT 变差 5–15%。它是 O9 decode 窗口的
另一种解法——窗口是「攒着别插」，双 stream 是「插了别堵门」，
两个一起 benchmark 按负载选。
预期：混合负载 decode P99 -30~50%（与 O9 二选一或叠加验证）。

### O13 graph 惰性捕获（ROADMAP P7 的落地设计）

启动只捕获 batch=1 与最大桶两图，其余桶第一遇才捕获（~0.5–1s 一次性代价），
空闲时后台补全高频桶；KV 预留按已捕获桶数逐步增长而非一次全扣。

实例：现在 `use_cuda_graph=True` 启动要等全桶捕获（每桶一次
capture + warmup），与 F3「冷启动秒级」卖点直接冲突。惰性化后 ~2s 进服务，
第一个 batch=16 请求多花 1s，之后零成本。教学场景（反复起停、跑单测）
收益最大。
预期：启动 -80%+（几十秒 → 2–3s），显存预留降到实际用到的桶。

### O14 fp8 KV 端到端强化

`kv_cache_dtype=fp8` 已存在（uint8 e4m3 存字节），补三件事让它从「能跑」
变「敢用」：(a) O1 页模型下按字节块共享/拷贝的路径打通；(b) attention
kernel 的 dequant（per-tensor scale 起步，head 级可选）；(c) golden 门禁
fp8 KV 单列一行精度指标，不达标不开默认。

实例：KV 每 token 每 rank 96KB（fp16）→ 48KB（fp8），同样的 22GB 池子
能装的 token 翻倍——并发 ×2 或上下文 ×2。隐藏收益：decode attention 是
纯带宽活，KV 读量减半 = attention 时间减半，8K 上下文约省 0.7ms/步；
与 O8 是乘法关系（一个减总量、一个提效率）。
预期：KV 容量 ×2；长上下文 TPOT -10~15%；精度风险由 golden 门禁兜底。

## 9 收益汇总

| 项 | 机制 | 预期收益 | 口径 |
|---|---|---|---|
| O2 循环重叠 | CPU 串行段藏进 GPU | TPOT -20~25% | batch=1 估算 |
| O3.1 one-shot AR | ring 延迟 → P2P 直写 | TPOT -15~20% | TP=2 PCIe 估算 |
| O3.1 → TP graph | launch 派发归零 | 与上两项合计 TPOT -40% | 合成口径 |
| O1 页 + Radix | 零拷贝共享 + 单 forward | 容量 ×5–16；多轮 TTFT 近零 | 负载相关 |
| O5 ngram | verify 一次付 k token | TPS ×1.3–2 | 代码/改写类负载 |
| O4 dequant 融合 | 省 fp16 权重往返 | MoE kernel -20~25% | A10 带宽瓶颈估算 |
| O7 prefill 桶图 | TTFT 里派发开销归零 | TTFT -5~10% | 大 prompt 场景 |
| O8 split-kv | attention 并行度 | 长上下文 TPOT -20~30% | 8K 上下文 |
| O14 fp8 KV | KV 量减半 | 容量 ×2；长上下文 TPOT -10~15% | 精度门禁前置 |
| O12 双 stream | compute/memory 互补 | decode P99 -30~50% | 混合负载 |
| O13 惰性捕获 | 启动只捕两桶 | 启动 -80%+ | 教学场景 |
| O6 / O9 / O10 / O11 | 调度细节 | 各 5–10%，组合有效 | 场景相关 |

组合后的整体预期：batch=1 短上下文 TPOT 7–8ms → ~3.5–4ms；8K 上下文
10–12ms → ~5–6ms；混合负载吞吐 ×1.5–2.5；启动进入秒级。均为机制推导
区间，P0 立零点后每项用 on/off 对照重新校准，达不到的项诚实砍掉。

## 10 落地阶段

| 阶段 | 内容 | 出口标准 |
|---|---|---|
| P0 | 立基线：TPOT / TTFT / TPS 三曲线 + nsys 全链路采集一次 | 各项收益的对照基准 |
| P1 | O2 + O3.1 + TP graph + O6.3 + O9 + O10 + O13 | TPOT -35%+，启动秒级 |
| P2 | O1 + O6.2（同版本）+ O14 | 容量与 TTFT 对照达标，golden 双布局全绿 |
| P3 | O5 ngram + O7 + O8 | accept ≥2；TTFT / 长上下文 TPOT 达标 |
| P4 | O3.2 TBO + O4 + O12 + O11 | 每项 on/off 对照正收益才保留 |

每阶段沿用 ROADMAP 第十一节的铁律：新东西必须有 on/off 对照 benchmark
归档进 `docs/release-vX.Y.Z.md`，golden 双跑；基线用 2×A10
Qwen3-30B-A3B-FP8 的 TPOT / 并发 TPS / prefix 命中 TTFT 三条曲线。

## 11 风险与明确不做

**风险三条**：O1 是唯一的大重构，走 A2 双布局并存迁移、golden 矩阵双跑、
随时回退，其余项全部是增量；O12 / O11 的收益依赖 SM 竞争与通信形态，
nsys 证伪就砍，不恋战；O14 的 fp8 KV 精度不达标就只开并发场景，
上下文场景退回 fp16，门禁说了算。

**明确不做的**：

- 不抄 sglang 的进程网格（EngineCore / scheduler / detokenizer 多进程全家桶）
  ——单人维护性是立身之本，O2 已拿到重叠收益。
- 不上 torch.compile——与 F3「冷启动秒级」直接冲突。
- 不引 C++ radix tree——blake2b 链式哈希对 4K 级 prompt 够快，等 prefix
  cache 真成为 CPU 热点再说。
- EP / DeepEP 不进本期——all_to_all 原语未落地，且 TP MoE 每 token 的
  全部专家都在本 rank 计算（只切中间维），不存在 expert 负载倾斜问题，
  2 卡 EP 的通信收益存疑；继续挂 ROADMAP v0.11 通信原语之后。
