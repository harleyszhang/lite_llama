# Release v0.11.5 — 计算通信重叠

**Date:** 2026-09-03 **Branch:** `support_deepseekv4` **Theme:** 四层重叠原语（L1 host↔device、L2 TBO、L3 chunked all-reduce、L4 tile-signaling）+ DP×CUDA Graph 交叉验证 + DeepSeek V4 端到端；每个特性独立 on/off 对照数据，正收益与负收益同表发布

## Summary

v0.11.5 沿三条轴把「通信时间藏在计算后面」做成可独立开关的原语，并用一个组合矩阵把三者交叉验证——矩阵里每一个数字都是同一负载、同一引擎、只动开关的对照：

* **L1（pinned-copy 重叠，默认开）**：上传/回读走 copy stream + pinned 环形缓冲，`StreamPool` + `Timeline`（CUDA event 级证据）。
* **L2（two-batch overlap，默认关）**：TP decode 步切两半 ping-pong，半 A 的 o_proj all-reduce 在通信流上飞行时，半 B 的注意力 GEMM 占着 SM。deferred-AR 上下文 + `execute_overlapped` 交错原语。
* **L3（chunked all-reduce，默认关）**：行并行 GEMM 输出按行分块，第 k 块的 AR 上线时第 k+1 块的 GEMM 在算。
* **L4（tile-signaling，单卡 kernel 级）**：persistent Triton 生产者每完成一个输出 tile 就 `atomic_xchg` 发布 epoch，消费者 kernel 有界自旋等 tile，GEMM→epilogue 逐 tile 流水。
* **P8（DP + CUDA Graph）**：每副本独立 capture/replay（tp=1/副本，graph 内无集合通信），TPOT -80%，DP2 吞吐 5.1×。

**质量故事（本版最有价值的产出之一）**：组合矩阵（M7）抓到了单特性测试漏掉的 TBO 数值回归——闭包在构建期快照了 `state.hidden`，所有层的 segment 都拿初始 embedding 当输入，交错输出与 eager 基线 maxdiff 27+；而「切分本身正确」的隔离证据来自 manual-halves 对照（同一 `TboSplitter` 切分、逐半顺序 forward 拼接 → maxdiff 0.0000）。修复为闭包运行时惰性读 state（`lite_llama/batch_overlap/two_batch_overlap.py` 的 `_attn_segment` docstring 记录了完整机理）。**这正是「不同优化特性做混合、交叉测试」在计划里的目的：单特性的 parity 测试过 ≠ 组合下依然对。**

**模型覆盖**：dense Qwen2.5-1.5B、DeepSeek-V2-Lite（MLA+MoE）、V3-4layers（biased noaux_tc 路由）、V4 裁剪版（mHC 残差/压缩器/Lightning Indexer/Hash MoE，transformers 5.8 随机初始化 checkpoint——V4 无公开权重）。

## 架构

```
三条轴，每条独立开关，汇聚于一个组合矩阵：
┌──────────────────────────────────────────────────────────────┐
│ A 轴 host↔device   L1  executor/overlap.py    StreamPool+Timeline    │
│ C 轴 compute↔comm  L2  batch_overlap/two_batch_overlap.py (TBO)      │
│                   L3  batch_overlap/comm_overlap.py (chunked AR)    │
│                   └─ 同一 row_parallel_forward 分发点：TBO>L3 退位     │
│ B 轴 单卡 kernel   L4  kernels/tile_signal.py (persistent+epoch)     │
│ 并行度             P8  DP×CUDA Graph（每副本独立 capture）           │
└──────────────────────────────────────────────────────────────┘
```

包结构（`lite_llama/batch_overlap/`，对齐 sglang 的 `batch_overlap` 布局）：

* `operations.py` — stage/yield 交错原语（`YieldOperation` + `execute_overlapped`）
* `comm_overlap.py` — 通信流底座：`CommStreamPool`（每 device 一条 NCCL 流）、`DeferredArContext`（defer/fence/collecting/drain）、`CommOverlapPolicy`（L3 策略）与 `row_parallel_forward` 单一分发点
* `two_batch_overlap.py` — L2 executor：`TboSplitter`（行切半 + KV 元数据窄化）+ `TwoBatchOverlap`（双 op 流交错）
* 模型侧唯一侵入点：`models/base.py` 的 `DecoderLayer.forward_attn_stage/forward_mlp_stage` 两段拆分（原 `forward` = 两段顺序调用，行为不变；段边界恰好是 o_proj 的行并行 all-reduce）

## Feature

### L2 two-batch overlap（含如实发布的负收益）

```bash
# TP2 对照（batch 8/16/32，TPOT on/off + greedy 一致性 + timeline）
python benchmarks/bench_overlap_l2.py --timeline
```

2×A10 **PCIe** 上 TBO 是负收益，数据与机理同表发布（`benchmarks/logs/bench_overlap_l2.json`）：

| batch | TBO off | TBO on | 变化 | greedy 一致 |
| --- | --- | --- | --- | --- |
| 8 | 27.9 ms | 65.9 ms | +136% | 6/8 |
| 16 | 27.0 ms | 65.6 ms | +142% | 16/16 |
| 32 | 34.4 ms | 67.3 ms | +96% | 28/32 |

机理（不是猜测，两个独立测量支撑）：① nsys trace 显示 on 臂 NCCL kernel 数 6778→12934——每半各一次 AR，消息减半但 PCIe 小消息 AR 有延迟下限，翻倍次数吃掉全部理论重叠；② NCCL kernel 在两端到达时间差内 spin-wait，A 半的 AR 等 B 半 launch 的间隙把 SM 白白烧掉。**重叠本身是真实发生的**（timeline 848 对重叠共 84.2 ms；nsys kernel 级 9.8% NCCL 时间被 compute 隐藏，off 臂为 0.0%），只是收益跑不赢代价。6/8、28/32 的分歧行是低置信度输出上「分批 AR vs 整批 AR」的 bf16 归约顺序差——与 batch16 的 16/16 完美一致同表呈现，不隐藏。

**取舍**：L2 保持默认 off，等 NVLink 机器（未测）再开。`TboPolicy.min_rows`（默认 8）是激活阈值。

### L3 chunked all-reduce

```bash
python benchmarks/bench_overlap_l3.py --json --timeline
```

GEMM 输出行分块、每块 GEMM 落地即上通信流（`docs/benchmark_logs/overlap_l3_20260903_215551.json`）：TP2 Qwen2.5-1.5B batch 16，TTFT 33.25→33.07 ms（-0.6%），timeline 记录 224 个 comm region 与 111 对真重叠（9.78 ms）。`L3_MIN_CHUNK_ROWS=256` 行下限：再细的分块在 PCIe 上付更多次小消息固定成本。分发点优先级 TBO > L3（同一 all-reduce 位点不叠加切分），组合矩阵验证退位成立（见下）。

### L4 tile-signaling

```bash
python benchmarks/bench_overlap_l4.py
```

单卡 kernel 级原语，与互联无关（`docs/benchmark_logs/overlap_l4_20260903_104621.json`）：GEMM→SiLU·mul 逐 tile 流水 vs 串行两 kernel，A10（72 SM）上大形状 +8.0~+13.7%（4096×4480×1536：5.85→5.05 ms），小形状负收益（64×4480×1536：-15.5%）如实入表——persistent kernel 的常驻占用在 tile 少时是纯开销。死锁规避：生产者+消费者 grid 之和 ≤ #SM，host 侧 watchdog 兜底。

### DP + CUDA Graph（P8）

```bash
python benchmarks/bench_dp_graph.py
```

`docs/benchmark_logs/dp_graph_20260903_143056.json`（Qwen3-0.6B，batch 16/副本，128 步）：

| 配置 | TPOT | 吞吐 |
| --- | --- | --- |
| dp1 eager | 25.9 ms | 618 tok/s |
| dp1 graph | 5.2 ms | 3102 tok/s |
| dp2 eager | 26.4 ms | 1200 tok/s |
| dp2 graph | 5.2 ms | **6162 tok/s** |

每副本独立 capture/replay 无锁步（tp=1/副本，graph 内无 NCCL 集合通信），DP2 下吞吐 5.1×、TPOT -80%；capture 耗时 +2.4 s、显存增量已记入 JSON。测试 `tests/engine/test_dp_cuda_graph.py` 断言双副本均有 captured graphs 且 greedy 输出一致。

### DeepSeek V4（裁剪版端到端）

V4 无公开权重（仅 config.json），用 transformers 5.8 随机初始化裁剪 checkpoint（借 Qwen tokenizer）做 parity 与性能对照：

* `modules/deepseek_v4.py`：mHC 残差（Sinkhorn 混合 + hc_head）、SWA/CSA 混合注意力（head_dim=512 latent KV + fused_wqa_wkv）、O-LoRA o_proj（inv-RoPE + wo_a einsum + wo_b 行并行）、Compressor（CR 累积→RMSNorm→RoPE→压缩 cache）、Lightning Indexer（topk 选块）、Hash MoE（num_hash_layers 层 tid2eid）+ sqrtsoftplus 路由
* `models/deepseek_v4.py`：组装 + registry + packed_modules 映射；TP2 下 heads/o_groups 整除校验
* 测试 `tests/models/test_deepseek_v4.py` 8 项（rotary/HC/router×2/MoE/decoder/e2e/incremental）+ `tests/distributed/test_tp2_v4.py` TP2 一致性
* benchmark：`benchmarks/bench_deepseek_v4.py`（vs transformers 速度对照，含噪声基线）
* **fp4 权重暂不支持**：本版以 bf16/fp16 unquantised 权重为 parity 基础，fp4 留待后续版本

## 组合矩阵与精度门禁（M7）

### 组合矩阵：L1×L2×L3 八格

`benchmarks/bench_overlap_matrix.py`（`docs/benchmark_logs/overlap_matrix_final.json`，Qwen2.5-1.5B TP2 batch 16，1024 tok/格）：

| 组合 | TPOT | vs baseline | 输出一致性 |
| --- | --- | --- | --- |
| baseline | 28.47 ms | — | — |
| l1 | 27.25 ms | -4.3% | 16/16 |
| l2 | 66.09 ms | +132% | 16/16 |
| l3 | 27.69 ms | -2.7% | 16/16 |
| l1l2 | 65.20 ms | +129% | 16/16 |
| l1l3 | 27.89 ms | -2.0% | 16/16 |
| l2l3 | 64.94 ms | +128% | 16/16（≡ l2） |
| all | 66.42 ms | +133% | 16/16（≡ l1l2） |

`l2l3`/`all` 两格的存在就是验证 L3 在 TBO 激活时退位：它们的输出与 `l2`/`l1l2` 逐字相同（`demotion_holds: true`），TPOT 也在 l2 的噪声带内——分发点优先级不是文档声明而是被测行为。

### 模型矩阵：baseline vs 推荐组合（L1+L3）

| 模型 | TPOT 变化 | TTFT 变化 | 输出一致性 |
| --- | --- | --- | --- |
| qwen2.5-1.5b | -0.5% | +0.4 ms | 16/16 |
| deepseek-v2-lite | -1.8% | -15.9 ms | 16/16 |
| deepseek-v3-4layers | -1.7% | -0.0 ms | 14/16* |
| deepseek-v4-trimmed | -2.5% | -0.6 ms | 16/16 |

\* V3-4layers 的 14/16 与 overlap 无关：**两次全关的 baseline 互比同样 14/16**，分歧行输出为乱码 token（平坦 logits 上 argmax 的 run-to-run 抖动）。这是引擎固有噪声的测量，不是特性引入的漂移。

### golden 门禁（双遍）

```bash
pytest tests/golden/ -q                                        # 默认：9 passed
LITE_LLAMA_OVERLAP=1 LITE_LLAMA_TBO=1 LITE_LLAMA_COMM_OVERLAP=1 \
LITE_LLAMA_TBO_MIN_ROWS=2 pytest tests/golden/ -q              # 全开：9 passed
```

第二遍把 TBO 激活阈值压到 2 强制 TP2 golden 走 `forward_tbo` 路径，V2-Lite 的 greedy/logprob parity 预算（mean 0.4/max 2.5 nats，2× 实测漂移校准——校准证据链见测试内注释）依然全绿。

### nsys kernel 级证据

`docs/benchmark_logs/nsys_overlap_report.md`（payload：`benchmarks/nsys_overlap_payload.py`，分析器：`benchmarks/nsys_overlap_report.py`）：

| trace | gpu | NCCL kernels | hidden under compute |
| --- | --- | --- | --- |
| overlap off | 0 | 6778 | 0.00 ms (0.0%) |
| overlap off | 1 | 6778 | 0.00 ms (0.0%) |
| overlap on | 0 | 12934 | 206.12 ms (9.8%) |
| overlap on | 1 | 12934 | 126.04 ms (8.6%) |

off 臂 NCCL 与 compute 零重叠（阻塞 AR 在 compute stream 上串行）；on 臂 kernel 级真并发出现。NCCL kernel 数翻倍（12934）同时如实呈现——这是 L2 负收益机理的直接观测。

## 全量回归

M7 收尾时全套测试分批跑过（本轮改动：TBO 闭包惰性读取修复、grouped_topk 单 expert 组退化、外部 batch_overlap 包重构适配、test_checkpoint_index MTP/trim 层豁免）：

* 批 1 `tests/models/ + tests/kernels/ + tests/batch_overlap/ + tests/executor/ + tests/engine/ + tests/distributed/`：1288 passed / 72 skipped（唯一失败为 HEAD 既有：V3-MTP checkpoint 的 MTP 层 key 未豁免，worktree 对照验证后修复，36 passed）
* 批 2 `tests/ops/ + tests/tools/ + tests/compile/ + tests/config/ + tests/entrypoints/ + tests/evals/ + tests/multimodal/ + tests/platform/ + tests/utils/ + tests/modules/ + tests/test_imports.py`：480 passed / 10 skipped
* `tests/golden/`：默认与 overlap 全开双遍 9 passed

## 已知边界（如实标注）

1. **互联**：全部数据来自 2×A10 **PCIe**（无 NVLink 硬件）。L2 的负收益结论、L3 的微收益、nsys 的 9.8% 隐藏率都是 PCIe 事实；NVLink 上的一切**未测**，不做推断。
2. **L2 默认 off**：负收益路径不进默认配置；`LITE_LLAMA_TBO=1` 显式开启。
3. **V4 fp4**：本版仅 bf16/fp16 unquantised 权重（parity 基础）；fp4 量化加载留待后续。
4. **V4 TBO 未接线**：mHC 栈的段结构与两段拆分不匹配，本版 V4 不走 `forward_tbo`（矩阵里 V4 只测 L1+L3 组合）。
5. **裁剪 checkpoint 的 grouped_topk**：V3-4layers（8 experts/8 组）落在所有参考实现（transformers/vLLM）都会崩的几何上；lite_llama 按数学极限退化处理（单 expert 组分数 = top-2 和的极限 = 该 expert 分数），`tests/kernels/test_grouped_topk_kernel.py` 锁定该语义。

## 图表

* `docs/images/overlap_combination_matrix.png` — 八格组合矩阵（L2 类红色标注）
* `docs/images/overlap_model_matrix.png` — 四模型 baseline vs L1+L3
* `docs/images/overlap_l2_tbo.png` — L2 负收益（batch 维度）
* `docs/images/overlap_l4_tile_signal.png` — L4 各形状正负收益
* `docs/images/dp_cuda_graph.png` — DP×Graph TPOT/吞吐

生成：`python benchmarks/plot_overlap_gains.py`（读 JSON logs，图数同源）。
