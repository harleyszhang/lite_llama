
# 一、对你 comment 的回应

**1. 单进程 —— 你对,但结论要改写而不是删掉**

DP 走 `torch.multiprocessing` 起 `_dp_worker` 进程,TP 也起镜像 worker,所以"单进程"只在 TP=1/DP=1 成立。但**这个差异化仍然站得住,只是表述要精确**:vLLM v1 即使单卡也默认把 EngineCore 放独立进程(`VLLM_ENABLE_V1_MULTIPROCESSING` 默认开),而你在 TP=1/DP=1 时是真单进程、`pdb` 直接可用。正确表述:**"默认单卡路径全程单进程,断点可直达 kernel 调用点;并行模式才多进程"**。

**2. golden 测试 —— 你的判断对,但根因不是"不严谨",而是"静默跳过 + 覆盖面窄"**

我读了代码,设计其实不错:`cases.py` 有 4 个用例(single / batch_uniform / batch_mixed / batch8 跨 graph 桶)× 2 个 penalty,而且分两层——`test_eager_matches_graph` 不需要基线数据、永不过期,`test_matches_committed_golden` 对committed JSON。这个分层是对的。

真正的问题是三个:
- `pytestmark = [gpu, weights, slow]` → **无卡或无 checkpoint 时静默 skip**,所以它当不了"保险丝"——你以为绿了,其实根本没跑
- 只覆盖 Qwen2.5-0.5B、只有文本、只有 greedy、只有 TP=1、**不覆盖 continuous batching / 量化 / VL / DP**
- 没有 CI 强制,没有多模型矩阵

所以要做的是"**把它从可选测试升级成带 GPU runner 的强制门禁 + 扩展矩阵**",不是重写。

**3. DeepSeek 单 layer 对比 —— 这个想法很好,而且我算了,真的可行**

DeepSeek-V3 单层参数量:

| 部件 | 参数量 |
|---|---|
| 256 routed experts(gate+up+down,inter=2048,hidden=7168) | 11.27 B |
| shared expert | 0.044 B |
| MLA(q_a/q_b/kv_a/kv_b/o_proj) | 0.187 B |
| **合计** | **≈ 11.5 B** |

fp8 ≈ **11.5 GB → 单张 A10(24GB)装得下**;fp16 ≈ 23GB → 2×A10 可行。

**结论:单层 DeepSeek-V3/V4 在你现有硬件上完全跑得起来**,而且 vLLM 也能用同样方式跑单层对比。这从"跑不了"变成了"可实测",是这次讨论最有价值的一条。我把它升级成一个正式的**单层 harness 工具**(见第五节),它同时解决 MLA/DSA 正确性验证、性能对比、和"支持前沿架构"的简历叙事。

**4. 并行能力(chunked prefill / prefix caching / EP / DCP / CP / PD / LPLB)** —— 见第四节,我给了依赖图和 2×A10 可实测性判定。注意 **PD 分离在 2 卡上做 1P1D 是可演示的**,不需要更多卡。

关于 LPLB:我确定 DeepSeek 开源过 **EPLB**(Expert Parallelism Load Balancer,冗余专家 + 重平衡)。"LPLB" 这个名字我不能确认是同一个东西还是新变体,你给我个链接我再对齐,下面按"专家负载均衡"family 规划。

**5. Vision 重叠 —— 你对了一半,我的表述错了**

- 单请求内:vision → LLM 是严格数据依赖,**无法重叠**。你说得对。
- 离线 batch(N 张图):**跨请求流水是可以的**——encode 第 i+1 张时 prefill 第 i 张。所以离线 batch>1 有效。
- 但真正好摘的果子我上次说错了重点:**CPU 侧图像预处理(PIL 解码/resize/normalize)与 GPU 计算重叠**。这是纯 CPU-GPU 重叠,离线在线都有效,且实现简单。

修正后的表述:"CPU 预处理与 GPU 计算重叠(离线也有效)+ 跨请求 encode/prefill 流水(需 batch>1)";单请求内不承诺收益。

**6. fused_moe 自动调优 —— 你说对了,而且现在完全没有**

`fused_moe.py` L214 的 `_launch_config(num_tokens, quant_mode)` 是硬编码启发式,没有 autotune。`flashattention2_nopad.py` L41-44 的 autotune 被注释掉。

你的方案(warm-up 搜索)是对的,但要加一步才实用:**搜索结果按 (GPU型号, shape key) 持久化到磁盘**,否则每次启动都付搜索代价。vLLM 是手工提交 `E=...,N=...,device_name=....json` 配置文件;你可以做成**自动生成 + 自动复用**,这反而比他们先进。见第三节。

**7. 20 条凑数 —— 认。#2(1.2万行可读)、#11(数值对齐报告)确实是软的**,下面按你要的四个维度重建,数量减到 16,每条都是可验证的能力而不是感觉。


# 二、差异化亮点(重建:功能 / 性能 / 架构 / 自动化)

标注「已有」「待建」,每条附"为什么 vLLM/SGLang 不会做"。

## 功能维度

| # | 亮点 | 状态 | 他们为何不做 |
|---|---|---|---|
| F1 | **任意模型任意单层的独立运行 harness**:单层跑 forward、对比 HF、测延迟/显存 | 待建 | 他们没有这个抽象;大模型验证靠整模型跑 + 8 卡 |
| F2 | 默认单卡路径全程单进程,`pdb` 可直达 kernel 调用点 | 已有 | v1 刻意多进程隔离,不会回退 |
| F3 | 冷启动秒级(无 CUDA C++ 编译、无 torch.compile、graph 捕获可关) | 已有 | 他们为长驻服务优化,启动 30s–2min 不在乎 |
| F4 | 后端缺失自动回退原生,永不硬失败 | 待建 | 他们缺库常直接报错退出 |

## 性能维度

| # | 亮点 | 状态 | 差异点 |
|---|---|---|---|
| P1 | **固定 shape 极致特化**:为单个模型单个 shape 手调到最优,不必通吃 | 待建 | 他们必须覆盖所有 expert 配置/head 组合 |
| P2 | MoE dequant 与 grouped GEMM 融合(消除中间张量往返 HBM) | 待建 | A10 无 fp8 算力,这是带宽瓶颈的最大单点;他们主攻 Hopper 不优化 Ampere 路径 |
| P3 | batch=1 / 低并发 decode 延迟(无 IPC、无跨进程序列化) | 已有部分 | 他们的多进程 IPC 在小 batch 占比高 |
| P4 | 峰值显存可预测(权重/KV/激活/graph 逐项可查可控) | 待建 | 他们的 `gpu_memory_utilization` 是黑盒比例 |

## 架构设计维度

| # | 亮点 | 状态 | 差异点 |
|---|---|---|---|
| A1 | **稀疏后端注册表 + 保底行**:外部后端只注册擅长的 (scheme, arch) 格,其余自动落原生 | 待建 | 避免 N×M 类爆炸,这是单人能维护多后端的前提 |
| A2 | **KV 布局可插拔**:token-level / paged / 未来 DiffKV 并存,重构可增量迁移随时回退 | 待建 | 把高风险大重构变成低风险增量,他们靠人力硬切 |
| A3 | 调度策略 policy 化:one-shot / continuous / chunked-prefill 共用一个 step 循环 | 待建 | 顺带消灭现有"两套生成循环"债务 |
| A4 | 模型定义薄:一个模型 = 一个类体十几行 + 一行注册 | 已有 | 他们的模型文件动辄上千行 |
| A5 | 每个 Triton kernel 旁并排 PyTorch 参考实现,作为语义定义者 | 部分已有 | 他们参考实现散在测试里 |

## 自动化工具维度(你要的重点,也是最能体现工程能力的)

| # | 亮点 | 状态 | 说明 |
|---|---|---|---|
| T1 | **warm-up 自动调优 + 配置持久化**:按 (GPU型号, shape key) 搜索最优 tile 并落盘复用 | 待建 | vLLM 是**手工提交** JSON 配置,你做**自动生成**,更先进 |
| T2 | **真实负载 shape 采集器**:跑一遍业务负载,导出所有出现过的 GEMM/attn/MoE shape,喂给 T1 | 待建 | 让调优针对真实分布而非猜测 |
| T3 | **后端选择 explain**:每个算子选了谁、候选有谁、为什么被排除 | 待建 | vLLM backend 选择不透明是社区高频吐槽点 |
| T4 | **性能回归看门狗**:benchmark 入库,劣化超阈值 CI 报警 | 待建 | 保证"更快"不会悄悄退步 |
| T5 | **golden 门禁自动化**:多模型 × 多路径矩阵,自托管 GPU runner 强制跑,禁止静默 skip | 待建(有基础) | 修掉当前"skip 了还显示绿"的假安全 |
| T6 | **数值对齐门禁**:每个外部后端 vs 原生 max-abs-diff 阈值化,超标阻止合入 | 待建 | 使测试量为 N+M 而非 N×M |

这 16 条里,**T1/T2/T3 + F1 + A1/A2 是真正没人做的**。自动化工具那一组尤其值得投入:它既是差异化,又是你一个人能维护这个项目的前提,面试讲"我怎么用自动化保证单人维护多后端框架的质量"是很强的叙事。


# 三、三个地基设计方案

后面所有功能都挂在这三个地基上,顺序不能反。

## 地基 1:真分页 KV(解锁 chunked prefill + prefix caching)

现状问题:`block_size` 恒为 1,按 token 行分配 + refcount,`alloc_contiguous_kvcache` 慢路径每 decode 步 3 次 D2H 同步,每请求预占 `max_seq_len` 行。

设计要点:

- `block_size=16`(A10 上 16 比 32 更省碎片),`block_table[req_id] → [block_ids]`
- **通过 A2(KV 布局可插拔)并存迁移**:新 `PagedLayout` 与现有 `TokenLayout` 同时注册,env 切换。golden 矩阵在两种布局下都必须过,通过后再删旧的。这是把"高风险重写"降级为"可回退增量"的关键手法。
- 分配路径纯 GPU,消除 `nonzero` + `.item()` 同步
- block hash(prompt token 前缀哈希)+ refcount → prefix caching 的挂点
- attention kernel 同步改造:`flash_decoding` 与 `flash_attention2_no_pad` 从行索引寻址改块寻址

验收:golden 全绿;并发容量提升(不再预占 max_seq_len);decode 步无 D2H 同步。

## 地基 2:多后端稀疏注册表

```
register(op="linear", scheme="fp8_blockwise", arch=">=sm90", provider=DeepGemm,     priority=100)
register(op="linear", scheme="*",             arch="*",      provider=NativeTriton, priority=0)  # 保底行
```

选择器 = f(scheme, arch, `is_available()`) → 取最高优先级。外部后端全放 `backends/`,全部 optional extra,永不进核心依赖。配套 T3 的 explain 输出决策链。

先只做 linear/GEMM 打通全链路,再 attention → MoE → 通信。

## 地基 3:自动调优与配置持久化(T1+T2)

这是你 comment 6 的正式方案:

| 阶段 | 做什么 |
|---|---|
| 采集 | T2 shape 采集器跑一遍真实负载,导出 shape 清单(含出现频次) |
| 搜索 | warm-up 或离线对高频 shape 搜索 tile/num_warps/num_stages |
| 落盘 | 按 `(gpu_name, op, shape_key, dtype)` 存 JSON 到用户缓存目录 + 可选提交进仓库 |
| 复用 | 启动时按 key 命中即用,未命中回退启发式并可后台补搜 |
| 门禁 | T4 看门狗对比历史最优,劣化报警 |

覆盖对象:`fused_moe`(`_launch_config` 替换)、`flashattention2_nopad`(把注释掉的 144 组配置启用为搜索空间)、量化 GEMM(`_launch_config`)。

# 四、并行与服务能力路线

依赖关系(箭头 = 前置):

```
真分页 KV ──┬─→ chunked prefill ──→ PD 分离
            └─→ prefix caching
通信原语补全 ─┬─→ EP ──→ 专家负载均衡(EPLB 族)
              ├─→ DCP
              └─→ CP/PCP
```

当前通信层只有 `all_reduce`(SUM) 和 `all_reduce_min`,**缺 all_gather / reduce_scatter / all_to_all / P2P send-recv**——这是 EP/DCP/CP 的共同前置。

| 能力 | 前置 | 2×A10 可实测? | 优先级 | 说明 |
|---|---|---|---|---|
| chunked prefill | 分页 KV + token budget 调度 | 可 | **高** | varlen attention 你已有,主要改调度和 KV 部分写 |
| prefix caching | 分页 KV + block hash | 可 | **高** | 多轮对话/共享 system prompt 收益直观 |
| EP | all_to_all | 可(EP=2) | 中高 | Qwen3-MoE 30B-A3B 上可实测 |
| DCP | all_gather + LSE 校正 | 可(DCP=2) | 中 | 长上下文 decode 扩容;MLA 场景价值最大 |
| CP / PCP | P2P ring 或 zigzag 切分 + LSE 合并 | 可 | 中 | 降 prefill 延迟 |
| **PD 分离** | chunked prefill + KV 传输 connector | **可(1P1D)** | 中 | 2 卡就能演示,不需要更多卡 |
| 专家负载均衡 | EP + 专家负载统计 | 可 | 低 | 冗余专家 + 重平衡;需先确认 LPLB 规格 |

关键判断:**全部能力在 2×A10 上都至少能做到"机制正确 + 小规模可测"**。收益规模肯定不如大集群,但"实现了且验证正确"对简历足够,而且 chunked prefill / prefix caching 的收益在你硬件上就是真实的。

# 五、DeepSeek 单层 harness(你 comment 3 的正式方案)

这是我认为**投入产出比最高的一个新增工具**,它一次解决四件事。

设计:

- 输入:HF config + 层索引 + 可选真实权重(或随机初始化)
- 只实例化一层,跑 forward,支持 prefill/decode 两种输入形态
- 输出:输出张量 vs HF 同层的 max-abs-diff、逐算子延迟、峰值显存、选中的后端
- 同一 harness 可跑 vLLM 的对应单层做横向对比

价值:

| 用途 | 说明 |
|---|---|
| MLA 正确性验证 | 不需要 671B 权重,随机权重也能验证结构与数值路径 |
| DSA / Lightning Indexer 验证 | MQA logits + TopK 稀疏选择的机制正确性 |
| 性能对比 | 单层 fp8 ≈ 11.5GB,单张 A10 可跑,vs vLLM 同层可比 |
| 产品化为 F1 亮点 | 任意模型任意层的通用调试工具,这是 vLLM 没有的 |

前置:必须先拆薄 `models/base.py` 的 `Attention`(当前硬编码 fused-KV + RoPE + RMSNorm,MLA 接不进来)。这和地基 2 是同一次改造。

补充建议:先用 **DeepSeek-V2-Lite(16B,MLA 完整)** 跑整模型验证端到端正确性,再用单层 harness 打 V3/V4。两条路互补——一个验证全链路,一个验证前沿架构。

# 六、版本计划

| 版本 | 主题 | 内容 | 验收 |
|---|---|---|---|
| **v0.4** | 可信基线 | 修 TP 采样 RNG 不同步;golden 升级为强制门禁(自托管 GPU runner、禁止静默 skip、扩到 continuous/量化/VL/DP 矩阵);T4 性能看门狗;AWQ/GPTQ 修复或撤下宣称 | 无卡时 CI 明确报"未验证"而非绿;TP=2 采样不 diverge |
| **v0.5** | 自动化调优 | T2 shape 采集器 + T1 warm-up 搜索 + 落盘复用;应用到 fused_moe / nopad attention / 量化 GEMM;w4a16 重写为 `tl.dot` | 高频 shape 有落盘配置;w4a16 出前后对比数字 |
| **v0.6** | 多后端骨架 | 地基 2(注册表+探测+选择+T3 explain+T6 对齐门禁);先只做 linear;attention 接口拆薄 | 一条命令切后端并解释原因;缺库自动回退 |
| **v0.7** | 分页 KV | 地基 1,KV 布局可插拔并存迁移 | 两种布局 golden 都绿;decode 无 D2H 同步 |
| **v0.8** | 调度能力 | chunked prefill + prefix caching;调度 policy 化,合并双引擎循环 | 长 prompt 期间 decode 不停顿;共享前缀命中率可观测 |
| **v0.9** | 前沿架构 | 单层 harness(F1);MLA(DeepSeek-V2-Lite 端到端 + V3/V4 单层);DSA indexer | HF 单层 max-abs-diff 达阈值;V3 单层 vs vLLM 对比数据 |
| **v0.10** | 并行扩展 | 通信原语补全;EP(EP=2);DCP(DCP=2) | Qwen3-MoE 上 EP 可跑并有数据 |
| **v0.11** | 服务能力 | PD 分离 1P1D;CP/PCP;专家负载均衡 | 1P1D 端到端可演示 |
| **v1.0** | 收口 | API 冻结、完整 benchmark 矩阵、文档站 | 公开 API 语义稳定 |

顺序原则:**v0.4 必须第一**——没有可信的 golden 门禁和 benchmark 基线,后面每个优化都说不清是真快了还是偷偷坏了。这也是你 comment 2 指出的问题的直接回应。

# 七、立即可动的第一步

我建议 v0.4 里先做**两件互相独立、都能当天见效**的事:

1. **golden 门禁自动化**:去掉静默 skip(无卡时显式 FAIL 或输出 "UNVERIFIED" 状态)、扩展 `cases.py` 覆盖 continuous batching 路径和量化路径、`scripts/golden_tokens.py` 支持多模型批量重录。
2. **修 TP 采样 RNG 不同步**:worker 采样参数改为从 rank0 broadcast(而非硬编码 `temperature=0.0`),并 broadcast 同一 seed 给所有 rank。

要不要我现在开始做第 1 件?它不需要 GPU 就能改完并验证逻辑,而且做完你后面所有重构都有真正的保险丝。或者你更想先修 TP bug——那个是正确性问题,也可以并行。

另外:这份计划要不要我写成 `ROADMAP.md` 存进仓库?后面每个版本可以直接在上面勾进度。