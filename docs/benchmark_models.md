## 一 量化 benchmark 性能测试

### 量化内核性能（W8A16 / W4A16 / SmoothQuant）

以下为量化 Triton 内核在 A10 (24 GB, SM86) 上的 `triton.testing.do_bench` 实测结果。基准为 cuBLAS fp16 `F.linear`；加速来自减半（或减至 1/4）的 HBM 权重读取量。

#### W8A16 (fp8-e4m3, 128×128 block scales)

| Shape (M×N×K) | fp16 (ms) | w8a16 (ms) | 加速比 | 场景 |
|---------------|-----------|------------|--------|------|
| 1×4096×4096 | 0.086 | 0.053 | **1.62×** | decode |
| 1×11008×4096 | 0.199 | 0.116 | **1.71×** | decode (MLP up) |
| 8×4096×4096 | 0.084 | 0.051 | **1.65×** | decode batch |
| 64×4096×4096 | 0.091 | 0.055 | **1.64×** | small prefill |
| 512×4096×4096 | 0.191 | 0.280 | 0.68× | prefill (compute-bound) |

结论：decode 阶段（M≤64）稳定 **1.6–1.7× 加速**；prefill 阶段（M≥512）内核为 compute-bound，fp8 路径无优势（此时应回退到 cuBLAS fp16）。

#### W4A16 (int4, group_size=128)

| Shape (M×N×K) | fp16 (ms) | w4a16 (ms) | 加速比 | 备注 |
|---------------|-----------|------------|--------|------|
| 1×4096×4096 | 0.086 | 0.176 | 0.49× | 未优化 |
| 8×4096×4096 | 0.084 | 0.311 | 0.27× | 未优化 |
| 64×4096×4096 | 0.091 | 0.832 | 0.11× | 未优化 |

> ⚠️ W4A16 内核当前为功能实现，尚未做 tile 级优化（逐元素 unpack + outer product）。
> 后续计划：向量化 unpack、`tl.dot` 替代 outer product、autotuning。
> 内存节省仍然有效：30B 模型 int4 权重仅占 ~15 GB（fp16 需 ~61 GB）。

#### SmoothQuant W8A8 (dynamic per-token)

| Shape (M×N×K) | fp16 (ms) | smoothquant (ms) | 加速比 | 备注 |
|---------------|-----------|------------------|--------|------|
| 8×256×512 | — | ✓ | — | 精度验证通过 |
| 64×2048×2048 | — | ✓ | — | 精度验证通过 |

精度：相对 fp32 参考的相对误差 < 2%（含激活 + 权重量化双重噪声）。

#### 量化算子精度汇总

| 量化方案 | 相对误差 (vs fp32) | 权重内存节省 |
|----------|-------------------|-------------|
| fp8 blockwise (128×128) | < 0.04% | 2× |
| int8 per-channel | < 0.03% | 2× |
| int4 group-wise (AWQ/GPTQ) | < 5% | 4× |
| smoothquant W8A8 | < 2% | 2× |

复现：

```bash
# 内核精度测试
python -m pytest tests/kernels/test_quantization.py -v

# 性能基准
python -c "
import torch, triton
from lite_llama.kernels.quantization import w8a16_matmul
M, N, K = 1, 4096, 4096
x = torch.randn(M, K, device='cuda', dtype=torch.float16)
qw = torch.randn(N, K, device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
sc = torch.ones(32, 32, device='cuda')
print(triton.testing.do_bench(lambda: w8a16_matmul(x, qw, sc, group_n=128, group_k=128)))
"
```

## 二 模型 e2e benchmark 汇总

同一批 checkpoint 的端到端性能，从两个互补视角测：一节拿 lite_llama 与 HF transformers 同口径对照，回答"比裸 transformers 快多少"；另一节拿 lite_llama 自己关/开 CUDA graph 对照，回答"graph 优化本身值多少"。前者 lite_llama 侧默认就跑在 graph 模式（`TextGenerator` 默认 `use_cuda_graph=True`），所以两表的 lite_llama 数字同源——只是 gen_len 与 TPOT 统计方式不同（整体摊销中位数 vs 逐步间隔均值），数字接近而不相等，各按原口径保留。

两节的测试矩阵相同：A10 22 GiB 上放得下的全部纯文本 checkpoint。未包含：**Qwen2.5-0.5B**（本机无权重，历史数字见 git 历史）、**Qwen-1_8B**（第一代 `qwen` model_type，不在支持列表，加载即被 registry 拒绝）、**llava / Qwen3-VL**（多模态，需另走 `VisionGenerator` 路径）、**Qwen3-30B-A3B 系 / Qwen3-Next-80B**（单卡放不下，30B-FP8 需 `--tensor-parallel-size 2` 双卡）、**Qwen3-MoE-Tiny**（2 层 4 专家的玩具 checkpoint，fp32 存储 547 MB，数字仅证明 qwen3_moe 架构与 fused_moe kernel 在三层 dispatch 下端到端可用，不代表 MoE 吞吐量级）。

### lite_llama vs HF transformers（examples/benchmark.py）

下表是用重构后的 `examples/benchmark.py` **实测**得到的结果（贪心解码、两端同一 tokenizer 统计输出 token、两端自然 EOS 停止、`torch.cuda.synchronize` 计时、取中位数）。指标口径对齐 vLLM/SGLang serving benchmark：

- **TTFT**（首 token 时延，s）= 预填充延迟；
- **TPOT**（每输出 token 时延，ms）= `(latency - ttft) / (output_len - 1)`；
- **TGS**（token 生成速度，tokens/s）= `总输出 token / latency`（聚合吞吐）。

| 模型 | GPU | batch | gen_len | 引擎 | TTFT (s) | TPOT (ms) | TGS (tok/s) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| Qwen1.5-0.5B | A10 | 8 | 128 | lite_llama | 0.0180 | 3.04 | 2535.3 |
| Qwen1.5-0.5B | A10 | 8 | 128 | transformers | 0.0215 | 19.65 | 406.8 |
| Qwen1.5-0.5B | A10 | 16 | 256 | lite_llama | 0.0192 | 3.55 | 4424.8 |
| Qwen1.5-0.5B | A10 | 16 | 256 | transformers | 0.0238 | 20.02 | 798.4 |
| Qwen2.5-1.5B | A10 | 8 | 128 | lite_llama | 0.0219 | 9.36 | 844.5 |
| Qwen2.5-1.5B | A10 | 8 | 128 | transformers | 0.0271 | 23.67 | 337.6 |
| Qwen2.5-1.5B | A10 | 16 | 256 | lite_llama | 0.0228 | 8.69 | 1830.1 |
| Qwen2.5-1.5B | A10 | 16 | 256 | transformers | 0.0289 | 24.21 | 660.4 |
| Qwen2.5-1.5B-Instruct | A10 | 8 | 128 | lite_llama | 0.0216 | 8.24 | 958.8 |
| Qwen2.5-1.5B-Instruct | A10 | 8 | 128 | transformers | 0.0277 | 24.14 | 331.1 |
| Qwen2.5-1.5B-Instruct | A10 | 16 | 256 | lite_llama | 0.0225 | 8.51 | 1868.2 |
| Qwen2.5-1.5B-Instruct | A10 | 16 | 256 | transformers | 0.0278 | 23.62 | 677.0 |
| Qwen2.5-3B | A10 | 8 | 128 | lite_llama | 0.0279 | 18.67 | 426.6 |
| Qwen2.5-3B | A10 | 8 | 128 | transformers | 0.0361 | 35.82 | 223.3 |
| Qwen2.5-3B | A10 | 16 | 256 | lite_llama | 0.0364 | 19.23 | 828.9 |
| Qwen2.5-3B | A10 | 16 | 256 | transformers | 0.0468 | 35.18 | 454.1 |
| Qwen3-0.6B | A10 | 8 | 128 | lite_llama | 0.0253 | 4.23 | 1820.3 |
| Qwen3-0.6B | A10 | 8 | 128 | transformers | 0.0317 | 28.94 | 276.2 |
| Qwen3-0.6B | A10 | 16 | 256 | lite_llama | 0.0256 | 4.70 | 3346.7 |
| Qwen3-0.6B | A10 | 16 | 256 | transformers | 0.0329 | 28.65 | 558.2 |
| Qwen3-0.6B-FP8 | A10 | 8 | 128 | lite_llama | 0.0293 | 4.09 | 1864.5 |
| Qwen3-0.6B-FP8 | A10 | 8 | 128 | transformers | 0.0311 | 29.08 | 274.9 |
| Qwen3-0.6B-FP8 | A10 | 16 | 256 | lite_llama | 0.0291 | 4.53 | 3460.5 |
| Qwen3-0.6B-FP8 | A10 | 16 | 256 | transformers | 0.0308 | 27.92 | 572.8 |
| Qwen3-1.7B | A10 | 8 | 128 | lite_llama | 0.0264 | 9.28 | 850.0 |
| Qwen3-1.7B | A10 | 8 | 128 | transformers | 0.0315 | 29.07 | 275.0 |
| Qwen3-1.7B | A10 | 16 | 256 | lite_llama | 0.0270 | 9.77 | 1626.2 |
| Qwen3-1.7B | A10 | 16 | 256 | transformers | 0.0342 | 29.68 | 538.8 |
| Qwen3-MoE-Tiny | A10 | 8 | 128 | lite_llama | 0.0059 | 0.93 | 8281.3 |
| Qwen3-MoE-Tiny | A10 | 8 | 128 | transformers | 0.0063 | 3.90 | 2043.3 |
| Qwen3-MoE-Tiny | A10 | 16 | 256 | lite_llama | 0.0068 | 0.98 | 15934.9 |
| Qwen3-MoE-Tiny | A10 | 16 | 256 | transformers | 0.0071 | 4.51 | 3540.1 |
| Llama-3.2-3B-Instruct | A10 | 8 | 128 | lite_llama | 0.0254 | 15.41 | 516.4 |
| Llama-3.2-3B-Instruct | A10 | 8 | 128 | transformers | 0.0309 | 25.33 | 315.3 |
| Llama-3.2-3B-Instruct | A10 | 16 | 256 | lite_llama | 0.0514 | 15.96 | 994.1 |
| Llama-3.2-3B-Instruct | A10 | 16 | 256 | transformers | 0.0557 | 27.74 | 574.6 |
| Qwen3-8B | A10 | 8 | 128 | lite_llama | 0.0561 | 36.79 | 216.6 |
| Meta-Llama-3.1-8B-Instruct | A10 | 8 | 128 | lite_llama | 0.0581 | 35.30 | 225.5 |
| Qwen3-14B-AWQ | A10 | 8 | 128 | lite_llama | 0.1499 | 43.49 | 180.5 |
| Qwen3-14B-AWQ | A10 | 16 | 256 | lite_llama | 0.2808 | 45.01 | 348.4 |

结论（2026-08-31 重测，torch 2.11.0+cu129 / transformers 5.8.0 / Python 3.12，覆盖受支持的全部纯文本架构）：lite_llama 的 **decode 全面更快** —— TPOT 比值（transformers / lite_llama）在 **1.6×～7.1×** 之间，模型越大比值越低（0.6B 档 ~6-7×，3B 档收敛到 ~1.6-1.9×：模型越大 decode 越偏 compute-bound，两端都吃满算力）；聚合吞吐 TGS 同步放大。每组配置两端输出 token 数一致，工作量对等。**TTFT** 绝对值小（6～50 ms），lite_llama 普遍略优但 run-to-run 抖动明显，不逐行解读。原始日志见 `benchmark_logs/bench_*.json`（22 份，每份含完整 config）。

> 本节表中未出现的组合：**8B 级 b16 档**的 KV 预算（16×2048 token ≈ 4.8 GiB + 16 GiB 权重）超出 22 GiB，只测 b8；**8B 级与 14B-AWQ 的 transformers 侧**分别因 transformers 5.8 的 `caching_allocator_warmup` 需要约双倍模型显存、AWQ 反量化需要 gptqmodel/autoawq（未安装）而无法在本机完成，标为 lite_llama 单侧。

复现：

```bash
# 全量复现（上表 12 个模型 × 两档配置，含各量化路径的差异化参数）：
PYTHON=/home/honggao/projects/.venv/bin/python ./benchmarks/run_benchmark_suite.sh
# 单模型：
python examples/benchmark.py --model my_weight/Qwen2.5-1.5B-Instruct \
    --batch-size 8 --gen-len 128 --iters 2      # 结果打印并存入 benchmark_logs/*.json
# FP8 checkpoint 的 transformers 基线：--hf-dtype auto（无原生 fp8 的卡上自动 dequant 为 bf16）
# transformers 无法加载的量化（AWQ 需 gptqmodel/autoawq）：--engine lite_llama 单侧
# 8B 级：--max-gpu-num-blocks 16384 收缩 KV 池（profile 默认值留给 graph 捕获的空间不足）
```

lite_llama 流式输出实录（Qwen2.5-3B，仅演示效果，非并排对比录制）：

![lite_llama 流式输出](images/qwen2.5-3b-output.gif)

### eager vs CUDA graph（benchmarks/bench_e2e.py）

batch 8、greedy、`max_gen_len=256`、A10 22 GiB、torch 2.11.0+cu129 / triton 3.6.0 / Python 3.12，`--mode both` 同时测 eager 与 CUDA graph。一次覆盖全部四种受支持架构与三条优化路径：

| 模型 | 架构 / 优化 | TTFT (ms) | TPOT eager (ms) | TPOT graph (ms) | graph 加速 | TPS (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen1.5-0.5B | qwen2 / bf16 | 17.6 | 18.07 | 3.39 | 5.3x | 2319.9 |
| Qwen2.5-1.5B | qwen2 / bf16 | 20.9 | 20.47 | 8.54 | 2.4x | 931.7 |
| Qwen2.5-3B | qwen2 / bf16 | 26.4 | 25.85 | 16.68 | 1.55x | 478.4 |
| Qwen3-0.6B | qwen3 / bf16 | 23.0 | 24.00 | 4.61 | 5.2x | 1706.7 |
| Qwen3-0.6B-FP8 | qwen3 / fp8 | 26.2 | 25.94 | 4.49 | 5.8x | 1746.9 |
| Qwen3-1.7B | qwen3 / bf16 | 25.1 | 25.00 | 9.63 | 2.6x | 825.3 |
| Qwen3-8B | qwen3 / bf16 | 57.2 | 38.66 | 37.33 | 1.04x | 213.9 |
| Qwen3-14B-AWQ | qwen3 / w4a16 | 154.7 | 44.49 | 43.90 | 1.01x | 180.4 |
| Qwen3-MoE-Tiny | qwen3_moe / fused MoE | 4.7 | 3.42 | 1.19 | 2.9x | 6620.0 |
| Llama-3.2-3B-Instruct | llama / bf16 | 24.7 | 20.99 | 15.80 | 1.33x | 505.1 |

统计口径与观察：

- 三项指标：TTFT 取 graph 档从提交到首 token 可见的墙钟（prefill 主导），TPOT 取首 token 之后所有步间隔的均值，TPS 为 batch 聚合吞吐（`gen_tokens / 总时间`）；每档先 warmup 两轮再计时。
- graph 加速随规模衰减是结构性的：≤1.7B 的 decode 步只有几毫秒，kernel launch 开销占比高，重放拿到 2.4-5.8x；8B/14B 算术时间主导，加速收敛到 1.01-1.04x。Qwen3-8B 的 KV 池收缩到 16384 token 以进 22 GiB（`--max-gpu-num-blocks 16384`）。
- FP8 与 bf16 的 0.6B TPOT 几乎相同（4.49 vs 4.61 ms）：小模型 decode 是 launch-bound，权重带宽减半的收益体现不出来，FP8 的收益要到大模型才显形。

复现（套件脚本一次跑全矩阵，产出同口径 JSON；解释器要有能跑 CUDA 的 torch 构建——项目 `.venv` 若装了比驱动新的 cu 版本，脚本预检会拦下并提示换 `PYTHON=`）：

```bash
./benchmarks/run_e2e_suite.sh /tmp/e2e                        # 全部模型
PYTHON=/home/honggao/projects/.venv/bin/python ./benchmarks/run_e2e_suite.sh
.venv/bin/python benchmarks/bench_e2e.py --model-dir my_weight/Qwen3-14B-AWQ \
    --greedy --mode both --json out.json                     # 单模型
```

## 三 性能优化历史记录

### 迭代式优化记录

输入提示词：

```bash
prompts: List[str] = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    """A brief message congratulating the team on the launch:

    Hi everyone,
    
    I just """,
    # Few shot prompt (providing a few examples before asking model to complete more);
    "Roosevelt was the first president of the United States, he has",
]
```

1，针对 decode 阶段使用 cuda graph 优化后，单次 decode 阶段时间为 `8.2402` ms，使用之前为 `17.2241` ms，性能提升 2x 倍，这个结果跟 vllm 应用 cuda graph 后的性能提升倍数几乎一致。

```bash
INFO: After apply cuda graph, Decode inference time: 8.2402 ms
INFO: Before apply cuda graph, Decode inference time: 17.2241 ms
```

2，在前面的基础上，继续优化，使用 flashattention 替代原有的标准 attention。

> flashattention1 对训练模型帮助更大，在提示词很短时，其速度提升效果有限。推理时的 decode 阶段应该用 flash-decoding。

```bash
INFO: input tokens shape is  torch.Size([8, 115])
# 使用 flashattention 前
INFO:lite_llama.generate:Batch inference time: 3152.0476 ms
INFO:lite_llama.generate:Tokens per second: 97.71 tokens/s
# 使用 flashattention1 后
INFO:lite_llama.generate:Batch inference time: 2681.3823 ms
INFO:lite_llama.generate:Tokens per second: 114.87 tokens/s
```

3，继续优化, 将 `flashattention` 升级到 `flashattention2`, 减少一定计算量。

```bash
INFO:lite_llama.generate:Batch inference time: 2103.0737 ms
INFO:lite_llama.generate:Tokens per second: 146.45 tokens/s
```

4，再次优化，decode 阶段的推理使用 `flashdecoding`，提升 decode 阶段的 attention 计算并行度，充分发挥 GPU 算力。

```bash
INFO:lite_llama.generate:Decode stage Batch inference time: 1641.4178 ms
INFO:lite_llama.generate:Decode stage tokens per second : 187.64 tokens/s
```

5，继续再次优化，支持 kv cache 高效的动态管理（类似 tokenattention），解决了 kv cache 显存浪费和分配低效的问题。

```bash
INFO:lite_llama.generate:Decode stage Batch inference time: 1413.9111 ms
INFO:lite_llama.generate:Decode stage tokens per second : 217.84 tokens/s
```

6，一个简单的优化, 使用 `GQA_KV_heads_index` 替代 `repeat_kv` 函数。

7，一个常见且简单的优化, kv 线性层融合。

8，一个常用的优化，算子融合：残差连接的 skip 操作和 `rmsnorm` 算子融合，形成新的 `skip_rmsnorm` 算子。

9，重构并优化 `MHA` 模块，优化 `context_attention` 和 `token_attention` 内核支持 `Nopad attention` 和 `kv cache` 动态分配和管理：

- token_attention 支持直接传入 kv_cache 索引和序列实际长度 seq_len, 减少了 kv cache 在 `MHA` 模块中的 `concat` 和 `view` 操作，并实现了 `Nopad` token_attention。
- 将每次 prefill/decode 过程动态分配实际 prompts 长度的 kv cache 索引个数，而不是在模型推理之前一次性分配连续的 `(max(promptes_len) + max_gen_len) * batch_size` 个 tokens 的 kv cache 空间。

10，引擎侧消除 decode 循环中的 GPU→CPU 同步。原实现每步执行 `bool(hit_stop[i])×batch + all() + .item()` 共 9 次同步，另外 `decode_alloc_kv_cache` 每步做一次 40960 元素的 `torch.nonzero` + 2 次 `.item()`。CPU 一旦读 GPU 张量就必须等前面的 kernel 全部完成，launch 流水线被反复清空，这是 eager decode 里 TPOT 比 GPU 真实计算时间高一倍的主因。

- 新增 [`StopCriteria`](lite_llama/engine/stop_criteria.py)：结束标志常驻 GPU，用词表大小的布尔查找表判 EOS，全批一次张量运算完成，每 8 步才做一次 `all()` 轮询。
- 新增 [`_DecodeSession`](lite_llama/engine/llm_engine.py)：把 per-request 状态封装出来，主机侧仅在轮询边界读回一次采样结果。
- 给 [`KVCacheManager`](lite_llama/executor/kv_cache_manager.py) 加 bump 分配器：`generate()` 内 KV 缓存是纯追加分配，只用一个 int 游标记录写入位置，任一部分释放后自动回退到原全表搜索。

11，消除 O(n²) 流式解码。原实现每步都 `tokenizer.decode(tokens[prompt_len:cur])` 整段解码再与已输出内容做差，256 token 时累计 ~0.8 ms/step。新增 [`IncrementalDetokenizer`](lite_llama/engine/detokenizer.py) 用滑动窗口只解码 `[prefix_offset:]` 和 `[prefix_offset:read_offset]` 两小段，仍能正确处理 SentencePiece 的前导空格（`▁` 需要上下文）与跨 token 的多字节 UTF-8（结尾遇 `\ufffd` 时先攥住不吐）。摊销后每步常数代价。

12，向量化 repetition penalty。原实现按 batch 逐行 `torch.unique + clone + index_put`，约 4·batch 次 kernel 启动 + 一次全量 clone。新增 [`GeneratedSpan`](lite_llama/engine/sampler.py) 数据类和 padding-safe scatter：把 batch 已生成 token 一次 scatter 到 `[batch, vocab+1]` 布尔表的哨兵列（避免 padding 位置用 False 覆盖真实命中），再两次 `torch.where`。共 3 个 kernel、无 clone。

以上 10-12 项主机侧优化的前后对比（NVIDIA A10 23 GB / Qwen2.5-0.5B / batch=8 / max_gen_len=256 / greedy；TTFT 为首 token 墙钟时间、TPOT 为稳态每 token 延迟、TPS 为 batch 聚合吞吐，口径对齐 vLLM）：

| 配置                    | TTFT (ms) | TPOT (ms) | TPS (token/s) |
|------------------------|-----------|-----------|---------------|
| eager（优化前）          | 15.0      | 15.04     | 532           |
| eager（10-12 项优化后）  | 13.7      | 13.55     | 590           |

eager 路径的 GPU 计算本身没有变化，TPOT 从 15.04 ms 降到 13.55 ms 几乎全部来自主机侧开销的削减：每步 GPU→CPU 同步从 9 次降到 0.125 次、流式解码从 O(n²) 降到摊销 O(1)、penalty 从约 4·batch 次 kernel 降到 3 次。

13，[`TextGenerator`](lite_llama/engine/generator.py) 默认开启 CUDA Graph 捕获（多模态显式关闭）。KV 显存预算里为 graph 捕获预留 workspace（[`estimate_capture_workspace`](lite_llama/executor/cuda_graph.py)），并把捕获 batch 上界钳到请求表容量，修复 `0.9 gpu-util + graph capture` 场景下的 OOM。

与 HuggingFace transformers 的对比（NVIDIA A10 23 GB / **Qwen2.5-1.5B-Instruct** fp16 / batch=8 / max_gen_len=256 / greedy，指标口径同上）。HF 侧由 [`bench_hf_baseline.py`](benchmarks/bench_hf_baseline.py) 测量：左 padding、不套 chat template、`min_new_tokens` 强制跑满 256 步、sdpa attention：

| 引擎                             | TTFT (ms) | TPOT (ms) | TPS (token/s) | 生成总时间 (s) |
|----------------------------------|-----------|-----------|---------------|---------------|
| transformers 5.15（sdpa）        | 27.6      | 24.24     | 330           | 6.21          |
| lite_llama eager                 | 15.8      | 16.86     | 475           | 4.31          |
| **lite_llama graph（10-13 项）** | **16.4**  | **9.05**  | **881**       | **2.32**      |

相对 transformers：eager 路径 TPS 1.44x、TPOT 1.44x、TTFT 1.75x；graph 路径 TPS 2.67x、TPOT 2.68x。1.5B fp16 权重约 3.09 GB，A10 带宽下限约 5.2 ms/token（3.09 GB / 600 GB/s）；graph 模式 TPOT 9.05 ms 中主机侧开销已被 CUDA Graph 消除，与下限之间剩余的约 3.9 ms 是 kernel 级优化空间（decode attention、小 GEMM 效率）。

> 0.5B 上的历史对照（同口径）：graph 优化前 TPOT 5.54 ms / TPS 1433，10-13 项后 TPOT 3.77 ms / TPS 2096（3.94x），已逼近 0.5B fp16 权重带宽下限 3.46 ms（1260 MB / 600 GB/s）。

精度验证（[`scripts/golden_tokens.py`](scripts/golden_tokens.py)）：8 个 greedy 用例覆盖单条 / 等长 batch / 混合长度 batch × 有无 repetition penalty，优化前后逐字节完全一致。

14，配置 / 权重加载 / 模型注册三个模块重构，参照 vLLM 的分层：

- **配置**。删除按架构手写的 `model_config.py` dataclass 与它的 HF 字段别名表，schema / 解析 / 默认值全部交给 `AutoConfig`（[`models/config.py`](lite_llama/models/config.py)）。`ModelConfig` 只补两件 HF config 给不了的东西：运行时旋钮 `max_seq_len`，以及 `num_kv_heads` / `head_dim` / `rope_theta` 的归一化。复用的是配置体系，**不引入** `modeling_*.py`——文本模型仍全跑自己的 Triton 内核。这也修好了一个真 bug：transformers 5.x 把 `rope_theta` / `mrope_section` 收进 `rope_parameters`，旧别名表不认识，Qwen3-VL 的 `mrope_section` 会静默丢失并退化成普通 RoPE。
- **权重加载**。删除 `tools/convert_weights.py` 与 `lite-llama-convert` 入口，不再需要离线产物。流程与 vLLM 一致：meta 设备上构造空模型 → 就地分配 fp16 参数 → 从 safetensors 流式 `copy_` 到位。只做名字 / 结构重映射（[`models/weights.py`](lite_llama/models/weights.py)）：K/V 写进 `kv_proj_weight` 的上下两半，MoE 逐专家矩阵堆叠进 `gate_up_proj` / `down_proj`，FP8 block 量化在目标设备上反量化。拷贝循环按**元素个数**统计覆盖率，漏写 / 半写 / 重写都会报错——这是 `strict=True` 的 `load_state_dict` 看不到的（fused 参数只写一半仍然“存在”）。
- **注册**。[`registry.py`](lite_llama/models/registry.py) 从 181 行降到 81 行：每个条目只剩 `model_type -> (实现类路径, 是否多模态)`，每架构一个 config loader 工厂、`load_config` / `build_model` / `read_model_type` 全部取消。新增一个模型 = 一行表项 + 一个类。

加载耗时（A10 / 页缓存已预热 / 取 3 次最小值）：

| 模型                    | 旧：转换一次 + 加载 `.pth` | 新：直读 safetensors | 硬盘占用变化 |
|------------------------|--------------------------|------------------------|-------------|
| Qwen2.5-0.5B           | 5.75 s + 0.26 s          | **0.22 s**             | −988 MB     |
| Qwen2.5-1.5B-Instruct  | — + 0.61 s               | **0.60 s**             | −3.09 GB    |

稳态推理吞吐不变（同机器各跑 3 次，Qwen2.5-0.5B / batch=8 / greedy）：graph 路径 TPS 2111 / 2119 / 2120（重构前）vs 2117 / 2104 / 2116（重构后），eager 路径两边同处 565–606 的噪声带内。真正省下的是部署路径：0.5B 从“转换 6.0 s + 多占 988 MB”变成“直接 0.22 s 加载”，30B-A3B-FP8 则不再需要那份 61 GB 的 `.pth` 副本。

权重加载的精度验证分三层：

- [`tests/models/test_weight_mapping.py`](tests/models/test_weight_mapping.py)：逐个 key 形状的映射单测 + 覆盖率记账（漏 key / 半写 fused / shape 不对 / 映射到不存在的参数，均必须报错）。
- [`tests/models/test_weight_parity.py`](tests/models/test_weight_parity.py)：6 个架构各随机初始化一个 tiny HF 模型存成真 safetensors，跑完整加载路径后逐参数逐元素对比——k/v 互换、专家下标错位、gate/up 颠倒这些“形状全对但值错位”的 bug 只有这层能抓。
- [`tests/models/test_checkpoint_index.py`](tests/models/test_checkpoint_index.py)：拿真实发布 checkpoint 的 `model.safetensors.index.json`（本地验证过 llava-1.5-7b-hf 686 key、Qwen3-VL-4B 713 key、Qwen3-30B-A3B-FP8 37491 key），在 meta 设备上不读一字节权重就验证“每个 key 都有参数接 / 每个参数都有 key 写”。

顺手抓出的两个旧 bug，都改变了输出，所以单独记一笔：

1. **Qwen3-VL 的 RoPE base 错了 500 倍**。transformers 5.x 只在 `rope_parameters` 里写 `rope_theta`，而多模态路径的旧配置是 `LlamaConfig.from_dict(config.text_config.to_dict())`，读不到顶层 `rope_theta` 就退到 dataclass 默认值 **10000.0**，而 checkpoint 声明的是 **5,000,000**。在 201 token 的纯文本 prompt 上与 HF `Qwen3VLForConditionalGeneration` 对照 logits：

   | rope_theta | 与 HF 的平均 cosine | 最小 cosine | top-1 一致率 |
   |------------|---------------------|--------------|--------------|
   | 10000（旧，默认值） | 0.928 | **−0.195** | 99.50% |
   | 5e6（新，读配置） | **0.99973** | **0.982** | **100%** |

   最小 cosine 为负意味着部分位置的 logits 向量完全反了方向。回归用例：[`test_nested_rope_theta_is_not_lost`](tests/config/test_config.py)、[`test_qwen3_vl_language_model_gets_mrope_and_the_right_base`](tests/models/test_weight_parity.py)。
2. **`inv_freq` 被降成 fp16**。旧 loader 最后一句 `model.half()` 连非持久化 buffer 一起转了，RoPE 的 `inv_freq` 静默变成 fp16。它以 `position × inv_freq` 参与相位计算，误差随位置线性放大：Qwen2.5-0.5B 在 position 1024 处相位误差 0.086 rad，LLaVA-1.5（theta=1e4）在 4096 处 0.99 rad。新 loader 不再动 buffer，`inv_freq` 保持 fp32。

因为这两项修正，golden 基线已重录。作为反向验证：强行把 `inv_freq` 改回 fp16 后，新加载路径在 Qwen2.5-0.5B 上与旧 golden 基线 8 个用例逐字节完全一致，说明三个模块的重构本身是 bit-exact 的，输出差异全部来自上面两个修正。重录 diff 里可以看到多条原本陷入重复循环的输出变成正常叙述。

复现命令：

```bash
# lite_llama 端到端指标（默认 my_weight/Qwen2.5-0.5B，--model-dir 切换模型）
python benchmarks/bench_e2e.py --greedy --max-gen-len 256 --batch 8 --model-dir my_weight/Qwen2.5-1.5B-Instruct

# HF transformers 基线（同 prompts、同指标口径）
python benchmarks/bench_hf_baseline.py --model-dir my_weight/Qwen2.5-1.5B-Instruct --max-gen-len 256 --batch 8

# 精度对照
python scripts/golden_tokens.py --save /tmp/golden.json          # 优化前录制
python scripts/golden_tokens.py --check /tmp/golden.json         # 优化后比对
python scripts/golden_tokens.py --check /tmp/golden.json --cuda-graph
```

### 历史吞吐对比总表（旧脚本，仅供参考）

> ⚠️ 数据来源：本表数字来自本文档下方各模型章节的**历史记录**（由仓库作者早前用
> 旧版 `benchmark.py` 在 3090 上跑出），**并非本次实测**。旧脚本存在方法学问题：
> transformers 被强制忽略 EOS（`eos_token_id=None`）跑满长度，而 lite_llama 会提前
> 停止，两端工作量并不一致；且仅有单次运行、只统计吞吐、无 TTFT/TPOT。因此这些倍数
> 仅作趋势参考，请以上方实测表为准。

| 模型 | GPU | batch_size | seq_len¹ | max_gen_len | lite_llama (tokens/s) | transformers (tokens/s) | 吞吐加速比 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct | 3090 的 1/4 卡 (B1.small) | 16 | 变长 | 1900 | 411.04 | 104.70 | 3.93× |
| Llama-3.2-3B-Instruct | 3090 整卡 (B1.big) | 8 | 变长 | 1900 | 458.97 | 134.37 | 3.42× |
| Llama-3.2-3B-Instruct | 3090 整卡 (B1.big) | 12 | 变长 | 1900 | 730.45 | 183.95 | 3.97× |
| Qwen2.5-3B-Instruct | 未标注 | 2 | 变长 | 2000 | 98.71 | 69.83 | 1.41× |
| Qwen2.5-3B-Instruct | 未标注 | 4 | 变长 | 256 | 182.28 | 133.33 | 1.37× |
| Qwen2.5-3B-Instruct | 未标注 | 12 | 变长 | 1900 | 581.20 | 172.19 | 3.38× |
| Qwen2.5-3B-Instruct | 未标注 | 16 | 变长 | 512 | 724.38 | 504.73 | 1.44× |
| Qwen2.5-3B-Instruct | 未标注 | 16 | 变长 | 1900 | 735.73 | 215.62 | 3.41× |

> ¹ 这些基准使用的是「多条不同长度提示词」组成的 batch，未固定或记录单一 prompt 长度，
> 故 seq_len 记为「变长」；Qwen2.5-3B 章节未标注 GPU 型号。

#### Llama-3.2-1B-Instruct 性能测试

趋动云 `B1.small` 等同于 `3090` 的 `1/4` 之一卡的硬件测试环境。运行性能测试对比 `python benchmark.py`，lite_llama 的运行速度最高是 transformers 的 `4x` 倍。

batch_size = 16 的提示词：

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
    "A Complete Introduction to the History of the American Civil War",
    "Python is a good programming language, how tolearn it?",
    "Please introduce llama model architecture and give implement cuda code."
    "Please introduce Qwen2.5 model structure and give cuda implement code."
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 67.8760 s
Transformers inference time: 131.8708 s
lite_llama throughput: 411.04 tokens/s
Transformers throughput: 104.70 tokens/s
lite_llama per token latency: 2.432831 ms/token
Transformers per token latency: 9.551007 ms/token
```

#### Llama-3.2-3B-Instruct 性能测试

/gemini/code/lite_llama/my_weight/Llama-3.2-1B-Instruct

趋动云 `B1.big` 等同于 `3090` 卡的硬件测试环境。运行性能测试对比 `python benchmark.py`，lite_llama 的运行速度最高是 transformers 的 `4x` 倍。

batch_size = 8 的提示词：

```bash
prompts: List[str] = [
        "I believe the meaning of life is to find happiness in the simple things. This is a very subjective and personal perspective, and it may vary from person to person. However, I believe that the simple things can bring a sense of joy and fulfillment to our lives.",
        "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
        "A Complete Introduction to the History of the American Civil War",
        "Roosevelt was the first president of the United States, he has a lot of information on the early history of the United States. He was born in 1883,",
        "How to learn c++, give me some code example.",
        "How to learn python, give me some code examples.",
        "How to learn llm, please introduce transformer architecture ",
        "How to learn cnn, please introduce resnet architecture and give code ",
    ]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 32.0826 s
Transformers inference time: 51.2225 s
lite_llama throughput: 458.97 tokens/s
Transformers throughput: 134.37 tokens/s
lite_llama per token latency: 2.178783 ms/token
Transformers per token latency: 7.441883 ms/token
```

batch_size = 12 的提示词：

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 31.3463 s
Transformers inference time: 69.1433 s
lite_llama throughput: 730.45 tokens/s
Transformers throughput: 183.95 tokens/s
lite_llama per token latency: 1.369015 ms/token
Transformers per token latency: 5.436221 ms/token
```

#### Qwen2.5-3B-Instruct 性能测试

`batch_size = 2` 时的提示词

```bash
prompts: List[str] = [
        "How to learn cnn, please introduce resnet architecture and give code ",
        "How to learn cuda programming, give me some code example.",
    ]
```

`max_gen_len = 2000` 时, benchmark 性能测试运行结果:
```bash
lite_llama inference time: 34.9293 s
Transformers inference time: 31.6787 s
lite_llama throughput: 98.71 tokens/s
Transformers throughput: 69.83 tokens/s
lite_llama per token latency: 10.130305 ms/token
Transformers per token latency: 14.321302 ms/token
```

`batch_size = 4` 时的提示词

```bash
    prompts: List[str] = [
        "How to learn cnn, please introduce resnet architecture and give code.",
        "How to learn cuda programming, give me some code example.",
        "How to learn rust, give me some code examples.",
        "How to learn java, give me some code example.",
    ]
```

`max_gen_len = 256` 时, benchmark 性能测试运行结果:

```bash
lite_llama inference time: 5.5739 s
Transformers inference time: 7.6803 s
lite_llama throughput: 182.28 tokens/s
Transformers throughput: 133.33 tokens/s
lite_llama per token latency: 5.486118 ms/token
Transformers per token latency: 7.500309 ms/token
```

`batch_size = 12` 时的提示词

```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
]
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 26.8804 s
Transformers inference time: 63.2376 s
lite_llama throughput: 581.20 tokens/s
Transformers throughput: 172.19 tokens/s
lite_llama per token latency: 1.720564 ms/token
Transformers per token latency: 5.807474 ms/token
```

`batch_size = 16` 时的提示词
```bash
prompts: List[str] = [
    "I believe the meaning of life is to find happiness in the simple things. but how to achieve the meaning of life?",
    "VGG is a very important cnn backbone, please introduce vgg architecture and give implement code ",
    "Can you introduce the History of the American Civil War. ",
    "who is the first president of the United States and what's his life story?",
    "How to learn c++, give me some code example.",
    "How to learn python, give me some code examples.",
    "How to learn llm, please introduce transformer architecture ",
    "How to learn cnn, please introduce resnet architecture and give code ",
    "How to learn cuda programming, give me some code example.",
    "How to learn rust, give me some code examples.",
    "How to learn java, give me some code example.",
    "How to learn linux c, give me some code examples.",
    "A Complete Introduction to the History of the American Civil War",
    "Python is a good programming language, how tolearn it?",
    "Please introduce llama model architecture and give implement cuda code."
    "Please introduce Qwen2.5 model structure and give cuda implement code."
]
```

`max_gen_len = 512` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 11.3434 s
Transformers inference time: 14.9981 s
lite_llama throughput: 724.38 tokens/s
Transformers throughput: 504.73 tokens/s
lite_llama per token latency: 1.380484 ms/token
Transformers per token latency: 1.981256 ms/token
```

`max_gen_len = 1900` 时，benchmark 性能测试运行结果:

```bash
lite_llama inference time: 38.4323 s
Transformers inference time: 70.3268 s
lite_llama inference output tokens number: 28276
Transformers inference output tokens number: 15164
lite_llama throughput: 735.73 tokens/s
Transformers throughput: 215.62 tokens/s
lite_llama per token latency: 1.359186 ms/token
Transformers per token latency: 4.637745 ms/token
```
