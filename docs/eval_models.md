# 精度评估

`tests/evals/` 是对齐 vLLM `tests/evals/` 的精度回归套件：数据集 → 提示词 → 生成 → 打分 → 与 `configs/*.yaml` 声明的阈值比对。当前实现 GSM8K。

本文记录**实测结果**与复现方式。套件本身的设计与扩展方式见 [`tests/evals/README.md`](../tests/evals/README.md)。

## GSM8K 实测结果

评测口径与 vLLM 的 `tests/evals/gsm8k` 一致，两边数字可直接对比：

- **提示词**：train split 的前 `num_fewshot` 条作为示例，接测试题，统一 `Question: ... \nAnswer: ...` 格式；题目取 test split 的前 N 条（固定前缀，不随机采样）；
- **解码**：greedy（`temperature=0`），关闭 `repetition_penalty` 与 `stop_on_repeat`；
- **停止**：截断到下一个 `Question`；
- **判分**：取 completion 里**最后一个整数**，与参考答案 `####` 后的数字做精确匹配。

测试环境：NVIDIA A10（23 GB）× 1，torch 2.13.0+cu129 / triton 3.7.1 / transformers 5.15.0 / Python 3.13，fp16 权重，CUDA graph 开启。

| 模型 | 题数 | few-shot | chat 模板 | **准确率** | 无效率 | 耗时 (s) | q/s | 生成吞吐 (tok/s) |
| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B（base） | 200 | 5 | 否 | **35.00%** | 0.00% | 12.6 | 15.9 | 2137 |
| Qwen2.5-0.5B（base） | 1319（全量） | 5 | 否 | **35.94%** | 0.00% | 82.1 | 16.1 | 2251 |
| Qwen2.5-1.5B-Instruct | 200 | 5 | 是 | **63.00%** | 0.00% | 43.1 | 4.6 | 675 |
| Qwen2.5-1.5B-Instruct | 1319（全量） | 5 | 是 | **63.76%** | 0.00% | 277.3 | 4.8 | 690 |

`batch_size` 分别为 32（0.5B）与 16（1.5B），`max_gen_len=256`，`max_seq_len=2048`。

两个模型上 200 题子集与 1319 题全量的差都在 1 个点以内（35.00 vs 35.94、 63.00 vs 63.76），说明 200 题子集足以当日常回归的信号，全量留给需要参考数值的场合。

**无效率全程为 0**：每条 completion 都能解析出数字。这一列是判断"模型答错了"还是 "评测根本没看到答案"的分界 —— 如果它显著大于 0，上面的准确率就不再是模型的性质，而是 `max_gen_len` 或提示词格式的问题。

### 复测记录：v0.9 kernels 三层重构（2026-08）

kernels 目录重组为 ops/dispatcher/backend 三层后，在 torch 2.11.0+cu129 / triton 3.6.0 / Python 3.12 环境下复测：

- **Qwen2.5-1.5B-Instruct / 200 题：61.50%，无效率 0，通过**（阈值 0.63±0.05）；同一环境在重构前的 main 分支上复测得到**完全相同的 61.50%**——重构对生成结果零影响。与上表 63.00% 的 1.5 点差来自环境（历史数字测于 torch 2.13/triton 3.7/Python 3.13），在 200 题子集 ±3.4 点的统计噪声内。
- 同一分支上 **Qwen3-0.6B（bf16）与 Qwen3-0.6B-FP8 的 golden token parity 全部通过**：eager 与 CUDA graph 重放、与各自入库基线字节级一致（4 种 batch 布局 × repetition penalty 全组合），这是比分数更强的逐 token 证据。
- e2e 性能复测（10 个 checkpoint × eager/CUDA graph，覆盖 llama/qwen2/qwen3/qwen3_moe 四种架构与 FP8/AWQ/fused-MoE 优化路径）见 `docs/benchmark_models.md`「模型 e2e benchmark 汇总」章节的 eager vs CUDA graph 小节：graph 加速从 0.5B 的 5.3x 收敛到 14B 的 1.01x，launch-bound 到 compute-bound 的过渡与规模相符。

## 复现

```bash
# pytest：按阈值判定通过/失败
make test-eval                              # 默认 models-small.txt（0.5B / 200 题，约 20 s）
make test-eval EVAL_CONFIGS=models-all.txt  # 全部配置

# 独立脚本：只出数，不判定
python -m tests.evals.gsm8k --model-dir my_weight/Qwen2.5-0.5B \
    --num-questions 1319 --batch-size 32 --max-gen-len 256

python -m tests.evals.gsm8k --model-dir my_weight/Qwen2.5-1.5B-Instruct \
    --num-questions 1319 --batch-size 16 --max-gen-len 256 --chat-template

# 追加结果到 JSON lines
python -m tests.evals.gsm8k --model-dir my_weight/Qwen2.5-0.5B --save-results eval_runs.jsonl
```

首次运行会把 GSM8K 下载到 `~/.cache/lite_llama/evals/gsm8k/`（train 4.0 MB + test 736 KB）。`LITE_LLAMA_EVAL_DATA_DIR` 改缓存目录，`LITE_LLAMA_EVAL_BASE_URL` 换下载源。

greedy 解码是确定性的：同一 checkpoint 重复跑得到逐字节相同的输出，因此准确率可精确复现（上表 0.5B/200 题的 35.00% 在独立脚本与 pytest 两条路径下取到同一个值）。

## 敏感性验证

只有一个准确率数字说明不了它测的是什么。下面几组都在 Qwen2.5-0.5B / 200 题上跑，用来确认口径本身没有引入偏差。

| 变量 | 设置 | 准确率 | 无效率 | 说明 |
| --- | --- | ---: | ---: | --- |
| `max_gen_len` | 256 | 35.00% | 0.00% | 基线 |
| `max_gen_len` | 512 | 35.00% | 0.00% | **完全一致** —— 256 步没有截断任何一条推理链 |
| few-shot | 0 | 31.50% | 0.00% | base 模型没有示例也能按格式作答，但掉 3.5 个点 |
| few-shot | 5 | 35.00% | 0.00% | 基线 |
| few-shot | 8 | 33.50% | 0.00% | 再加示例不再有增益 |

`max_gen_len` 从 256 加到 512 准确率一字不差，配合 0% 的无效率，说明 256 的解码预算对 GSM8K 足够；这条是选 256 作默认值的依据，而不是拍脑袋。

chat 模板对 instruct 模型的影响单列（Qwen2.5-1.5B-Instruct / 200 题）：

| chat 模板 | 准确率 | 无效率 |
| :---: | ---: | ---: |
| 关 | 54.50% | 0.00% |
| 开 | **63.00%** | 0.00% |

差 8.5 个点。instruct 模型被微调的格式是模板而不是裸文本，所以配置里 `chat_template: true` 对它们不是可选项 —— 关掉测到的是"模型在陌生格式下的表现"。 base 模型（Qwen2.5-0.5B）没有对应的微调格式，必须保持关闭。

## 阈值与回归判定

每个配置声明一条实测基线，测试断言 `准确率 ≥ accuracy_threshold - tolerance`：

| 配置 | 模型 | 题数 | 阈值 | 容差 |
| --- | --- | ---: | ---: | ---: |
| `Qwen2.5-0.5B.yaml` | Qwen2.5-0.5B | 200 | 0.35 | 0.05 |
| `Qwen2.5-0.5B-full.yaml` | Qwen2.5-0.5B | 1319 | 0.36 | 0.03 |
| `Qwen2.5-1.5B-Instruct.yaml` | Qwen2.5-1.5B-Instruct | 200 | 0.63 | 0.05 |

用下界而不是相等：greedy 是确定性的，但 kernel 改动即便数值上没问题，也可能让若干道临界题翻面。容差吸收这部分抖动，超出就是真的回归。题数越多抖动越小，所以全量配置的容差收到 0.03。

另有 `max_invalid_rate` 单独断言无效率上限。两种失败模式要分开看：准确率低但无效率也低 = 模型算错了；无效率高 = 评测没拿到答案，此时准确率不含任何关于模型的信息。

## 未覆盖的模型

`my_weight/` 下其余 checkpoint 没有纳入：

- **Qwen3-0.6B**：目录里只有 `config.json`，没有 `*.safetensors`；
- **llava-1.5-7b-hf / Qwen3-VL-4B-Instruct**：多模态，GSM8K 是纯文本任务，需要另配视觉基准；
- **Qwen3-30B-A3B-Instruct-2507-FP8**：A10 的 23 GB 装不下。

配置里点名的 checkpoint 不存在时用例自己 skip 并说明原因，所以 `models-all.txt` 在任何机器上都能直接跑。

补齐权重后新增一个 YAML、把文件名写进 `models-all.txt` 即可，无需改测试代码。
