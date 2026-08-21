# 精度评估（tests/evals）

对齐 vLLM `tests/evals/` 的精度回归套件：数据集 → 提示词 → 生成 → 打分 →
与 `configs/*.yaml` 里声明的阈值比对。区别只在执行路径 —— vLLM 起一个 server 走
OpenAI HTTP，lite_llama 没有 server，所以直接驱动离线的 `LLM.generate()`。

目前实现了 GSM8K（小学数学应用题，5-shot，取答案里最后一个整数做精确匹配）。

## 跑起来

```bash
# pytest：按阈值判定通过/失败，默认只跑 models-small.txt 里的配置
make test-eval
make test-eval EVAL_CONFIGS=models-all.txt

pytest -s -v tests/evals --config-list-file=models-all.txt

# 独立脚本：只出数，不判定
python -m tests.evals.gsm8k --model-dir my_weight/Qwen2.5-0.5B --num-questions 200
python -m tests.evals.gsm8k --model-dir my_weight/Qwen2.5-1.5B-Instruct \
    --num-questions 200 --batch-size 16 --chat-template
```

实测结果与复现命令见 [`docs/eval_models.md`](../../docs/eval_models.md)。

## 配置格式

```yaml
model_dir: my_weight/Qwen2.5-0.5B  # 相对仓库根目录，或绝对路径
num_questions: 200                 # 测试题数，上限 1319（GSM8K test 全量）
num_fewshot: 5                     # few-shot 示例数，取自 train split
max_gen_len: 256                   # 每题解码预算
batch_size: 32                     # 一次 generate() 的题数
max_seq_len: 2048                  # 上下文上限，须装得下 prompt + max_gen_len
chat_template: false               # true 时把 prompt 包成一轮 user 消息
accuracy_threshold: 0.35           # 实测基线
tolerance: 0.05                    # 允许的下滑幅度
max_invalid_rate: 0.05             # 解析不出答案的比例上限
```

`configs/models-*.txt` 每行一个配置文件名，`#` 开头为注释。配置里点名的
checkpoint 不存在时该用例自己 skip，所以 `models-all.txt` 在任何机器上都能跑。

## 模块划分

| 文件 | 职责 |
| --- | --- |
| `dataset.py` | 下载并缓存 benchmark 数据（JSONL），不可达时抛 `DatasetUnavailable` |
| `runner.py` | 离线执行层：显式分批、stop 截断、KV cache 定量、采样参数 |
| `gsm8k.py` | GSM8K 本体：提示词构造、答案抽取、打分，附带 CLI |
| `test_gsm8k_correctness.py` | GPU 层：按配置跑真模型，比对阈值 |
| `test_gsm8k_scoring.py` | CPU 层：纯函数单测，不联网、不用 GPU、不用权重 |

## 三处与 vLLM 不同的适配

**stop 是事后截断的。** `SamplingParams` 没有 `stop` 字段，序列只在 EOS、重复检测
或 `max_gen_len` 处停下。few-shot 场景里 base 模型答完必然接着编下一道题，此时
"最后一个数字"会变成下一题的答案 —— 所以 `runner.truncate_at_stop()` 在文本层把
completion 切在第一个 `Question` 处。打分结果与 server 侧 stop 完全一致，差的只是
白跑的解码步数，这也是 `max_gen_len` 要卡住的原因。

**KV cache 是算出来的，不是 profile 出来的。** 引擎默认按空闲显存的 90% 分配 KV
cache —— 对 server 是对的，对 benchmark 是灾难：prefill 阶段 `lm_head` 要算
`batch × prompt_len × vocab` 的 logits，batch 32、15 万词表下就是 7 GB，profile 完
的显存根本不够它落地，prefill 直接 OOM。评估侧一批最多只可能用到
`batch_size × max_seq_len` 行（一 token 一行），所以 `runner.kv_cache_tokens()`
按这个精确上界给，剩下的显存留给 logits。顺带把跑分变成了跨机器可复现的 ——
profile 出来的大小取决于当时显卡上还驻留着什么。

**采样默认值要关掉。** lite_llama 的 `SamplingParams` 默认
`repetition_penalty=1.1`、`stop_on_repeat=True`，这两个都是为交互式聊天准备的
（小模型容易进复读循环）。评估要的是原始 argmax，两个都开着测的就不是模型本身而是
解码策略，所以 `runner.greedy_params()` 统一关掉。

## 数据集缓存

首次运行会把 GSM8K 的 `train.jsonl` / `test.jsonl` 下到
`~/.cache/lite_llama/evals/gsm8k/`。两个环境变量可以改这个行为：

- `LITE_LLAMA_EVAL_DATA_DIR`：缓存目录；
- `LITE_LLAMA_EVAL_BASE_URL`：下载源，用于换镜像或离线机器自建服务。

离线且没有预置缓存时，GPU 层用例 skip 而不是 fail —— 一台连不上网的 CI 机器该报
"没有数据集"，不该报"模型精度掉了"。

## 新增一个 benchmark

`dataset.py` 加一个 loader，仿照 `gsm8k.py` 写 `build_prompts` / `score` /
`evaluate_*`（保持纯函数，方便 CPU 层单测），执行层直接复用 `runner.py`。
