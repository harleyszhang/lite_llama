<div align="center">

# Litellama

**A light llama-like llm inference framework based on the triton kernel.**

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/harleyszhang/lite_llama/blob/main/README.md)
[![zh](https://img.shields.io/badge/lang-zh-yellow.svg)](https://github.com/harleyszhang/lite_llama/blob/main/README.zh.md)
![PyPI - Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)

<pre>
         ✅ Flash attention    ✅ Cuda Graph Optimize    ✅ Beginner friendly    ✅ Fused MoE
         ✅ W8A16 Quantization ✅ W4A16 (AWQ/GPTQ)      ✅ SmoothQuant W8A8     ✅ Tensor Parallel
         ✅ Continuous batching ✅ OpenAI API server     ✅ Online batch inference
</pre>

</div>

## Features

- Up to `4x` speedup over transformers, llama3 1B and 3B models.
- **Online batch inference with continuous batching**: requests join and leave a running
  batch, so an arrival never waits for the current generation to finish. On one A10 with
  Qwen2.5-1.5B-Instruct and requests arriving 250 ms apart, throughput goes from 93 to
  644 tok/s (`6.9x`) and mean latency from 19.1 s to 2.3 s (`8.3x`) — see
  [docs/continuous_batching.md](./docs/continuous_batching.md).
- **OpenAI-compatible server** (`lite-llama serve`): `/v1/completions` and
  `/v1/chat/completions`, streaming included, so the official `openai` client works
  unchanged — see [docs/online_serving.md](./docs/online_serving.md).
- Supports the latest `llama3`, `Qwen2.5`, `Qwen3`, `Llava1.5`, `Qwen3-vl`, `Qwen3-MoE` model inference, `top-p` sampling, streaming output.
- Supports GQA, decode stage support cuda graph optimization (with batch_size limitations).
- Supports `flashattention1`, `flashattention2`, `flashdecoding` (supports `NopadAttention`).
- Support efficient dynamic management of kv cache (`auto tokenattnetion`).
- Support fusion of operators, e.g. fusion of `*` and `silu` for element-by-element multiplication, k v linear layer fusion, fusion of `skip` and `rmsnorm`.
- Some custom operators such as `rmsnorm`, `rope`, `softmax`, `element-by-element-multiplication`, etc. are implemented using the efficient `triton` kernel.
- **Quantization**: W8A16 (fp8/int8), W4A16 (AWQ/GPTQ), SmoothQuant W8A8 — up to `1.7x` decode speedup.
- **Tensor Parallelism**: split a 30B MoE model across 2× A10 (24 GB) with one all-reduce per block.

## Setup and Installation

If you don't have a physical server, you can try using [virtal cloud remote server](https://growthdata.virtaicloud.com/t/hK).

Requires Python 3.10+, CUDA-capable PyTorch 2.4+ and Triton 3.0+.

```bash
uv pip install -e .           # runtime deps
uv pip install -e . --group dev
pre-commit install
```

## Quick start

### Get the weights

Point `--model-dir` at a HuggingFace checkpoint directory — the one holding `config.json` and `*.safetensors` — exactly as `modelscope download` leaves it.

There is no conversion step: `config.json` is parsed by `AutoConfig`, and the weights are streamed from the safetensors shards straight into the model, with the K/V projections fused and the MoE experts stacked on the way in (see `lite_llama/models/weights.py`).

```bash
modelscope download Qwen/Qwen2.5-0.5B         --local-dir my_weight/Qwen2.5-0.5B
modelscope download Qwen/Qwen3-0.6B           --local-dir my_weight/Qwen3-0.6B
modelscope download llava-modelscope/llava-1.5-7b-modelscope  --local-dir my_weight/llava-1.5-7b-modelscope
modelscope download Qwen/Qwen3-VL-4B-Instruct --local-dir my_weight/Qwen3-VL-4B-Instruct
modelscope download Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --local-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8
```

Legacy `pytorch_model*.bin` checkpoints still load; safetensors wins when both are present.

### Examples

Qwen3-VL model, single pictures inference:

```bash
cd /home/honggao/projects/open_source/lite_llama
python -m lite_llama.cli vl-chat \
    --model-dir my_weight/Qwen3-VL-4B-Instruct \
    --image docs/images/llava_test/dog.jpeg \
    --prompt "What animal is in this picture? Answer in one sentence." \
    --temperature 0.0 --max-gen-len 48
```

Qwen3-VL-4B-Instruct, Multi-image + Sampling mode:

```bash
# 多图 + 采样模式
python -m lite_llama.cli vl-chat \
    --model-dir my_weight/Qwen3-VL-4B-Instruct \
    --image docs/images/llava_test/dog.jpeg docs/images/llava_test/dog2.png \
    --prompt "Compare the animals in these pictures." \
    --temperature 0.7 --top-p 0.9

# 默认 prompt(Describe this image.)
python -m lite_llama.cli vl-chat \
    --model-dir my_weight/Qwen3-VL-4B-Instruct \
    --image docs/images/llava_test/extreme_ironing.jpg
```

### Text generation

```python
from lite_llama import TextGenerator, SamplingParams

gen = TextGenerator(checkpoints_dir="my_weight/Qwen2.5-0.5B")
params = SamplingParams(temperature=0.0, max_gen_len=64)
print(gen.generate(["The capital of France is"], params))
```

Streaming:

```python
for step in gen.stream(["The capital of France is"], params):
    print(step[0], end="", flush=True)
```

### Online batch inference

Six requests, three slots. Watch the slot column: when a request finishes, the slot it
held is decoding a queued request on the very next step.

![continuous batching](./docs/images/continuous_batching.gif)

Serve an OpenAI-compatible API:

```bash
pip install 'lite-llama[serve]'
lite-llama serve --model-dir my_weight/Qwen2.5-1.5B-Instruct --port 8000
```

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen2.5-1.5B-Instruct",
       "messages": [{"role": "user", "content": "Explain a GPU in one sentence."}],
       "max_tokens": 64}'
```

Or drive the engine directly — every prompt is an independent request, and each carries
its own sampling parameters:

```python
from lite_llama import ContinuousBatchingEngine, SamplingParams

engine = ContinuousBatchingEngine.from_pretrained(
    "my_weight/Qwen2.5-1.5B-Instruct", max_num_seqs=16
)
engine.add_request("Name the capital of Japan.", SamplingParams(max_gen_len=32))
engine.add_request("Write a haiku about rain.", SamplingParams(temperature=0.8, max_gen_len=64))

while engine.has_unfinished_requests():
    for request in engine.step():
        print(f"[{request.request_id}] {request.delta}", end="", flush=True)
```

Asynchronously, with concurrent coroutines sharing one batch:

```python
import asyncio
from lite_llama import AsyncLLMEngine, SamplingParams

async def main():
    async with AsyncLLMEngine.from_pretrained("my_weight/Qwen2.5-1.5B-Instruct") as engine:
        async def ask(prompt):
            async for chunk in engine.generate(prompt, SamplingParams(max_gen_len=64)):
                print(chunk.delta, end="", flush=True)
        await asyncio.gather(ask("Hello"), ask("Goodbye"))

asyncio.run(main())
```

### Image conditioned generation

```python
from PIL import Image
from lite_llama import VisionGenerator, SamplingParams

gen = VisionGenerator(checkpoints_dir="my_weight/llava-1.5-7b-modelscope")
img = Image.open("docs/images/llava_test/dog.jpeg").convert("RGB")
prompt = "USER: <image>\nDescribe the animal in one sentence. ASSISTANT:"
print(gen.generate(prompt, [img], SamplingParams(temperature=0.0, max_gen_len=48)))
```

### CLI

llava-1.5-7b-modelscope default inference:

```bash
export LITE_LLAMA_MODEL_DIR=my_weight/Qwen2.5-0.5B
lite-llama chat                              # interactive text chat
lite-llama serve --port 8000                 # OpenAI-compatible API server
lite-llama batch --show-stats                # a prompt set through the scheduler
lite-llama vl-chat --model-dir my_weight/llava-1.5-7b-modelscope \
                   --image docs/images/dog.jpeg \
                   --prompt "USER: <image>\nWhat animal is this? ASSISTANT:"
```

llava-1.5-7b-modelscope default inference:

```bash
python -m lite_llama.cli chat --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8
```

### Evaluation

After `cli.py` runs successfully, the terminal displays the interface as shown below, and you can enter your question in the terminal.

![cli](./docs/images/cli_stream.png)

After `generate.py` runs successfully, the terminal displays the interface as shown below, and you can enter your question in the terminal.

![generate](./docs/images/generate_stream.png)

After `cli_llava.py` runs successfully, the terminal displays the interface as shown below, enter your picture and prompt word in the terminal, and then enter.

![llava model streaming output](./docs/images/llava_output2.gif)

For performance test, after changing your model weight path, run `lite_llama/examples/benchmark.py` file directly, it will output the latency and throughput performance comparison between lite_llama and transformers libraries, the result of the first run is not very accurate, so we suggest you to take the second run as a reference. For example, for the Llama-3.2-3B model with `prompt_len = 25`, `batch_size = 12`, and `max_gen_len = 1900`, the result of benchmark:
```bash
lite_llama inference time: 31.3463 s
Transformers inference time: 69.1433 s
lite_llama throughput: 730.45 tokens/s
Transformers throughput: 183.95 tokens/s
lite_llama per token latency: 1.369015 ms/token
Transformers per token latency: 5.436221 ms/token
```

### Quantization & Tensor Parallelism

lite_llama supports multiple weight quantization schemes and multi-GPU tensor
parallelism. See [docs/quantization.md](docs/quantization.md) for the full guide.

**FP8 checkpoint** (auto-detected from `config.json`):

```bash
python -m lite_llama.cli chat --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8
```

**Tensor parallelism** (30B MoE on 2× A10):

```bash
python -m lite_llama.cli chat \
    --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --tensor-parallel-size 2
```

**Runtime int8 quantisation** (halve memory of any fp16 model):

```bash
python -m lite_llama.cli chat --model-dir my_weight/Qwen2.5-0.5B --quantization int8
```

**Vision-language models** (Qwen3-VL / LLaVA with TP + quantization):

```bash
# Qwen3-VL vision chat with TP=2
python -m lite_llama.cli vl-chat \
    --model-dir /data/shared/llm_weights/Qwen3-VL-4B-Instruct \
    --image photo.jpg --tensor-parallel-size 2

# LLaVA with INT8 quantisation + TP=2
python -m lite_llama.cli vl-chat \
    --model-dir /data/shared/llm_weights/llava-hf/llava-1.5-7b-hf \
    --image photo.jpg --quantization int8 --tensor-parallel-size 2
```

| Model | FP16 TP=1 | FP16 TP=2 | INT8 TP=1 | INT8 TP=2 |
|-------|-----------|-----------|-----------|----------|
| Qwen3-VL-4B | 9.66 GB | 6.35 GB/GPU | 6.93 GB | 5.34 GB/GPU |
| LLaVA-1.5-7B | 13.74 GB | 7.90 GB/GPU | 8.12 GB | 5.15 GB/GPU |

Quantized inference demo (Qwen3-30B-A3B-FP8, tensor parallel × 2):

![quantization tp demo](./docs/images/qwen2.5-3b-output.gif)

## Architecture

```text
lite_llama/
├── engine/
│   ├── llm.py               # LLM entry point
│   ├── llm_engine.py        # one-shot batch: a single prefill/decode loop
│   ├── continuous_engine.py # continuous batching: one step at a time
│   ├── scheduler.py         # who prefills, who decodes, who holds which slot
│   ├── async_engine.py      # asyncio front end over a worker thread
│   ├── generator.py         # TextGenerator / VisionGenerator facades
│   ├── sampler.py           # temperature / top-p / repetition penalty, per request
│   ├── detokenizer.py       # incremental text output
│   ├── stop_criteria.py     # EOS, repetition and length stopping
│   ├── multimodal.py        # processor call + mrope position ids
│   └── outputs.py           # RequestOutput / CompletionOutput
├── executor/
│   ├── model_runner.py      # owns the model, KV cache and per-step forward
│   ├── loader.py            # HF checkpoint -> fp16/8-bit parameters
│   ├── weight_utils.py      # safetensors reading, FP8 passthrough
│   ├── kv_cache_manager.py  # paged KV pool + memory profiler
│   ├── attention_metadata.py # per-step KV bookkeeping handed to the kernels
│   ├── slot_batch.py        # fixed-slot KV layout for continuous batching
│   └── cuda_graph.py        # decode graph capture, replay and batch padding
├── entrypoints/
│   ├── api_server.py        # OpenAI-compatible FastAPI app
│   └── protocol.py          # request/response schemas
├── kernels/                 # Triton kernels used by the models
│   ├── w8a16.py             # fp8/int8 weight-only GEMM
│   ├── w4a16.py             # AWQ/GPTQ int4 GEMM
│   ├── smoothquant.py       # W8A8 dynamic quantisation GEMM
│   └── fused_moe.py         # MoE grouped GEMM (fp16/fp8/int8)
├── models/
│   ├── config.py            # ModelConfig over the HF AutoConfig
│   ├── registry.py          # model_type -> implementation class
│   ├── weights.py           # HF checkpoint keys -> parameters (+ TP shard)
│   ├── quantization.py      # QuantConfig registry (fp8/int8/int4/smoothquant)
│   ├── linear.py            # ColumnParallelLinear / RowParallelLinear
│   ├── interfaces.py        # MultiModalCausalLM, the multimodal capability
│   ├── base.py              # PagedAttention, FusedMLP, DecoderLayer, CausalLM
│   ├── rotary_embedding.py  # RoPE / mrope tables
│   └── llama.py / qwen2.py / qwen3.py / qwen3_moe.py / llava.py / qwen3_vl.py
├── distributed/
│   └── parallel_state.py    # TP process group, all-reduce, divide
└── utils/                   # chat templates, logger, image and path helpers
```

File and class names follow vLLM's, so the two are easy to read side by side: `model_runner.py` matches `v1/worker/gpu_model_runner.py`, `kv_cache_manager.py` matches `v1/core/kv_cache_manager.py`, `continuous_engine.py` plus `scheduler.py` match `v1/engine/` plus `v1/core/sched/`,
`async_engine.py` matches `AsyncLLMEngine`, `entrypoints/` matches `entrypoints/openai/`,
`models/interfaces.py` and `models/registry.py` match `model_executor/models/`, and the weight-loading split (key mapping in `models/weights.py`, file reading in `executor/weight_utils.py`) mirrors vLLM's `model_executor/models/utils.py` versus `model_loader/weight_utils.py`.

The per-model files declare only their differences — bias flags, per-head qk-norm, mrope, or DeepStack layer injection — while all shared behaviour lives in `models/base.py`. A new architecture typically means one class body plus one `ModelRegistry` entry; its config is whatever `AutoConfig` already returns.

## Development

```bash
make lint      # ruff check + ruff format --check
make format    # ruff --fix + ruff format
make test-cpu  # runs everything not marked gpu/weights
make test-gpu  # requires CUDA
```

`pre-commit` bundles ruff, typos, markdownlint, actionlint, a filename-space guard, and a custom hook that rejects hard-coded absolute paths in library code. The `tests` GitHub Actions workflow runs the CPU test subset on 3.10+ for every PR; the `pre-commit` workflow runs every hook against the whole tree.

## Acknowledgement

- [meta-llama/llama-models](https://github.com/meta-llama/llama-models/tree/main)
- [transformers](https://github.com/huggingface/transformers)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/tree/main)
- [kernl](https://github.com/ELS-RD/kernl/tree/main)
- [unsloth](https://github.com/unslothai/unsloth/tree/main)
- [openai-triton](https://triton-lang.org/main/getting-started/tutorials/)
- [lightllm](https://github.com/ModelTC/lightllm)
- [vllm](https://github.com/vllm-project/vllm)

## Citation

If you use Litellama in your research, please cite the following work:

```bibtex
@misc{litellama-2023,
  author       = {Litellama AI team},
  title        = {Litellama},
  howpublished = {\url{https://github.com/harleyszhang/lite_llama}},
  year         = {2023},
}
```