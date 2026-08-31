<div align="center">

# Litellama

**A light llama-like llm inference framework based on the triton/cuda kernel.**

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/harleyszhang/lite_llama/blob/main/README.md)
[![zh](https://img.shields.io/badge/lang-zh-yellow.svg)](https://github.com/harleyszhang/lite_llama/blob/main/README.zh.md)
![PyPI - Python Version](https://img.shields.io/badge/python-3.13-blue)

<pre>
<b>Acceleration Features</b>
         ✅ Flash attention     ✅ Cuda Graph Optimize   ✅ Chunked Prefill         ✅ Prefix Caching
         ✅ W8A16 (fp8/int8)    ✅ W4A16 (AWQ/GPTQ)      ✅ SmoothQuant W8A8        ✅ FP8 KV Cache (2×)
         ✅ Kernel Autotune     ✅ Fused MoE             ✅ Tensor Parallel         ✅ Data Parallel

<b>Framework Design</b>
         ✅ Continuous batching ✅ OpenAI API server     ✅ Preemption              ✅ Ops Backend Registry
</pre>

</div>

## Features

- Up to **6.5×** speedup over HuggingFace `transformers` (Qwen3-0.6B, A10, greedy) — see the [benchmark table](#qwen3-06b-benchmark) below.
- **Online batch inference with continuous batching**: requests join and leave a running batch, so an arrival never waits for the current generation to finish. On one A10 with Qwen2.5-1.5B-Instruct and requests arriving 250 ms apart, throughput goes from 93 → 644 tok/s (**6.9×**) and mean latency from 19.1 s → 2.3 s (**8.3×**) — see [docs/continuous_batching.md](./docs/continuous_batching.md).
- **OpenAI-compatible server** (`lite-llama serve`): `/v1/completions` and `/v1/chat/completions` with streaming — the official `openai` client works unchanged. See [docs/online_serving.md](./docs/online_serving.md).
- Supports `llama3`, `Qwen2.5/Qwen3`, `Qwen3-MoE`, `LLaVA-1.5`, `Qwen3-VL`; `top-p` / `top-k` sampling and streaming output.
- **CUDA graph**: decode-stage CUDA graph capture (within batch-size limits).
- **Attention backends**: `flashattention2`, `flashdecoding` (with `NopadAttention` for unpadded sequences and GQA support). Dynamic KV-cache management via paged `TokenAttention` slots.
- **Operator fusion**: `silu` multiply, K/V projection fusion, skip-connection + `rmsnorm`. Custom `triton` kernels for `rmsnorm`, `rope`, `softmax`, and element-wise multiply.
- **Quantization**: W8A16 (fp8/int8), W4A16 (AWQ/GPTQ), SmoothQuant W8A8 — up to **6.9×** decode speedup over HF fp16.
- **Tensor Parallelism**: split a 30B MoE model across 2× A10 (24 GB) with one all-reduce per block.
- **Data Parallelism**: replicate the model across GPUs and route requests between them — **2.00×** throughput on 2 GPUs (100% linear).
- **Kernel Autotune** (v0.5): offline search persists optimal tile configs per `(GPU, op, shape)` to `~/.cache/lite_llama/autotune/`; kernels auto-load on startup.
- **FP8 KV Cache** (v0.6): `--kv-cache-dtype fp8` halves KV memory — **1.91× capacity** (282K vs 148K tokens on A10) with only 9% throughput cost.
- **Chunked Prefill** (v0.7): long prompts split into 512-token chunks so per-step prefill work is bounded (2000 → 512 tokens, 3.9× lower peak) — decode requests interleave instead of waiting behind a whole prompt.
- **Prefix Caching** (v0.7): block-hash chained prefix reuse — shared system prompts are prefilled once and reused by later requests; LRU-evicted under capacity pressure (aligned with vLLM's `BlockPool`).
- **Preemption** (v0.7): opt-in recompute-based eviction (`enable_preemption`) when the running set exceeds slot capacity; evicted requests re-queue with a progress quantum that prevents livelock.
- **Backend Registry** (v0.8): declarative kernel-backend selection with probe + `explain_selection()`; environment-variable override and graceful degradation when a backend's dependency is missing.

## Setup and Installation

> If you don't have a physical server, you can try using [virtal cloud remote server](https://growthdata.virtaicloud.com/t/hK).

Requires Python 3.13+, CUDA-capable PyTorch 2.13.0+ and Triton 3.7.1+.

```bash
uv pip install -e .           # runtime deps
uv pip install -e . --group dev
pre-commit install
```

Development:

```bash
make lint      # ruff check + ruff format --check
make format    # ruff --fix + ruff format
make test-cpu  # runs everything not marked gpu/weights
make test-gpu  # requires CUDA
```

`pre-commit` bundles ruff, typos, markdownlint, actionlint, a filename-space guard, and a custom hook that rejects hard-coded absolute paths in library code. The `tests` GitHub Actions workflow runs the CPU test subset on 3.13+ for every PR; the `pre-commit` workflow runs every hook against the whole tree.

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

### Offline Batched Inference

Text generation:

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

**Image conditioned generation**:

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

Multimodal decode can also replay a captured CUDA graph — the vision tokens
sit in the KV cache by then, so the decode step is the same graph a text
model replays. Like `chat`, the REPL defaults to eager (one turn in flight
never amortises capture latency); pass `--cuda-graph` for long replies:

```bash
python -m lite_llama.cli vl-chat \
    --model-dir my_weight/Qwen3-VL-4B-Instruct \
    --image docs/images/llava_test/dog.jpeg \
    --prompt "What animal is in this picture? Answer in one sentence." \
    --temperature 0.0 --max-gen-len 48 --cuda-graph
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

## Optimize Features

### Continuous Batching

Six requests, three slots. Watch the slot column: when a request finishes, the slot it held is decoding a queued request on the very next step.

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

Or drive the engine directly — every prompt is an independent request, and each carries its own sampling parameters:

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

### Tensor Parallelism

Where data parallelism replicates the model, **tensor parallelism** splits the weights themselves — the only option when one card cannot hold the checkpoint. Pass `--tensor-parallel-size N` and the engine spawns the extra ranks itself; one GPU stays in one process, so a breakpoint in the engine loop is still a breakpoint in the kernel.

```bash
# 30B MoE on 2x A10
python -m lite_llama.cli chat \
    --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --tensor-parallel-size 2
```

```python
from lite_llama.engine.continuous_engine import ContinuousBatchingEngine

engine = ContinuousBatchingEngine.from_pretrained("my_weight/Qwen3-8B", tensor_parallel_size=2)
```

The engine never learns how many processes run its model: it hands a plan to an `Executor` (`UniProcExecutor` for one GPU, `MultiprocExecutor` for many) and gets sampled tokens back. Because the plan is pure data, driver and follower ranks run one code path rather than two — no mirror process re-deriving the batch from a broadcast prompt, which is what used to turn any disagreement into an NCCL hang. Plans travel on a CPU (gloo) group so the control plane never stages through GPU memory, while the vocabulary-parallel sampler exchanges **two scalars per row** instead of gathering logits, keeping per-step traffic independent of vocabulary size.

That last claim is not an argument — it is measured. Every collective reports its payload to a **collective ledger**, so you can ask a step what it cost:

![tensor parallel](./docs/images/tensor_parallel.gif)

```python
from lite_llama.tools.observability import CollectiveStats

with CollectiveStats.collect() as stats:
    engine.step()
print(stats.report())          # per-op calls and bytes, split data / control plane
```

Recording is windowed, so the default path costs one `if`; windows nest, so a per-step window inside a whole-run window comes out of a single pass. Regenerate the GIF above with `python scripts/gen_collective_gif.py` — it drives a real `tp=2` engine and every byte in it is a measurement.

See [docs/tensor_parallel.md](docs/tensor_parallel.md) for the design, the sharding rules (including why QKV is split per segment under GQA), and what byte-exact parity between `tp=1` and `tp=2` can and cannot assert under fp16.

### Quantization

lite_llama supports multiple weight quantization schemes (architecture aligned with [sglang](https://github.com/sgl-project/sglang)). See [docs/quantization.md](docs/quantization.md) for the full design and API.

| Scheme | CLI Flag | Weight | Activation | Speedup vs HF |
|--------|----------|--------|------------|---------------|
| fp8 (checkpoint) | auto-detected | fp8-e4m3 | fp16 | 6.4× |
| int8 (runtime) | `--quantization int8` | int8 | fp16 | 6.3× |
| fp8 W8A8 (runtime) | `--quantization fp8` | fp8-e4m3 | fp8-e4m3 | 3.1× |
| int4 AWQ/GPTQ | auto-detected | int4 | fp16 | — |
| smoothquant | `--quantization smoothquant` | int8 | int8 | — |
| fp8 KV cache | `--kv-cache-dtype fp8` | — | — | 2× KV capacity |

**FP8 checkpoint** (auto-detected from `config.json`):

```bash
python -m lite_llama.cli chat --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8
```

**Runtime int8 quantisation** (halve memory of any fp16 model):

```bash
python -m lite_llama.cli chat --model-dir my_weight/Qwen2.5-0.5B --quantization int8
```

**True W8A8 fp8** (no weight dequantisation, per-token fp8 activations):

```bash
python -m lite_llama.cli chat --model-dir my_weight/Qwen3-0.6B --quantization fp8
```

**FP8 KV cache** (halve decode memory footprint):

```bash
python -m lite_llama.cli chat --model-dir my_weight/Qwen3-0.6B --kv-cache-dtype fp8
```

**Vision-language models** (Qwen3-VL / LLaVA, single GPU):

```bash
# Qwen3-VL vision chat
python -m lite_llama.cli vl-chat \
    --model-dir /data/shared/llm_weights/Qwen3-VL-4B-Instruct \
    --image photo.jpg

# LLaVA with INT8 quantisation
python -m lite_llama.cli vl-chat \
    --model-dir /data/shared/llm_weights/llava-hf/llava-1.5-7b-hf \
    --image photo.jpg --quantization int8
```

> `vl-chat` is single-GPU: tensor parallelism runs through the continuous-batching
> engine, which hosts text checkpoints only, so `--tensor-parallel-size > 1` exits with
> that message rather than pretending.

#### Qwen3-0.6B Benchmark 

Envirnment: (A10, batch=4, greedy)

How to run Benchmarks:

```bash
# Single model
python benchmarks/bench_quant.py --model-dir /data/shared/llm_weights/Qwen3-0.6B \
    --schemes fp16 int8 fp8 --json docs/benchmark_logs/bench_quant_Qwen3-0.6B.json

# All representative models (Qwen3-0.6B, 0.6B-FP8, VL-4B, 30B-MoE)
python benchmarks/bench_quant.py --all
```

Quantization Benchmark Result (A10, Qwen3-0.6B, batch=4, seq_len=25, gen_len=64, greedy):

| Config | Model Mem | KV Capacity | TPOT (ms) | TPS | vs HF fp16 |
|--------|-----------|-------------|-----------|-----|------------|
| HF fp16 (baseline) | 1.17 GB | — | 28.19 | 141.7 | 1.0× |
| lite fp16 | 1.40 GB | 147,875 tok | 4.14 | 918.8 | **6.5×** |
| lite int8 | 0.99 GB | 141,549 tok | 4.16 | 904.1 | **6.4×** |
| lite int8-blockwise | 1.00 GB | 138,385 tok | 4.44 | 849.4 | **6.0×** |
| lite fp8 (W8A8) | 0.99 GB | 139,153 tok | 8.35 | 448.1 | **3.2×** |
| lite smoothquant (W8A8) | 0.99 GB | 135,642 tok | 3.70 | 983.8 | **6.9×** |

> Model Mem = model weights only; KV Capacity = max cached tokens (paged pool fills remaining GPU memory).
> Benchmark logs: [`docs/benchmark_logs/`](docs/benchmark_logs/)

Quantization benchmark visualization (Qwen3-0.6B, A10, all schemes vs HF fp16):

![quantization benchmark](./docs/images/quantization_benchmark.gif)

#### Qwen3-VL-4B-Instruct Benchmark

A10, batch=4, seq_len=25, gen_len=64, greedy benchmark result:

| Config | Model Mem | KV Capacity | TPOT (ms) | TPS |
|--------|-----------|-------------|-----------|-----|
| lite fp16 | 8.99 GB | 73,676 tok | 23.36 | 170.7 |
| lite int8 | 5.61 GB | 93,559 tok | 27.47 | 145.3 |
| lite int8-blockwise | 5.71 GB | 92,748 tok | 27.97 | 142.7 |
| lite fp8 (W8A8) | 5.61 GB | 93,345 tok | 59.25 | 67.4 |
| lite smoothquant (W8A8) | 5.61 GB | 93,559 tok | 34.00 | 117.5 |

> Vision tower stays fp16 (not quantised); only language model projections are quantised.

![Qwen3-VL-4B quantization benchmark](./docs/images/Qwen3-VL-4B-Instruct_quantization_benchmark.gif)

### Data Parallelism

Where tensor parallelism splits one model across GPUs, data parallelism replicates the whole model onto each GPU and routes the request stream between the replicas — for throughput once one card is saturated. Each prompt is dealt to a replica by a load balancer (round-robin or least-loaded), and the replicas decode their own batches concurrently. See [docs/data_parallel.md](docs/data_parallel.md) for the design (it mirrors vLLM's `DPEngineCoreProc` / `DPLBAsyncMPClient` and SGLang's `DataParallelController`) and the benchmarks.

![data parallel](./docs/images/data_parallel.gif)

```python
from lite_llama import DataParallelEngine, SamplingParams

# Two whole-model replicas, one per GPU; requests routed round-robin.
with DataParallelEngine(model="my_weight/Qwen2.5-1.5B-Instruct", data_parallel_size=2) as engine:
    outputs = engine.generate(prompts, SamplingParams(temperature=0.0))
```

For serving, `lite-llama serve --data-parallel-size 2 --load-balancer total_tokens` swaps in `AsyncDataParallelEngine`, which streams each request's chunks from whichever replica the balancer picks and aborts a request whose connection drops.

On 2× A10 (Qwen2.5-1.5B-Instruct): **weak scaling 2.00x** (100% linear, 1857 → 3716 tok/s) with byte-identical outputs, and **1.64x** on a fixed 256-prompt batch. Compose it with TP — `data_parallel_size=2, tensor_parallel_size=2` — on a 4-GPU box.

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
│   ├── executor.py          # Executor seam: UniProc (1 GPU) / Multiproc (TP)
│   ├── worker.py            # ModelInput: one model pass described as data
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
│   ├── quantization/        # w8a16 / w4a16 / w8a8 / fp8 GEMMs
│   ├── backends/            # probe + priority registry, per op
│   ├── autotune/            # config search, keying and persistence
│   ├── flashattention2_nopad.py / flashdecoding.py
│   └── fused_moe.py         # MoE grouped GEMM (fp16/fp8/int8)
├── modules/                 # layers, reusable across architectures
│   ├── linear.py            # Column / Row / QKVParallelLinear
│   ├── vocab_parallel.py    # VocabParallelEmbedding / ParallelLMHead
│   ├── attention.py         # PagedAttention over the KV pool
│   ├── mlp.py / moe.py      # FusedMLP, sparse MoE block
│   ├── rotary_embedding.py  # RoPE / mrope tables
│   └── quantization/        # QuantConfig registry + per-scheme methods
├── models/
│   ├── config.py            # ModelConfig over the HF AutoConfig
│   ├── registry.py          # model_type -> implementation class
│   ├── weights.py           # HF checkpoint keys -> parameters (+ TP shard)
│   ├── interfaces.py        # MultiModalCausalLM, the multimodal capability
│   ├── base.py              # DecoderLayer, CausalLM, shared forward
│   └── llama.py / qwen2.py / qwen3.py / qwen3_moe.py / llava.py / qwen3_vl.py
├── distributed/
│   └── parallel_state.py    # dp x tp grid, NCCL data plane + gloo control plane
└── utils/                   # chat templates, logger, image and path helpers
```

File and class names follow vLLM's, so the two are easy to read side by side: `model_runner.py` matches `v1/worker/gpu_model_runner.py`, `kv_cache_manager.py` matches `v1/core/kv_cache_manager.py`, `continuous_engine.py` plus `scheduler.py` match `v1/engine/` plus `v1/core/sched/`, `async_engine.py` matches `AsyncLLMEngine`, `entrypoints/` matches `entrypoints/openai/`, `models/interfaces.py` and `models/registry.py` match `model_executor/models/`, and the weight-loading split (key mapping in `models/weights.py`, file reading in `executor/weight_utils.py`) mirrors vLLM's `model_executor/models/utils.py` versus `model_loader/weight_utils.py`.

The per-model files declare only their differences — bias flags, per-head qk-norm, mrope, or DeepStack layer injection — while all shared behaviour lives in `models/base.py`. A new architecture typically means one class body plus one `ModelRegistry` entry; its config is whatever `AutoConfig` already returns.

## Acknowledgement

- [meta-llama/llama-models](https://github.com/meta-llama/llama-models/tree/main)
- [transformers](https://github.com/huggingface/transformers)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/tree/main)
- [kernl](https://github.com/ELS-RD/kernl/tree/main)
- [unsloth](https://github.com/unslothai/unsloth/tree/main)
- [openai-triton](https://triton-lang.org/main/getting-started/tutorials/)
- [lightllm](https://github.com/ModelTC/lightllm)
- [vllm](https://github.com/vllm-project/vllm)
- [sglang](https://github.com/sgl-project/sglang)

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
