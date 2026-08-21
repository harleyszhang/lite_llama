# lite_llama

A lightweight, Triton-kernel based LLM inference framework for LLaMA / Qwen2 / Qwen3 /
LLaVA / Qwen3-VL, refactored for a small, testable and readable codebase.

* Custom Triton kernels: flash-attention 2 (variable-length prefill),
  flash-decoding, SwiGLU, skip-RMSNorm, split-softmax, paged KV read/write.
* Paged KV cache with dynamic memory profiling.
* Unified engine for text and vision-language models — no per-model generation loop.
* Loads HuggingFace checkpoints directly: configs come from `AutoConfig`, weights are
  streamed from `*.safetensors`. No conversion step, no private file format.
* Single `ModelRegistry` mapping HF `model_type` to an implementation class.

## Supported models

| Family    | Text | Vision-language                | Notes                                    |
| --------- | :--: | :----------------------------: | ---------------------------------------- |
| LLaMA     |  ✅  |                                | HF `LlamaForCausalLM` layout             |
| Qwen2     |  ✅  |                                | q/k/v projections carry a bias           |
| Qwen3     |  ✅  |                                | per-head RMSNorm on q/k, wide q_size     |
| Qwen3-MoE |  ✅  |                                | top-k routed experts; FP8 block-quantised checkpoints are dequantised to fp16 while loading |
| LLaVA-1.5 |      |               ✅               | CLIP vision tower + MLP projector        |
| Qwen3-VL  |      |               ✅               | mrope + DeepStack visual feature merge   |

## Installation

Requires Python 3.10+, CUDA-capable PyTorch 2.4+ and Triton 3.0+.

```bash
uv pip install -e .           # runtime deps
uv pip install -e . --group dev
pre-commit install
```

## Quick start

### Get the weights

Point `--model-dir` at a HuggingFace checkpoint directory — the one holding
`config.json` and `*.safetensors` — exactly as `modelscope download` leaves it.
There is no conversion step: `config.json` is parsed by `AutoConfig`, and the weights
are streamed from the safetensors shards straight into the model, with the K/V
projections fused and the MoE experts stacked on the way in
(see `lite_llama/models/weights.py`).

```bash
modelscope download Qwen/Qwen2.5-0.5B         --local-dir my_weight/Qwen2.5-0.5B
modelscope download Qwen/Qwen3-0.6B           --local-dir my_weight/Qwen3-0.6B
modelscope download llava-modelscope/llava-1.5-7b-modelscope  --local-dir my_weight/llava-1.5-7b-modelscope
modelscope download Qwen/Qwen3-VL-4B-Instruct --local-dir my_weight/Qwen3-VL-4B-Instruct
modelscope download Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --local-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8
```

Legacy `pytorch_model*.bin` checkpoints still load; safetensors wins when both are
present.

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
lite-llama vl-chat --model-dir my_weight/llava-1.5-7b-modelscope \
                   --image docs/images/dog.jpeg \
                   --prompt "USER: <image>\nWhat animal is this? ASSISTANT:"
```

llava-1.5-7b-modelscope default inference:

```bash
python -m lite_llama.cli chat --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8
```

## Architecture

```text
lite_llama/
├── engine/
│   ├── llm.py               # LLM entry point
│   ├── llm_engine.py        # the single prefill/decode loop
│   ├── generator.py         # TextGenerator / VisionGenerator facades
│   ├── sampler.py           # temperature / top-p / repetition penalty
│   ├── detokenizer.py       # incremental text output
│   ├── stop_criteria.py     # EOS, repetition and length stopping
│   ├── multimodal.py        # processor call + mrope position ids
│   └── outputs.py           # RequestOutput / CompletionOutput
├── executor/
│   ├── model_runner.py      # owns the model, KV cache and per-step forward
│   ├── loader.py            # HF checkpoint -> fp16 parameters
│   ├── weight_utils.py      # safetensors reading, FP8 dequantisation
│   ├── kv_cache_manager.py  # paged KV pool + memory profiler
│   ├── attention_metadata.py # per-step KV bookkeeping handed to the kernels
│   └── cuda_graph.py        # decode graph capture and replay
├── kernels/                 # Triton kernels used by the models
├── models/
│   ├── config.py            # ModelConfig over the HF AutoConfig
│   ├── registry.py          # model_type -> implementation class
│   ├── weights.py           # HF checkpoint keys -> parameters
│   ├── interfaces.py        # MultiModalCausalLM, the multimodal capability
│   ├── base.py              # PagedAttention, FusedMLP, DecoderLayer, CausalLM
│   ├── rotary_embedding.py  # RoPE / mrope tables
│   └── llama.py / qwen2.py / qwen3.py / qwen3_moe.py / llava.py / qwen3_vl.py
└── utils/                   # chat templates, logger, image and path helpers
```

File and class names follow vLLM's, so the two are easy to read side by side:
`model_runner.py` matches `v1/worker/gpu_model_runner.py`, `kv_cache_manager.py`
matches `v1/core/kv_cache_manager.py`, `models/interfaces.py` and `models/registry.py`
match `model_executor/models/`, and the weight-loading split (key mapping in
`models/weights.py`, file reading in `executor/weight_utils.py`) mirrors vLLM's
`model_executor/models/utils.py` versus `model_loader/weight_utils.py`.

The per-model files declare only their differences — bias flags, per-head qk-norm,
mrope, or DeepStack layer injection — while all shared behaviour lives in
`models/base.py`. A new architecture typically means one class body plus one
`ModelRegistry` entry; its config is whatever `AutoConfig` already returns.

## Development

```bash
make lint      # ruff check + ruff format --check
make format    # ruff --fix + ruff format
make test-cpu  # runs everything not marked gpu/weights
make test-gpu  # requires CUDA
```

`pre-commit` bundles ruff, typos, markdownlint, actionlint, a filename-space guard,
and a custom hook that rejects hard-coded absolute paths in library code. The
`tests` GitHub Actions workflow runs the CPU test subset on 3.10 / 3.12 for every
PR; the `pre-commit` workflow runs every hook against the whole tree.

## License

Apache-2.0 — see `LICENSE`.
