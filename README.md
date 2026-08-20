# lite_llama

A lightweight, Triton-kernel based LLM inference framework for LLaMA / Qwen2 / Qwen3 /
LLaVA / Qwen3-VL, refactored for a small, testable and readable codebase.

* Custom Triton kernels: flash-attention 2 (variable-length prefill),
  flash-decoding, SwiGLU, skip-RMSNorm, split-softmax, paged KV read/write.
* Paged KV cache with dynamic memory profiling.
* Unified engine for text and vision-language models — no per-model generation loop.
* Single :class:`~lite_llama.models.registry.ModelRegistry` mapping HF `model_type`
  to config + implementation.

## Supported models

| Family    | Text | Vision-language                | Notes                                    |
| --------- | :--: | :----------------------------: | ---------------------------------------- |
| LLaMA     |  ✅  |                                | HF `LlamaForCausalLM` layout             |
| Qwen2     |  ✅  |                                | q/k/v projections carry a bias           |
| Qwen3     |  ✅  |                                | per-head RMSNorm on q/k, wide q_size     |
| Qwen3-MoE |  ✅  |                                | top-k routed experts; FP8 block-quantised checkpoints are dequantised to fp16 at convert time |
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

### Convert weights

lite_llama loads a single fused `<name>.pth` state_dict. `lite-llama-convert` reads
HuggingFace safetensors or `pytorch_model.bin*` shards, applies the per-architecture
rename table, fuses K/V projections along dim 0, and writes the result to
`my_weight/<name>/`.

```bash
lite-llama-convert /path/to/Qwen2.5-0.5B
lite-llama-convert /path/to/Qwen3-0.6B
lite-llama-convert /path/to/llava-hf/llava-1.5-7b-hf
lite-llama-convert /path/to/Qwen3-VL-4B-Instruct
lite-llama-convert /path/to/Qwen3-30B-A3B-Instruct-2507-FP8
```

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

gen = VisionGenerator(checkpoints_dir="my_weight/llava-1.5-7b-hf")
img = Image.open("docs/images/llava_test/dog.jpeg").convert("RGB")
prompt = "USER: <image>\nDescribe the animal in one sentence. ASSISTANT:"
print(gen.generate(prompt, [img], SamplingParams(temperature=0.0, max_gen_len=48)))
```

### CLI

llava-1.5-7b-hf default inference:

```bash
export LITE_LLAMA_MODEL_DIR=my_weight/Qwen2.5-0.5B
lite-llama chat                              # interactive text chat
lite-llama vl-chat --model-dir my_weight/llava-1.5-7b-hf \
                   --image docs/images/dog.jpeg \
                   --prompt "USER: <image>\nWhat animal is this? ASSISTANT:"
```

llava-1.5-7b-hf default inference:

```bash
python -m lite_llama.cli chat --model-dir my_weight/Qwen3-30B-A3B-Instruct-2507-FP8
```

## Architecture

```text
lite_llama/
├── engine/              # Single prefill/decode loop, sampler, high-level generators
├── executor/            # Model runtime: weight load, KV cache manager, forward dispatch
├── kernels/             # Triton kernels used by the models
├── models/              # base.py (PagedAttention, FusedMLP, DecoderLayer, CausalLM)
│                        # + llama.py / qwen2.py / qwen3.py / llava.py / qwen3_vl.py
├── tools/               # lite-llama-convert weight converter
└── utils/               # prompt templates, logger, image helpers
```

The five model files declare only their differences — bias flags, per-head qk-norm,
mrope, or DeepStack layer injection — while all shared behaviour lives in
`models/base.py`. A new architecture typically means one config + one class body plus
one registry entry.

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
