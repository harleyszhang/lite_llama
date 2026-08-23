"""viz.memory: static memory budget table (L1).

Computes and renders a breakdown of GPU memory usage: model weights, KV cache,
activations, and CUDA graph workspace. Pure computation from config parameters,
no GPU required.

Usage:
    table = export_memory_budget(config, num_kv_blocks=100000, kv_dtype="fp16")
    print_memory_budget(config, num_kv_blocks=100000)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryBudget:
    """Static memory breakdown in bytes."""

    model_weights_bytes: int
    kv_cache_bytes: int
    activation_bytes: int
    cuda_graph_bytes: int
    total_bytes: int

    @property
    def model_weights_gb(self) -> float:
        return self.model_weights_bytes / (1024**3)

    @property
    def kv_cache_gb(self) -> float:
        return self.kv_cache_bytes / (1024**3)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)


def _dtype_bytes(dtype_str: str) -> int:
    """Bytes per element for a dtype string."""
    return {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "uint8": 1, "fp8": 1, "int4": 1}.get(
        dtype_str, 2
    )


def compute_memory_budget(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    num_kv_blocks: int,
    weight_dtype: str = "fp16",
    kv_dtype: str = "fp16",
    max_batch_size: int = 16,
    max_seq_len: int = 2048,
) -> MemoryBudget:
    """Compute static memory budget from model config.

    Args:
        num_layers: Number of transformer layers.
        hidden_size: Model hidden dimension.
        intermediate_size: FFN intermediate dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (GQA).
        head_dim: Dimension per head.
        vocab_size: Vocabulary size.
        num_kv_blocks: KV cache capacity in tokens.
        weight_dtype: Model weight dtype string.
        kv_dtype: KV cache dtype string.
        max_batch_size: Maximum batch size for activation estimate.
        max_seq_len: Maximum sequence length.
    """
    w_bytes = _dtype_bytes(weight_dtype)
    kv_bytes = _dtype_bytes(kv_dtype)

    # Model weights: embedding + layers + lm_head
    embed_params = vocab_size * hidden_size
    # Per layer: qkv_proj + o_proj + gate_proj + up_proj + down_proj + norms
    qkv_params = hidden_size * (num_heads + 2 * num_kv_heads) * head_dim
    o_params = num_heads * head_dim * hidden_size
    ffn_params = hidden_size * intermediate_size * 3  # gate + up + down
    norm_params = hidden_size * 2  # input_norm + post_norm
    per_layer_params = qkv_params + o_params + ffn_params + norm_params
    total_params = embed_params + num_layers * per_layer_params + vocab_size * hidden_size
    model_bytes = total_params * w_bytes

    # KV cache: 2 (K+V) * num_layers * num_kv_heads * head_dim * num_blocks
    kv_cache_bytes = 2 * num_layers * num_kv_heads * head_dim * num_kv_blocks * kv_bytes

    # Activation estimate: one forward pass peak (rough upper bound)
    act_bytes = max_batch_size * max_seq_len * hidden_size * 2 * 4  # fp32 intermediates

    # CUDA graph workspace (conservative estimate)
    graph_bytes = 256 * 1024 * 1024  # 256 MB

    total = model_bytes + kv_cache_bytes + act_bytes + graph_bytes

    return MemoryBudget(
        model_weights_bytes=model_bytes,
        kv_cache_bytes=kv_cache_bytes,
        activation_bytes=act_bytes,
        cuda_graph_bytes=graph_bytes,
        total_bytes=total,
    )


def export_memory_budget(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    num_kv_blocks: int,
    weight_dtype: str = "fp16",
    kv_dtype: str = "fp16",
    max_batch_size: int = 16,
    max_seq_len: int = 2048,
) -> str:
    """Export memory budget as a markdown table string."""
    budget = compute_memory_budget(
        num_layers, hidden_size, intermediate_size, num_heads,
        num_kv_heads, head_dim, vocab_size, num_kv_blocks,
        weight_dtype, kv_dtype, max_batch_size, max_seq_len,
    )

    lines = [
        "| Component | Size | Percentage |",
        "|-----------|------|------------|",
        f"| Model Weights | {budget.model_weights_gb:.2f} GB | {budget.model_weights_bytes/budget.total_bytes*100:.1f}% |",
        f"| KV Cache ({kv_dtype}) | {budget.kv_cache_gb:.2f} GB | {budget.kv_cache_bytes/budget.total_bytes*100:.1f}% |",
        f"| Activations | {budget.activation_bytes/(1024**3):.2f} GB | {budget.activation_bytes/budget.total_bytes*100:.1f}% |",
        f"| CUDA Graph | {budget.cuda_graph_bytes/(1024**3):.2f} GB | {budget.cuda_graph_bytes/budget.total_bytes*100:.1f}% |",
        f"| **Total** | **{budget.total_gb:.2f} GB** | 100% |",
    ]
    return "\n".join(lines)


def print_memory_budget(**kwargs) -> None:
    """Print memory budget table to stdout."""
    print(export_memory_budget(**kwargs))
