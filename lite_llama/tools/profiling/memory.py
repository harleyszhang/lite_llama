"""Static GPU memory budget from config numbers alone — no GPU, no model load.

:class:`ModelShape` captures the config-derived sizes,
:func:`compute_memory_budget` turns them into a :class:`MemoryBudget`
(weights, KV cache, activations, CUDA graphs), and the print/export
helpers render the table.

Usage:
    print_memory_budget()
"""

from __future__ import annotations

from dataclasses import dataclass

#: Bytes per element, keyed by the dtype names the config files use.
DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "uint8": 1, "fp8": 1, "int4": 1}

#: Workspace a captured CUDA graph set costs, over the tensors it replays. A fixed
#: over-estimate: it depends on how many shapes get captured, which is a runtime fact.
CUDA_GRAPH_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class ModelShape:
    """Every number a budget needs. Named as the model config names them.

    Attributes:
        num_kv_blocks: KV cache capacity in *tokens*, i.e. blocks x block size.
        tie_word_embeddings: When set, `lm_head` shares the embedding matrix and is
            not counted twice — worth getting right, it is a third of a small model.
    """

    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    num_kv_blocks: int
    weight_dtype: str = "fp16"
    kv_dtype: str = "fp16"
    max_batch_size: int = 16
    max_seq_len: int = 2048
    tie_word_embeddings: bool = False


@dataclass(frozen=True)
class MemoryBudget:
    """Static memory breakdown in bytes, plus GB views for printing."""

    model_weights_bytes: int
    kv_cache_bytes: int
    activation_bytes: int
    cuda_graph_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.model_weights_bytes
            + self.kv_cache_bytes
            + self.activation_bytes
            + self.cuda_graph_bytes
        )

    @property
    def model_weights_gb(self) -> float:
        return self.model_weights_bytes / (1024**3)

    @property
    def kv_cache_gb(self) -> float:
        return self.kv_cache_bytes / (1024**3)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)


def compute_memory_budget(shape: ModelShape) -> MemoryBudget:
    """Break the resident footprint into weights, KV cache, activations, graphs.

    Args:
        shape: The config numbers to budget for.

    Returns:
        The four components, in bytes.
    """
    w_bytes = DTYPE_BYTES.get(shape.weight_dtype, 2)
    kv_bytes = DTYPE_BYTES.get(shape.kv_dtype, 2)

    # Weights: embedding + layers + lm_head, where the head is free when tied.
    embed_params = shape.vocab_size * shape.hidden_size
    qkv_params = shape.hidden_size * (shape.num_heads + 2 * shape.num_kv_heads) * shape.head_dim
    o_params = shape.num_heads * shape.head_dim * shape.hidden_size
    ffn_params = shape.hidden_size * shape.intermediate_size * 3  # gate + up + down
    norm_params = shape.hidden_size * 2  # input_norm + post_norm
    per_layer_params = qkv_params + o_params + ffn_params + norm_params
    head_params = 0 if shape.tie_word_embeddings else embed_params
    total_params = embed_params + shape.num_layers * per_layer_params + head_params

    # KV cache: K and V, every layer, one entry per token of capacity.
    kv_cache_bytes = (
        2 * shape.num_layers * shape.num_kv_heads * shape.head_dim * shape.num_kv_blocks * kv_bytes
    )

    # Activations: a peak upper bound, assuming fp32 intermediates for the widest
    # batch x sequence the server admits. Deliberately loose.
    activation_bytes = shape.max_batch_size * shape.max_seq_len * shape.hidden_size * 2 * 4

    return MemoryBudget(
        model_weights_bytes=total_params * w_bytes,
        kv_cache_bytes=kv_cache_bytes,
        activation_bytes=activation_bytes,
        cuda_graph_bytes=CUDA_GRAPH_BYTES,
    )


def export_memory_budget(**shape_fields) -> str:
    """Render the budget for a :class:`ModelShape` as a markdown table.

    Args:
        **shape_fields: Any :class:`ModelShape` field, by keyword.

    Returns:
        A five-row markdown table: four components and their total.
    """
    shape = ModelShape(**shape_fields)
    budget = compute_memory_budget(shape)
    rows = [
        ("Model Weights", budget.model_weights_bytes),
        (f"KV Cache ({shape.kv_dtype})", budget.kv_cache_bytes),
        ("Activations", budget.activation_bytes),
        ("CUDA Graph", budget.cuda_graph_bytes),
    ]
    lines = ["| Component | Size | Percentage |", "|-----------|------|------------|"]
    lines += [
        f"| {label} | {nbytes / (1024**3):.2f} GB | {nbytes / budget.total_bytes * 100:.1f}% |"
        for label, nbytes in rows
    ]
    lines.append(f"| **Total** | **{budget.total_gb:.2f} GB** | 100% |")
    return "\n".join(lines)


def print_memory_budget(**shape_fields) -> None:
    """Print :func:`export_memory_budget` to stdout."""
    print(export_memory_budget(**shape_fields))
