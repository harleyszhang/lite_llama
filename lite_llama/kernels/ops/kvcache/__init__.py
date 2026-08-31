"""KV-cache domain: writing new tokens into the paged buffer.

One op, one row: ``kv_write`` has no backend politics — the buffer layout is
this repo's own ``[max_tokens, 2 * num_kv_heads, head_dim]`` allocation, and
any backend that wants to serve it has to write into exactly that. The
``kv:paged`` tag is the contract; no dtype window because K/V arrive already
quantised when the cache is fp8, so uint8 rows are as legal here as bf16 ones.

Usage:
    from lite_llama.kernels.ops import kvcache  # noqa: F401  (registers the row)
"""

from lite_llama.kernels.dispatcher import NATIVE_BASELINE, PAGED_KV, KernelSpec, register

register(
    KernelSpec(
        name="native/update_kv_buffer",
        op="kv_write",
        backend="native",
        target="lite_llama.kernels.ops.kvcache.update_kv_buffer:update_kv_buffer",
        layout=PAGED_KV,
        golden=NATIVE_BASELINE,
    )
)
