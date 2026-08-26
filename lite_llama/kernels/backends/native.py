"""KernelSpec rows of the ``native`` backend: what the in-tree kernels can do.

The native backend is the floor every logical op falls back to — the Triton and
torch kernels that ship with lite_llama and that the golden baselines are
measured against. Each row here is *data about* a kernel, not a wrapper around
it: ``target`` names the real implementation module under
:mod:`lite_llama.kernels` (``flashdecoding``, ``update_kv_buffer``, ``linear``,
...), so there is no adapter layer to read through and no second place a
signature can drift. Rows stay torch-free on purpose — targets are
``"module:attr"`` strings that :func:`~lite_llama.kernels.ops.dispatch`
resolves on first use, which keeps import cost at parse speed.

Native rows therefore declare wide domains (any dtype/shape the kernel really
accepts), no capability window, and ``golden.verified`` — they *are* the
baseline. What they do declare is the layout contract the attention kernels
assume (``kv:paged``), so an external backend with its own KV pool (M2) is
excluded from dispatch until the call site can offer that layout.

Usage:
    from lite_llama.kernels.backends import native  # noqa: F401  (registers rows)
"""

from lite_llama.kernels.ops import GoldenRecord, KernelSpec, LayoutRequirement, register

#: The native rows define the golden baseline themselves.
BASELINE = GoldenRecord(verified=True, max_abs_diff=0.0, baseline="native")

#: The paged KV buffer this repo's cache manager allocates:
#: ``[2 * max_tokens, num_kv_heads, head_dim]``, K in the first half.
PAGED_KV = LayoutRequirement(required=("kv:paged",))

# --------------------------------------------------------------------------- #
# linear — one row per quantisation scheme, so dispatch never needs a branch
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="native/linear_torch",
        op="linear",
        backend="native",
        target="lite_llama.kernels.linear:linear_torch",
        schemes=("unquantized",),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w8a16",
        op="linear",
        backend="native",
        target="lite_llama.kernels.linear:linear_w8a16",
        dtypes=("bf16", "fp16"),
        schemes=("fp8", "blockwise_int8"),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w4a16",
        op="linear",
        backend="native",
        target="lite_llama.kernels.linear:linear_w4a16",
        dtypes=("bf16", "fp16"),
        schemes=("awq", "gptq"),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w8a8_int8",
        op="linear",
        backend="native",
        target="lite_llama.kernels.linear:linear_w8a8_int8",
        dtypes=("bf16", "fp16"),
        schemes=("w8a8_int8",),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/linear_w8a8_fp8",
        op="linear",
        backend="native",
        target="lite_llama.kernels.linear:linear_w8a8_fp8",
        dtypes=("bf16", "fp16"),
        schemes=("w8a8_fp8",),
        golden=BASELINE,
    )
)

# --------------------------------------------------------------------------- #
# attention — the two phases plus the cache write they share
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="native/flash_attention2_no_pad",
        op="attention.prefill",
        backend="native",
        target="lite_llama.kernels.flashattention2_nopad:flash_attention2_no_pad",
        # fp32 has no path through the kernel: it casts inputs to fp16.
        dtypes=("bf16", "fp16"),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/flash_decoding",
        op="attention.decode",
        backend="native",
        target="lite_llama.kernels.flashdecoding:flash_decoding",
        dtypes=("bf16", "fp16"),
        layout=PAGED_KV,
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/update_kv_buffer",
        op="kv_write",
        backend="native",
        target="lite_llama.kernels.update_kv_buffer:update_kv_buffer",
        # No dtype window: K/V arrive already quantised when the cache is fp8,
        # so uint8 rows are as legal here as bf16/fp16 ones.
        layout=PAGED_KV,
        golden=BASELINE,
    )
)
