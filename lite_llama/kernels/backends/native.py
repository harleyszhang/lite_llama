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

#: The paged KV buffer this repo's cache manager allocates
#: (``kv_cache_manager.py``): ``[max_tokens, 2 * num_kv_heads, head_dim]``, the
#: K heads first then the V heads, so one token's K and V are adjacent.
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

# --------------------------------------------------------------------------- #
# moe — one row, because the kernel reads the format off the weight dtype
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="native/fused_moe",
        op="moe",
        backend="native",
        target="lite_llama.kernels.fused_moe:fused_moe",
        dtypes=("bf16", "fp16"),
        # Every scheme the quantisation methods route here: fused_moe derives
        # the expert format from w1.dtype (uint8 fp8-e4m3 / int8 / packed int32)
        # rather than from a flag, so splitting the row per scheme would be one
        # spec claim per branch of the same dispatch it already does internally.
        schemes=(
            "unquantized",
            "fp8",
            "w8a8_fp8",
            "w8a8_int8",
            "blockwise_int8",
            "awq",
            "gptq",
        ),
        golden=BASELINE,
    )
)

# --------------------------------------------------------------------------- #
# norm / rope / elementwise — the per-layer glue around the two GEMM domains
# --------------------------------------------------------------------------- #
register(
    KernelSpec(
        name="native/skip_rmsnorm",
        op="rmsnorm",
        backend="native",
        target="lite_llama.kernels.skip_rmsnorm:skip_rmsnorm",
        # One row covers both the fused (residual) and plain paths: the kernel
        # picks between them on `residual is None`, and a backend that only has
        # the plain one would be a different row, not a different flag.
        dtypes=("bf16", "fp16"),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/rope_emb_forward",
        op="rope",
        backend="native",
        target="lite_llama.kernels.rope_emb:rope_emb_forward",
        dtypes=("bf16", "fp16"),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/swiglu_forward_fused",
        op="elementwise.swiglu",
        backend="native",
        target="lite_llama.kernels.swiglu:swiglu_forward_fused",
        dtypes=("bf16", "fp16"),
        golden=BASELINE,
    )
)
register(
    KernelSpec(
        name="native/swiglu_forward",
        op="elementwise.swiglu_split",
        backend="native",
        target="lite_llama.kernels.swiglu:swiglu_forward",
        dtypes=("bf16", "fp16"),
        golden=BASELINE,
    )
)
