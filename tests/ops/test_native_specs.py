"""Tests for the native spec rows: registration, routing and the floor impl.

Importing :mod:`lite_llama.kernels` registers the native spec rows as a side
effect; these tests pin that catalogue and the scheme→row routing on CPU —
no kernel runs here, only dispatch and the ``F.linear`` floor (GPU numerics
of the Triton rows live in ``tests/kernels/test_linear_dispatch.py``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import lite_llama.kernels  # noqa: F401 — import side effect: native spec registration
from lite_llama.kernels.backends import native as native_specs
from lite_llama.kernels.linear import linear_torch
from lite_llama.kernels.ops import REGISTRY, dispatch
from lite_llama.kernels.ops.dispatch import resolve_target
from lite_llama.modules.quantization.unquant import UnquantizedLinearMethod

#: scheme key -> spec name, the routing every quant method relies on.
SCHEME_TO_ROW = {
    "unquantized": "native/linear_torch",
    "fp8": "native/linear_w8a16",
    "blockwise_int8": "native/linear_w8a16",
    "awq": "native/linear_w4a16",
    "gptq": "native/linear_w4a16",
    "w8a8_int8": "native/linear_w8a8_int8",
    "w8a8_fp8": "native/linear_w8a8_fp8",
}

LINEAR_ROWS = {
    "native/linear_torch",
    "native/linear_w8a16",
    "native/linear_w4a16",
    "native/linear_w8a8_int8",
    "native/linear_w8a8_fp8",
}

#: ``op -> row``, the attention domain. One native row each, so the assertions
#: are about the *gates* on those rows rather than about a choice.
ATTENTION_ROWS = {
    "attention.prefill": "native/flash_attention2_no_pad",
    "attention.decode": "native/flash_decoding",
    "kv_write": "native/update_kv_buffer",
}

#: Layout tags the paged KV buffer of this repo satisfies.
PAGED = frozenset({"kv:paged"})


class TestNativeCatalogue:
    def test_linear_has_exactly_the_five_native_rows(self) -> None:
        assert {s.name for s in REGISTRY.implementations("linear")} == LINEAR_ROWS

    def test_rows_are_verified_native_floor(self) -> None:
        for spec in REGISTRY.implementations("linear"):
            assert spec.backend == "native"
            assert spec.golden.verified, f"{spec.name} must be golden-verified"

    @pytest.mark.parametrize("scheme,row", sorted(SCHEME_TO_ROW.items()))
    def test_scheme_routes_to_its_row(self, scheme: str, row: str) -> None:
        sel = dispatch("linear", dtype="bf16", scheme=scheme, shape={"m": 8, "n": 8, "k": 8})
        assert sel.spec.name == row

    def test_quantised_rows_reject_fp32_activations(self) -> None:
        # The Triton rows declare bf16/fp16 only. fp8 weights with fp32
        # activations is physically invalid — no floor can serve it, so
        # dispatch fails loud naming the dtype gate.
        sel = dispatch("linear", dtype="fp32", scheme="unquantized", shape={"m": 8, "n": 8, "k": 8})
        assert sel.spec.name == "native/linear_torch"
        with pytest.raises(LookupError, match="dtype"):
            dispatch("linear", dtype="fp32", scheme="fp8", shape={"m": 8, "n": 8, "k": 8})

    def test_all_targets_resolve(self) -> None:
        for spec in REGISTRY.implementations("linear"):
            assert callable(resolve_target(spec.target)), spec.name

    def test_floor_row_is_the_native_floor(self) -> None:
        assert REGISTRY.native_floor("linear").name == "native/linear_torch"


class TestAttentionCatalogue:
    @pytest.mark.parametrize("op,row", sorted(ATTENTION_ROWS.items()))
    def test_each_phase_has_its_native_row(self, op: str, row: str) -> None:
        assert {s.name for s in REGISTRY.implementations(op)} == {row}
        assert REGISTRY.native_floor(op).name == row

    def test_paged_rows_need_the_layout_tag(self) -> None:
        # The cache-facing rows read this repo's paged buffer. Without the tag
        # the call site is describing some other pool, and dispatch must say so
        # instead of handing over a kernel that would read the wrong strides.
        for op in ("attention.decode", "kv_write"):
            with pytest.raises(LookupError, match="layout"):
                dispatch(op, dtype="bf16")
            assert dispatch(op, dtype="bf16", layout=PAGED).spec.name == ATTENTION_ROWS[op]

    def test_prefill_reads_no_cache_so_needs_no_layout(self) -> None:
        # Prefill attends the freshly projected tensors, not the cache.
        sel = dispatch("attention.prefill", dtype="bf16")
        assert sel.spec.name == ATTENTION_ROWS["attention.prefill"]

    def test_kv_write_accepts_quantised_rows(self) -> None:
        # An fp8 cache is written as uint8 bytes: quantisation happens before
        # the write, so the row must not gate on the activation dtype.
        assert dispatch("kv_write", dtype="u8", layout=PAGED).spec.name == "native/update_kv_buffer"

    def test_all_targets_resolve(self) -> None:
        for op in ATTENTION_ROWS:
            for spec in REGISTRY.implementations(op):
                assert callable(resolve_target(spec.target)), spec.name


class TestLinearTorchFloor:
    def test_matches_f_linear_on_cpu(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        b = torch.randn(8, dtype=torch.bfloat16)
        assert torch.equal(linear_torch(x, w, bias=b), F.linear(x, w, b))

    def test_rejects_quantised_inputs_loudly(self) -> None:
        x = torch.randn(2, 8)
        w = torch.randn(4, 8)
        with pytest.raises(ValueError, match="unquantised floor"):
            linear_torch(x, w, weight_scale=torch.ones(4))


class TestUnquantMethodRoutesThroughDispatch:
    def test_apply_is_dispatched_not_hardwired(self) -> None:
        class _Layer:
            weight = torch.randn(8, 16)

        out = UnquantizedLinearMethod().apply(_Layer(), torch.randn(2, 16))  # type: ignore[arg-type]
        assert out.shape == (2, 8)


class TestRegistryStaysTorchFree:
    def test_registry_module_declares_no_torch_import(self) -> None:
        """Registration must not pay the torch import — targets are strings."""
        tree = ast.parse(Path(native_specs.__file__).read_text(encoding="utf-8"))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        offenders = [n for n in names if n.split(".")[0] in {"torch", "triton"}]
        assert not offenders, f"native.py must stay torch-free, found {offenders}"

    def test_rows_point_at_real_kernel_modules(self) -> None:
        """No wrapper tier: every target names a module under ``lite_llama.kernels``."""
        for op in ["linear", *ATTENTION_ROWS]:
            for spec in REGISTRY.implementations(op):
                module, attr = spec.target.split(":")
                assert module.startswith("lite_llama.kernels."), spec.name
                # ops/ is the contract layer; kernels never live there.
                assert not module.startswith("lite_llama.kernels.ops"), spec.name
                assert attr.isidentifier()
