"""Tests for the native spec rows: registration, routing and the floor impl.

Importing :mod:`lite_llama.kernels` registers the native spec rows as a side
effect; these tests pin that catalogue and the scheme→row routing on CPU —
no kernel runs here, only dispatch and the ``F.linear`` floor (GPU numerics
of the Triton rows live in ``tests/kernels/test_linear_dispatch.py``).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import ClassVar

import pytest
import torch
import torch.nn.functional as F

import lite_llama.kernels  # noqa: F401 — import side effect: native spec registration
from lite_llama.kernels.backends import native as native_specs
from lite_llama.kernels.linear import linear_torch
from lite_llama.kernels.ops import LOGICAL_OPS, REGISTRY, dispatch
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

#: ``op -> row`` for the per-layer glue: one native row each, no choice to make.
#: ``elementwise.*`` is two rows because the two arities are two contracts:
#: the fused projection hands over one packed tensor, the split path two halves.
GLUE_ROWS = {
    "moe": "native/fused_moe",
    "rmsnorm": "native/skip_rmsnorm",
    "rope": "native/rope_emb_forward",
    "elementwise.swiglu": "native/swiglu_forward_fused",
    "elementwise.swiglu_split": "native/swiglu_forward",
}

#: Ops whose contract is declared but has no native row, with the reason. Pinned
#: so the gap stays a decision on record rather than something forgotten:
#: sampling already lives in the engine over a TP vocab slice, and EP comms need
#: an expert-parallel group this repo does not have yet (both land in M2).
OPS_WITHOUT_NATIVE_ROW = {
    "sample": "engine.Sampler owns it, over a tensor-parallel vocab slice",
    "comm.dispatch": "no EP group; MoE here is tensor parallel",
    "comm.combine": "pairs with comm.dispatch",
    "attention.mla_decode": "no MLA model in tree until v0.10",
    "elementwise": "open namespace root; only its members register rows",
}


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


class TestGlueCatalogue:
    """moe / rmsnorm / rope / elementwise: the domains between the two GEMMs."""

    @pytest.mark.parametrize("op,row", sorted(GLUE_ROWS.items()))
    def test_each_has_one_native_row(self, op: str, row: str) -> None:
        assert {s.name for s in REGISTRY.implementations(op)} == {row}
        assert REGISTRY.native_floor(op).name == row

    @pytest.mark.parametrize("op,row", sorted(GLUE_ROWS.items()))
    def test_dispatches_at_the_default_precision(self, op: str, row: str) -> None:
        assert dispatch(op, dtype="bf16").spec.name == row

    @pytest.mark.parametrize("op", sorted(GLUE_ROWS))
    def test_rows_declare_only_measured_dtypes(self, op: str) -> None:
        # bf16 (the default) and fp16 (what the kernel tests cover). fp32 is not
        # claimed: no path in the repo runs these kernels at fp32, so declaring
        # it would be an unbacked promise dispatch would happily act on.
        for spec in REGISTRY.implementations(op):
            assert set(spec.dtypes) == {"bf16", "fp16"}, spec.name
        with pytest.raises(LookupError, match="dtype"):
            dispatch(op, dtype="fp32")

    def test_one_moe_row_serves_every_scheme(self) -> None:
        # fused_moe reads the expert format off ``w1.dtype`` (uint8 -> fp8,
        # int8, int32 -> int4), so the scheme is not a choice between rows;
        # splitting them would write several specs for one internal branch.
        row = REGISTRY.native_floor("moe")
        for scheme in SCHEME_TO_ROW:
            assert dispatch("moe", dtype="bf16", scheme=scheme).spec.name == row.name


class TestContractCoverage:
    def test_every_registered_op_is_a_declared_contract(self) -> None:
        from lite_llama.kernels.ops import is_logical_op

        for op in REGISTRY.ops():
            assert is_logical_op(op), f"{op!r} has rows but no ABC"

    def test_contracts_without_rows_are_the_documented_ones(self) -> None:
        """A contract with no implementation is fine — silently is not."""
        registered = set(REGISTRY.ops())
        assert set(LOGICAL_OPS) - registered == set(OPS_WITHOUT_NATIVE_ROW)


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
        for op in REGISTRY.ops():
            for spec in REGISTRY.implementations(op):
                module, attr = spec.target.split(":")
                assert module.startswith("lite_llama.kernels."), spec.name
                # ops/ is the contract layer; kernels never live there.
                assert not module.startswith("lite_llama.kernels.ops"), spec.name
                assert attr.isidentifier()

    def test_every_row_resolves_to_a_callable(self) -> None:
        for op in REGISTRY.ops():
            for spec in REGISTRY.implementations(op):
                assert callable(resolve_target(spec.target)), spec.name


class TestTargetsMatchTheirContract:
    """Parameter *names* are the contract, because nothing adapts them.

    Dispatch hands the caller the kernel function itself — there is no wrapper
    tier translating argument names on the way through. So a kernel whose
    parameters merely happen to sit in the right order silently satisfies the
    ABC while breaking any caller that passes one by keyword, and breaking the
    next backend that reads the ABC to know what it must accept. This test is
    what caught ``update_kv_buffer(K_Values, ...)`` and ``skip_rmsnorm(X, ...)``.
    """

    #: ``elementwise.*`` members declare their own arity under an ABC that is
    #: deliberately ``(x, *args)``, so beyond "takes at least one operand" there
    #: is nothing to compare; see :class:`ElementwiseOp`.
    OPEN_ARITY = ("elementwise.",)

    #: The arity each open-arity member promises in its own docstring: the fused
    #: row takes the packed gate/up tensor, the split row takes the two halves.
    ELEMENTWISE_ARITY: ClassVar[dict[str, int]] = {
        "native/swiglu_forward_fused": 1,
        "native/swiglu_forward": 2,
    }

    def _ops_to_check(self) -> list[str]:
        return [op for op in sorted(REGISTRY.ops()) if not op.startswith(self.OPEN_ARITY)]

    def test_parameter_names_match_the_abc(self) -> None:
        for op in self._ops_to_check():
            expected = [
                p for p in inspect.signature(LOGICAL_OPS[op].__call__).parameters if p != "self"
            ]
            for spec in REGISTRY.implementations(op):
                got = list(inspect.signature(resolve_target(spec.target)).parameters)
                assert got == expected, (
                    f"{spec.name} takes {got} but the {op!r} contract says {expected}"
                )

    def test_open_arity_members_keep_the_arity_they_advertise(self) -> None:
        # The two swiglu rows differ only in arity, which is exactly why they are
        # two ops rather than one: dispatch cannot guess how many tensors the
        # call site holds, so the op id has to say it.
        for op in sorted(REGISTRY.ops()):
            if not op.startswith(self.OPEN_ARITY):
                continue
            for spec in REGISTRY.implementations(op):
                params = inspect.signature(resolve_target(spec.target)).parameters
                assert len(params) == self.ELEMENTWISE_ARITY[spec.name], spec.name
