"""Tests for :class:`KernelSpec` and its requirement dataclasses.

Shape constraints, shape and layout requirements, spec validation,
membership predicates and torch-free registration — the declarative
contract dispatch trusts.

Usage:
    pytest tests/ops/test_spec.py
"""

from __future__ import annotations

import pytest

import lite_llama.kernels.dispatcher as lite_llama_kernels_dispatcher
from lite_llama.kernels.dispatcher import (
    GoldenRecord,
    KernelSpec,
    LayoutRequirement,
    ShapeConstraint,
    ShapeRequirement,
)
from lite_llama.platform.spec import CapabilityRequirement


def spec(**overrides) -> KernelSpec:
    """A minimal valid spec; tests override individual fields."""
    base = {
        "name": "native/example",
        "op": "linear",
        "backend": "native",
        "target": "some.module:callable",
        "golden": GoldenRecord(verified=True, max_abs_diff=0.0, baseline="F.linear"),
    }
    return KernelSpec(**{**base, **overrides})


class TestShapeConstraint:
    def test_min_max_mod(self) -> None:
        dims = {"m": 64, "k": 48}
        assert ShapeConstraint("m", "min", 16).satisfied_by(dims)
        assert ShapeConstraint("m", "max", 128).satisfied_by(dims)
        assert ShapeConstraint("k", "mod", 16).satisfied_by(dims)
        assert not ShapeConstraint("m", "mod", 48).satisfied_by(dims)  # 64 % 48 != 0

    def test_boundaries_are_inclusive(self) -> None:
        assert ShapeConstraint("m", "min", 64).satisfied_by({"m": 64})
        assert ShapeConstraint("m", "max", 64).satisfied_by({"m": 64})

    def test_missing_dimension_never_matches(self) -> None:
        """Declaring a constraint means the impl cannot run without that dim."""
        assert not ShapeConstraint("n", "mod", 8).satisfied_by({"m": 16})

    def test_str_reads_like_the_math(self) -> None:
        assert str(ShapeConstraint("k", "mod", 16)) == "k%16"
        assert str(ShapeConstraint("m", "min", 8)) == "m>=8"


class TestShapeRequirement:
    def test_hard_gate_filters(self) -> None:
        req = ShapeRequirement(hard=(ShapeConstraint("k", "mod", 16),))
        assert req.is_feasible({"k": 64})
        assert not req.is_feasible({"k": 40})  # 40 % 16 != 0

    def test_empty_hard_admits_anything(self) -> None:
        assert ShapeRequirement().is_feasible({})

    def test_prefer_scores_without_filtering(self) -> None:
        req = ShapeRequirement(prefer=(ShapeConstraint("m", "mod", 128),))
        assert req.is_feasible({"m": 7})  # not preferred, still feasible
        assert req.preference_score({"m": 7}) == 0
        assert req.preference_score({"m": 256}) == 1


class TestLayoutRequirement:
    def test_satisfied_when_subset(self) -> None:
        req = LayoutRequirement(required=("weight:nt", "scale:block_128"))
        assert req.satisfied_by(frozenset({"weight:nt", "scale:block_128", "x:row"}))

    def test_missing_reports_the_exact_tags(self) -> None:
        req = LayoutRequirement(required=("weight:nt", "scale:block_128"))
        missing = req.missing_from(frozenset({"weight:nt"}))
        assert missing == frozenset({"scale:block_128"})

    def test_empty_requires_nothing(self) -> None:
        assert LayoutRequirement().satisfied_by(frozenset())


class TestKernelSpecValidate:
    def test_minimal_spec_is_valid(self) -> None:
        spec().validate()

    def test_name_must_carry_the_backend_prefix(self) -> None:
        with pytest.raises(ValueError, match="name prefix"):
            spec(name="flashinfer/impl", backend="native").validate()
        with pytest.raises(ValueError, match="'<backend>/<impl>'"):
            spec(name="no-slash").validate()

    def test_rejects_malformed_target_and_probe(self) -> None:
        with pytest.raises(ValueError, match=r"KernelSpec\.target"):
            spec(target="no-colon-here").validate()
        with pytest.raises(ValueError, match=r"KernelSpec\.available"):
            spec(available="also:bad:name").validate()

    def test_external_backend_must_declare_a_probe(self) -> None:
        """A missing wheel must be observable, or the never-fail floor breaks."""
        with pytest.raises(ValueError, match="available probe"):
            spec(name="deepgemm/fp8_gemm", backend="deepgemm").validate()
        spec(
            name="deepgemm/fp8_gemm", backend="deepgemm", available="deepgemm:_is_available"
        ).validate()

    def test_rejects_uppercase_op_ids(self) -> None:
        with pytest.raises(ValueError, match="dotted id"):
            spec(op="Linear.Fast").validate()


class TestMembershipPredicates:
    def test_dtype_empty_means_agnostic(self) -> None:
        assert spec().dtype_ok("bf16")
        assert not spec(dtypes=("fp16",)).dtype_ok("bf16")

    def test_scheme_membership(self) -> None:
        s = spec(schemes=("fp8", "w8a8"))
        assert s.scheme_ok("fp8")
        assert not s.scheme_ok("unquantized")

    def test_shape_and_layout_delegate(self) -> None:
        s = spec(
            shape=ShapeRequirement(hard=(ShapeConstraint("n", "mod", 8),)),
            layout=LayoutRequirement(required=("kv:paged",)),
        )
        assert s.shape_ok({"n": 16}) and not s.shape_ok({"n": 17})
        assert s.layout_missing(frozenset({"kv:ragged"})) == frozenset({"kv:paged"})


class TestTorchFreeRegistration:
    def test_spec_module_declares_no_torch_import(self) -> None:
        """The ops tier must not add torch to the import graph of its own accord.

        The ``lite_llama`` package root already imports torch via the engine, so
        "torch-free" here means the ops tier itself contributes no new heavy
        imports: spec/registry files may only import stdlib plus the platform
        descriptors. Asserted on the parsed AST so the check cannot be fooled by
        comments or strings.
        """
        import ast
        from pathlib import Path

        spec_py = Path(lite_llama_kernels_dispatcher.__file__).parent / "spec.py"
        tree = ast.parse(spec_py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] in {"torch", "triton"} for n in names), (
                f"dispatcher/spec.py must stay torch-free, found import of {names}"
            )

    def test_capability_composes_with_platform(self) -> None:
        """The spec tier reuses platform's requirement type — one vocabulary."""
        from lite_llama.platform.spec import PlatformInfo, capabilities_match

        s = spec(capability=(CapabilityRequirement("cuda", min_cc=(9, 0)),))
        assert capabilities_match(s.capability, PlatformInfo("cuda", 9, 0, "H100"))
        assert not capabilities_match(s.capability, PlatformInfo("cuda", 8, 6, "A10"))
