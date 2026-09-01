"""Tests for the external-backend tier: availability, install metadata, optionality.

``library_present`` semantics, every backend module's availability shape, the
install metadata against pyproject extras, and the rule that external
rows never make an op unusable when absent.

Usage:
    pytest tests/ops/test_external_backends.py
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

import lite_llama.kernels  # noqa: F401 — import side effect: native row registration
from lite_llama.kernels.backend import capability
from lite_llama.kernels.backend.capability import EXTERNAL_BACKENDS, BackendInstall, library_present
from lite_llama.kernels.dispatcher import REGISTRY, resolve_target

#: Import name each backend module must check, verified against the upstream
#: install docs. The distribution name and the import name differ for three of
#: the five, which is exactly the mistake this pin exists to catch.
IMPORT_NAMES = {
    "flashinfer": "flashinfer",  # distribution: flashinfer-python
    "deepgemm": "deep_gemm",  # distribution: deepgemm (renamed from deep-gemm)
    "flashmla": "flash_mla",  # source install
    "deepep": "deep_ep",  # source install
}

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


@pytest.fixture
def extras() -> dict[str, list[str]]:
    """The ``[project.optional-dependencies]`` table as declared on disk."""
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["optional-dependencies"]


class TestLibraryPresent:
    @pytest.fixture(autouse=True)
    def _clear_capability_cache(self):
        # The cache is process-wide; leaving a fake answer behind would make an
        # unrelated test believe a backend is (un)available.
        yield
        library_present.cache_clear()

    def test_a_module_that_imports_is_present(self) -> None:
        assert library_present("json") is True

    def test_a_module_that_does_not_exist_is_absent(self) -> None:
        # The everyday case: no wheel, no row, native serves the call.
        assert library_present("lite_llama_no_such_backend") is False

    def test_a_module_that_raises_on_import_is_absent(self, tmp_path, monkeypatch) -> None:
        """find_spec would say yes here; a compiled backend still would not run.

        This is the shape of a real failure — the package directory is on the
        path, so import machinery finds it, and loading it dies inside the
        extension. Dispatch must read that as "unavailable", not propagate it.
        """
        (tmp_path / "broken_backend.py").write_text('raise OSError("libcudart.so.12: not found")')
        monkeypatch.syspath_prepend(str(tmp_path))
        assert importlib.util.find_spec("broken_backend") is not None

        library_present.cache_clear()
        assert library_present("broken_backend") is False
        sys.modules.pop("broken_backend", None)

    def test_the_answer_is_cached(self, monkeypatch) -> None:
        """One failed import per module name per process, not one per dispatch."""
        calls: list[str] = []

        def counting_import(name: str) -> None:
            calls.append(name)
            raise ImportError(name)

        monkeypatch.setattr(
            capability, "importlib", types.SimpleNamespace(import_module=counting_import)
        )
        library_present.cache_clear()
        assert library_present("counted_backend") is False
        assert library_present("counted_backend") is False
        assert calls == ["counted_backend"]


class TestBackendModules:
    @pytest.mark.parametrize("backend", EXTERNAL_BACKENDS)
    def test_module_declares_the_metadata_dispatch_needs(self, backend: str) -> None:
        module = importlib.import_module(f"lite_llama.kernels.backend.{backend}")
        assert isinstance(module.INSTALL, BackendInstall)
        assert module.INSTALL.backend == backend, "INSTALL must name its own module"
        assert callable(module.available)

    @pytest.mark.parametrize("backend", EXTERNAL_BACKENDS)
    def test_available_answers_a_bool_without_raising(self, backend: str) -> None:
        # Called during dispatch filtering on machines that have none of these.
        module = importlib.import_module(f"lite_llama.kernels.backend.{backend}")
        assert isinstance(module.available(), bool)

    @pytest.mark.parametrize("backend", EXTERNAL_BACKENDS)
    def test_import_name_matches_upstream(self, backend: str) -> None:
        module = importlib.import_module(f"lite_llama.kernels.backend.{backend}")
        assert module.INSTALL.module == IMPORT_NAMES[backend]

    def test_importing_the_package_costs_no_third_party_import(self) -> None:
        """Backend modules are data; none of them may pull their library in.

        This is what lets ``backend/__init__`` import all four eagerly, which
        in turn is what makes an installed backend show up in dispatch without
        any registration step at the call site. Checked in a fresh interpreter
        because a check running earlier in this session may legitimately have
        imported a library that *is* installed.
        """
        code = (
            "import sys, lite_llama.kernels.backend as b; "
            f"print([m for m in {sorted(IMPORT_NAMES.values())} if m in sys.modules])"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert done.stdout.strip() == "[]"


class TestInstallMetadata:
    @pytest.mark.parametrize("backend", EXTERNAL_BACKENDS)
    def test_a_named_extra_exists_in_pyproject(self, backend: str, extras) -> None:
        """An extra in the metadata is a promise `pip install` must be able to keep."""
        install = importlib.import_module(f"lite_llama.kernels.backend.{backend}").INSTALL
        if install.extra is not None:
            assert install.extra in extras, f"{backend}: extra {install.extra!r} not in pyproject"
            assert extras[install.extra], f"{backend}: extra {install.extra!r} installs nothing"

    @pytest.mark.parametrize("backend", EXTERNAL_BACKENDS)
    def test_a_backend_without_an_extra_documents_a_source_build(self, backend: str) -> None:
        install = importlib.import_module(f"lite_llama.kernels.backend.{backend}").INSTALL
        assert install.extra or install.source_recipe
        assert install.homepage.startswith("https://")
        assert install.requires, "state the hardware/toolchain window in prose too"

    def test_neither_an_extra_nor_a_recipe_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="extra, a source recipe, or both"):
            BackendInstall(backend="ghost", module="ghost", homepage="https://x", requires="none")

    def test_how_to_get_it_reports_every_step(self) -> None:
        install = BackendInstall(
            backend="both",
            module="both",
            homepage="https://x",
            requires="sm90",
            extra="both",
            source_recipe="git clone https://x && pip install -e .",
        )
        line = install.how_to_get_it()
        assert "pip install lite-llama[both]" in line
        assert "git clone" in line

    def test_the_extras_never_reach_the_core_dependencies(self) -> None:
        with PYPROJECT.open("rb") as fh:
            project = tomllib.load(fh)["project"]
        core = " ".join(project["dependencies"])
        for extra in ("flashinfer",):
            for requirement in project["optional-dependencies"][extra]:
                assert requirement.split(">")[0] not in core


class TestSurvey:
    def test_reports_one_line_per_backend_in_declaration_order(self) -> None:
        results = capability.survey()
        assert tuple(install.backend for install, _ in results) == EXTERNAL_BACKENDS
        assert all(isinstance(present, bool) for _, present in results)

    def test_every_absent_backend_still_says_how_to_get_it(self) -> None:
        for install, present in capability.survey():
            if not present:
                assert install.how_to_get_it()


class TestExternalRowsStayOptional:
    """Gates on external rows, enforced from the milestone that adds the first one.

    They pass trivially while ``native`` is the only backend registered, and
    that is the point: the invariants are written down before the rows arrive,
    so M2.1 onwards cannot land a row that hard-fails on a machine without the
    library or that misspells its backend family.
    """

    def test_backend_families_are_the_declared_ones(self) -> None:
        families = {spec.backend for spec in REGISTRY.specs()}
        assert families <= {"native", *EXTERNAL_BACKENDS}

    def test_every_external_row_has_a_working_availability_check(self) -> None:
        for spec in REGISTRY.specs():
            if spec.backend == "native":
                continue
            assert spec.available is not None, f"{spec.name}: non-native row needs an availability check"
            assert isinstance(resolve_target(spec.available)(), bool)
