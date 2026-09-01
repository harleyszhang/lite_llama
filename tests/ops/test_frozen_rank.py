"""Frozen measured ranking: store -> provider -> dispatch rank (ROADMAP v0.10).

A store with fabricated measurements is installed as the perf
provider; dispatch must then rank by the frozen numbers, and
uninstalling restores the default provider.

Usage:
    pytest tests/ops/test_frozen_rank.py
"""

from __future__ import annotations

import pytest

from lite_llama.kernels.dispatcher import (
    DispatchKey,
    GoldenRecord,
    KernelSpec,
    OpRegistry,
    dispatch,
    set_perf_provider,
)
from lite_llama.kernels.dispatcher.autotune import ConfigStore, TuneKey
from lite_llama.kernels.dispatcher.autotune.frozen import (
    FROZEN_RANK_ENV,
    freeze_record,
    frozen_bucket,
    frozen_store,
    install_frozen_perf_provider,
    make_frozen_perf_provider,
)
from lite_llama.platform.spec import PlatformInfo

A10 = PlatformInfo("cuda", 8, 6, "NVIDIA A10")
H100 = PlatformInfo("cuda", 9, 0, "NVIDIA H100 80GB HBM3")

VERIFIED = GoldenRecord(verified=True, max_abs_diff=0.0, baseline="self")

#: The measurement set every dispatch test below freezes: x/measured is the
#: fastest, x/slow the static-priority winner, the floor in between.
MEASURED = {"native/floor": 50.0, "x/slow": 40.0, "x/measured": 30.0}


def _available_yes() -> bool:
    return True


def native(**over) -> KernelSpec:
    base = {
        "name": "native/floor",
        "op": "test.op",
        "backend": "native",
        "target": "math:sin",
        "golden": VERIFIED,
    }
    return KernelSpec(**{**base, **over})


def external(name: str, **over) -> KernelSpec:
    base = {
        "name": name,
        "op": "test.op",
        "backend": name.split("/")[0],
        "target": "math:cos",
        "available": "tests.ops.test_frozen_rank:_available_yes",
        "golden": VERIFIED,
    }
    return KernelSpec(**{**base, **over})


def make_reg(*extra: KernelSpec) -> OpRegistry:
    reg = OpRegistry()
    reg.register(native(priority=0))
    reg.register(external("x/slow", priority=9))  # the static-priority winner
    reg.register(external("x/measured", priority=1))
    for spec in extra:
        reg.register(spec)
    return reg


def key(
    *,
    op: str = "test.op",
    scheme: str = "unquantized",
    dims: dict[str, int] | None = None,
    dtype: str = "bf16",
    platform: PlatformInfo = A10,
) -> DispatchKey:
    return DispatchKey(
        op=op,
        dtype=dtype,
        scheme=scheme,
        shape=tuple(sorted((dims or {}).items())),
        layout=frozenset(),
        platform=platform,
        forced_backend=None,
    )


@pytest.fixture
def store(tmp_path):
    return ConfigStore(tmp_path)


@pytest.fixture
def frozen_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(FROZEN_RANK_ENV, "1")


@pytest.fixture
def restore_provider():
    yield
    set_perf_provider(lambda spec, key: None)


class TestFrozenBucket:
    def test_folds_scheme_and_buckets_m(self) -> None:
        assert (
            frozen_bucket("unquantized", {"m": 8, "n": 4096, "k": 4096})
            == "unquantized@M16_N4096_K4096"
        )
        assert frozen_bucket("fp8", {"m": 17, "n": 1, "k": 2}) == "fp8@M32_N1_K2"

    def test_shape_less_ops_collapse_to_the_canonical_bucket(self) -> None:
        assert frozen_bucket("unquantized", {}) == "unquantized@M16_N0_K0"

    def test_schemes_never_share_a_bucket(self) -> None:
        assert frozen_bucket("fp8", {}) != frozen_bucket("unquantized", {})


class TestFreezeRecord:
    def test_roundtrip_keeps_every_impls_latency(self, store: ConfigStore) -> None:
        tune_key = freeze_record(
            store,
            op="test.op",
            scheme="unquantized",
            dims={},
            dtype="bf16",
            measurements={"native/floor": 50.0, "x/fast": 30.1234567},
            gpu="NVIDIA_A10",
        )
        entry = store.get_entry(tune_key)
        assert entry is not None
        assert entry["config"]["kind"] == "frozen_rank"
        assert entry["config"]["winner"] == "x/fast"
        assert entry["config"]["impls"] == {"native/floor": 50.0, "x/fast": 30.123}
        assert entry["latency_us"] == 30.1234567

    def test_schemes_do_not_overwrite_each_other(self, store: ConfigStore) -> None:
        for scheme in ("unquantized", "fp8"):
            freeze_record(
                store,
                op="test.op",
                scheme=scheme,
                dims={"m": 8, "n": 8, "k": 8},
                dtype="bf16",
                measurements={"native/floor": 10.0},
                gpu="NVIDIA_A10",
            )
        assert len(store.load_all("test.op")) == 2

    def test_empty_measurements_are_rejected(self, store: ConfigStore) -> None:
        with pytest.raises(ValueError, match="nothing to freeze"):
            freeze_record(
                store,
                op="test.op",
                scheme="unquantized",
                dims={},
                dtype="bf16",
                measurements={},
                gpu="NVIDIA_A10",
            )


class TestProvider:
    def test_recorded_impl_reports_its_latency_in_ms(
        self, store: ConfigStore, frozen_on: None
    ) -> None:
        freeze_record(
            store,
            op="test.op",
            scheme="unquantized",
            dims={},
            dtype="bf16",
            measurements=MEASURED,
            gpu="NVIDIA_A10",
        )
        provider = make_frozen_perf_provider(store)
        assert provider(external("x/measured"), key()) == pytest.approx(0.030)
        assert provider(native(), key()) == pytest.approx(0.050)

    def test_unrecorded_impl_is_unmeasured(self, store: ConfigStore, frozen_on: None) -> None:
        freeze_record(
            store,
            op="test.op",
            scheme="unquantized",
            dims={},
            dtype="bf16",
            measurements=MEASURED,
            gpu="NVIDIA_A10",
        )
        provider = make_frozen_perf_provider(store)
        assert provider(external("x/other"), key()) is None

    def test_no_record_is_unmeasured(self, store: ConfigStore, frozen_on: None) -> None:
        assert make_frozen_perf_provider(store)(native(), key()) is None

    def test_disabled_by_env(self, store: ConfigStore, monkeypatch: pytest.MonkeyPatch) -> None:
        freeze_record(
            store,
            op="test.op",
            scheme="unquantized",
            dims={},
            dtype="bf16",
            measurements=MEASURED,
            gpu="NVIDIA_A10",
        )
        monkeypatch.setenv(FROZEN_RANK_ENV, "0")
        assert make_frozen_perf_provider(store)(external("x/measured"), key()) is None

    def test_records_only_apply_on_the_measured_gpu(
        self, store: ConfigStore, frozen_on: None
    ) -> None:
        freeze_record(
            store,
            op="test.op",
            scheme="unquantized",
            dims={},
            dtype="bf16",
            measurements=MEASURED,
            gpu="NVIDIA_A10",
        )
        provider = make_frozen_perf_provider(store)
        assert provider(external("x/measured"), key(platform=H100)) is None
        assert provider(external("x/measured"), key(platform=A10)) is not None

    def test_wrong_scheme_or_dtype_misses(self, store: ConfigStore, frozen_on: None) -> None:
        freeze_record(
            store,
            op="test.op",
            scheme="unquantized",
            dims={},
            dtype="bf16",
            measurements=MEASURED,
            gpu="NVIDIA_A10",
        )
        provider = make_frozen_perf_provider(store)
        assert provider(external("x/measured"), key(scheme="fp8")) is None
        assert provider(external("x/measured"), key(dtype="fp16")) is None

    def test_tile_config_entries_are_not_ranking_records(
        self, store: ConfigStore, frozen_on: None
    ) -> None:
        # Same bucket shape, but a tile-config payload (no kind marker) must
        # read as "no measurement", not as a zero-cost impl.
        store.put(
            TuneKey(
                gpu="NVIDIA_A10",
                op="test.op",
                shape_bucket=frozen_bucket("unquantized", {}),
                dtype="bf16",
            ),
            {"BLOCK_M": 16},
            latency_us=12.0,
        )
        assert make_frozen_perf_provider(store)(native(), key()) is None


class TestDispatchFlips:
    def test_measured_row_beats_static_priority(
        self, store: ConfigStore, frozen_on: None, restore_provider: None
    ) -> None:
        freeze_record(
            store,
            op="test.op",
            scheme="unquantized",
            dims={},
            dtype="bf16",
            measurements=MEASURED,
            gpu="NVIDIA_A10",
        )
        set_perf_provider(make_frozen_perf_provider(store))
        reg = make_reg()
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "x/measured"
        assert "perf=0.030ms" in sel.explain()

    def test_same_key_keeps_the_same_impl_across_cache_invalidation(
        self, store: ConfigStore, frozen_on: None, restore_provider: None
    ) -> None:
        freeze_record(
            store,
            op="test.op",
            scheme="unquantized",
            dims={},
            dtype="bf16",
            measurements=MEASURED,
            gpu="NVIDIA_A10",
        )
        set_perf_provider(make_frozen_perf_provider(store))
        reg = make_reg()
        first = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        reg.notify_change("test.op")  # pretend a row was re-registered
        second = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert first.spec.name == second.spec.name == "x/measured"

    def test_without_records_static_priority_rules(
        self, store: ConfigStore, frozen_on: None, restore_provider: None
    ) -> None:
        set_perf_provider(make_frozen_perf_provider(store))  # empty store
        reg = make_reg()
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "x/slow"

    def test_unmeasured_new_row_never_beats_a_frozen_one(
        self, store: ConfigStore, frozen_on: None, restore_provider: None
    ) -> None:
        freeze_record(
            store,
            op="test.op",
            scheme="unquantized",
            dims={},
            dtype="bf16",
            measurements=MEASURED,
            gpu="NVIDIA_A10",
        )
        set_perf_provider(make_frozen_perf_provider(store))
        reg = make_reg(external("x/new", priority=99))  # registered after the freeze
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "x/measured"


class TestInstall:
    def test_frozen_store_lives_under_the_frozen_subdir(self, tmp_path) -> None:
        store = frozen_store(tmp_path)
        assert store.cache_dir == tmp_path / "frozen"

    def test_install_wires_the_global_provider(
        self, store: ConfigStore, frozen_on: None, restore_provider: None
    ) -> None:
        freeze_record(
            store,
            op="test.op",
            scheme="unquantized",
            dims={},
            dtype="bf16",
            measurements=MEASURED,
            gpu="NVIDIA_A10",
        )
        install_frozen_perf_provider(store)
        reg = make_reg()
        sel = dispatch("test.op", dtype="bf16", platform_info=A10, registry=reg)
        assert sel.spec.name == "x/measured"
