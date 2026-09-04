"""Tests for the autotune config store — pure CPU, no GPU required.

Bucket maths, key equality, JSON round-trips and the lookup miss path:
the storage contract everything else trusts.

Usage:
    pytest tests/kernels/test_autotune_store.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from rapid_llm.kernels.dispatcher.autotune import (
    ConfigStore,
    TuneKey,
    bucket_m,
    get_best_config,
    make_shape_bucket,
    normalize_gpu_name,
    reset,
)


# --------------------------------------------------------------------------- #
# bucket_m
# --------------------------------------------------------------------------- #
class TestBucketM:
    def test_one_maps_to_16(self):
        assert bucket_m(1) == 16

    def test_16_maps_to_16(self):
        assert bucket_m(16) == 16

    def test_17_maps_to_32(self):
        assert bucket_m(17) == 32

    def test_32_maps_to_32(self):
        assert bucket_m(32) == 32

    def test_33_maps_to_64(self):
        assert bucket_m(33) == 64

    def test_64_maps_to_64(self):
        assert bucket_m(64) == 64

    def test_65_maps_to_128(self):
        assert bucket_m(65) == 128

    def test_129_maps_to_256(self):
        assert bucket_m(129) == 256

    def test_257_maps_to_512(self):
        assert bucket_m(257) == 512

    def test_overflow_caps_at_512(self):
        assert bucket_m(1000) == 512
        assert bucket_m(99999) == 512


# --------------------------------------------------------------------------- #
# TuneKey
# --------------------------------------------------------------------------- #
class TestTuneKey:
    def test_key_structure_deterministic(self):
        """Same inputs must produce the exact same key (hashable, eq-comparable)."""
        k1 = TuneKey(
            gpu="NVIDIA_A10", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16"
        )
        k2 = TuneKey(
            gpu="NVIDIA_A10", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16"
        )
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_different_gpu_not_equal(self):
        k1 = TuneKey(
            gpu="NVIDIA_A10", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16"
        )
        k2 = TuneKey(
            gpu="NVIDIA_H100", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16"
        )
        assert k1 != k2

    def test_build_applies_bucketing(self):
        key = TuneKey.build("fused_moe", m=5, n=4096, k=11008, dtype="fp16", gpu="NVIDIA_A10")
        assert key.shape_bucket == "M16_N4096_K11008"

    def test_to_dict_roundtrip(self):
        k = TuneKey(
            gpu="NVIDIA_A10", op="w4a16_matmul", shape_bucket="M32_N3584_K18944", dtype="int4"
        )
        assert TuneKey.from_dict(k.to_dict()) == k

    def test_make_shape_bucket_format(self):
        assert make_shape_bucket(4, 4096, 11008) == "M16_N4096_K11008"
        assert make_shape_bucket(100, 1024, 512) == "M128_N1024_K512"

    def test_normalize_gpu_name(self):
        assert normalize_gpu_name("NVIDIA A10") == "NVIDIA_A10"
        assert normalize_gpu_name("  NVIDIA H100 SXM  ") == "NVIDIA_H100_SXM"


# --------------------------------------------------------------------------- #
# ConfigStore
# --------------------------------------------------------------------------- #
class TestConfigStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> ConfigStore:
        return ConfigStore(cache_dir=tmp_path)

    def test_put_get_roundtrip(self, store: ConfigStore):
        key = TuneKey(
            gpu="NVIDIA_A10", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16"
        )
        config = {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
        store.put(key, config, latency_us=42.5)

        got = store.get(key)
        assert got == config

    def test_miss_returns_none(self, store: ConfigStore):
        key = TuneKey(
            gpu="NVIDIA_A10", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16"
        )
        assert store.get(key) is None

    def test_overwrite_keeps_latest(self, store: ConfigStore):
        key = TuneKey(
            gpu="NVIDIA_A10", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16"
        )
        store.put(key, {"BLOCK_M": 16}, latency_us=50.0)
        store.put(key, {"BLOCK_M": 32}, latency_us=40.0)
        assert store.get(key) == {"BLOCK_M": 32}

    def test_load_all(self, store: ConfigStore):
        k1 = TuneKey(
            gpu="NVIDIA_A10", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16"
        )
        k2 = TuneKey(
            gpu="NVIDIA_A10", op="fused_moe", shape_bucket="M32_N4096_K11008", dtype="fp16"
        )
        store.put(k1, {"BLOCK_M": 16}, latency_us=42.0)
        store.put(k2, {"BLOCK_M": 32}, latency_us=38.0)

        all_entries = store.load_all("fused_moe")
        assert len(all_entries) == 2
        assert k1 in all_entries
        assert k2 in all_entries

    def test_different_ops_separate_files(self, store: ConfigStore):
        k1 = TuneKey(
            gpu="NVIDIA_A10", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16"
        )
        k2 = TuneKey(
            gpu="NVIDIA_A10", op="flash_attn_nopad", shape_bucket="M64_N128_K64", dtype="fp16"
        )
        store.put(k1, {"BLOCK_M": 16}, latency_us=42.0)
        store.put(k2, {"BLOCK_M_SIZE": 64}, latency_us=20.0)

        assert (store.cache_dir / "fused_moe.json").is_file()
        assert (store.cache_dir / "flash_attn_nopad.json").is_file()

    def test_json_format_stable(self, store: ConfigStore):
        """The JSON must be parseable by external tools (schema contract)."""
        key = TuneKey(
            gpu="NVIDIA_A10", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16"
        )
        config = {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3}
        store.put(key, config, latency_us=42.5)

        raw = json.loads((store.cache_dir / "fused_moe.json").read_text())
        assert raw["version"] == 1
        assert len(raw["entries"]) == 1
        entry = raw["entries"][0]
        assert entry["gpu"] == "NVIDIA_A10"
        assert entry["shape_bucket"] == "M16_N4096_K11008"
        assert entry["dtype"] == "fp16"
        assert entry["config"] == config
        assert entry["latency_us"] == 42.5
        assert "timestamp" in entry

    def test_persistence_across_instances(self, tmp_path: Path):
        """A new ConfigStore instance must load what the previous one wrote."""
        key = TuneKey(gpu="NVIDIA_A10", op="test_op", shape_bucket="M32_N1024_K512", dtype="fp16")
        config = {"BLOCK_M": 32, "BLOCK_N": 128}

        store1 = ConfigStore(cache_dir=tmp_path)
        store1.put(key, config, latency_us=10.0)

        store2 = ConfigStore(cache_dir=tmp_path)
        assert store2.get(key) == config


# --------------------------------------------------------------------------- #
# get_best_config (lookup module)
# --------------------------------------------------------------------------- #
class TestLookup:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path, monkeypatch):
        """Point the lookup module at a temporary store."""
        reset()
        monkeypatch.setenv("RAPID_LLM_AUTOTUNE_DIR", str(tmp_path))
        monkeypatch.setenv("RAPID_LLM_AUTOTUNE", "1")
        # Pre-populate the store
        store = ConfigStore(cache_dir=tmp_path)
        key = TuneKey(gpu="test_gpu", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16")
        store.put(key, {"BLOCK_M": 16, "BLOCK_N": 128}, latency_us=35.0)
        # Patch GPU detection so lookup uses "test_gpu"
        import rapid_llm.kernels.dispatcher.autotune.lookup as lk

        monkeypatch.setattr(lk, "_gpu_name", "test_gpu")
        monkeypatch.setattr(lk, "_store", ConfigStore(cache_dir=tmp_path))
        yield
        reset()

    def test_hit(self):
        result = get_best_config("fused_moe", m=4, n=4096, k=11008, dtype="fp16")
        assert result == {"BLOCK_M": 16, "BLOCK_N": 128}

    def test_miss(self):
        result = get_best_config("fused_moe", m=4, n=9999, k=11008, dtype="fp16")
        assert result is None

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("RAPID_LLM_AUTOTUNE", "0")
        result = get_best_config("fused_moe", m=4, n=4096, k=11008, dtype="fp16")
        assert result is None


# --------------------------------------------------------------------------- #
# default cache dir: rename compatibility
# --------------------------------------------------------------------------- #


def _fake_home(monkeypatch, config_store, home: Path) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))


class TestDefaultCacheDir:
    """``_default_cache_dir`` honours the pre-rename location.

    Explicit env wins; the legacy ``LITE_LLAMA_`` spelling still works
    (with a deprecation warning); the legacy ``~/.cache/lite_llama``
    directory is only used when the new one does not exist yet, so
    measured configs survive the rename without a migration step.
    """

    @pytest.fixture(autouse=True)
    def _fresh_warn_ledger(self, monkeypatch):
        from rapid_llm.utils import env_compat

        monkeypatch.setattr(env_compat, "_warned", set())

    def test_explicit_new_env_wins(self, monkeypatch, tmp_path):
        from rapid_llm.kernels.dispatcher.autotune import config_store

        monkeypatch.setenv("RAPID_LLM_AUTOTUNE_DIR", str(tmp_path))
        assert config_store._default_cache_dir() == tmp_path

    def test_legacy_env_accepted(self, monkeypatch, tmp_path):
        from rapid_llm.kernels.dispatcher.autotune import config_store

        monkeypatch.delenv("RAPID_LLM_AUTOTUNE_DIR", raising=False)
        monkeypatch.setenv("LITE_LLAMA_AUTOTUNE_DIR", str(tmp_path))
        with pytest.warns(DeprecationWarning, match="LITE_LLAMA_AUTOTUNE_DIR"):
            assert config_store._default_cache_dir() == tmp_path

    def test_legacy_dir_used_when_only_it_exists(self, monkeypatch, tmp_path):
        from rapid_llm.kernels.dispatcher.autotune import config_store

        home = tmp_path / "home"
        legacy = home / ".cache" / "lite_llama" / "autotune"
        legacy.mkdir(parents=True)
        _fake_home(monkeypatch, config_store, home)
        monkeypatch.delenv("RAPID_LLM_AUTOTUNE_DIR", raising=False)
        monkeypatch.delenv("LITE_LLAMA_AUTOTUNE_DIR", raising=False)
        assert config_store._default_cache_dir() == legacy

    def test_new_dir_wins_when_both_exist(self, monkeypatch, tmp_path):
        from rapid_llm.kernels.dispatcher.autotune import config_store

        home = tmp_path / "home"
        (home / ".cache" / "lite_llama" / "autotune").mkdir(parents=True)
        new = home / ".cache" / "rapid_llm" / "autotune"
        new.mkdir(parents=True)
        _fake_home(monkeypatch, config_store, home)
        monkeypatch.delenv("RAPID_LLM_AUTOTUNE_DIR", raising=False)
        monkeypatch.delenv("LITE_LLAMA_AUTOTUNE_DIR", raising=False)
        assert config_store._default_cache_dir() == new

    def test_new_default_when_neither_exists(self, monkeypatch, tmp_path):
        from rapid_llm.kernels.dispatcher.autotune import config_store

        home = tmp_path / "home"
        home.mkdir()
        _fake_home(monkeypatch, config_store, home)
        monkeypatch.delenv("RAPID_LLM_AUTOTUNE_DIR", raising=False)
        monkeypatch.delenv("LITE_LLAMA_AUTOTUNE_DIR", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert config_store._default_cache_dir() == (home / ".cache" / "rapid_llm" / "autotune")
