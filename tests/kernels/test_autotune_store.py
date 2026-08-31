"""Tests for the autotune config store — pure CPU, no GPU required.

Validates the key structure, bucket logic, JSON persistence round-trip,
and miss-returns-None behaviour. These tests pin the format that v0.6
perf_key will depend on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lite_llama.kernels.autotune import (
    ConfigStore,
    TuneKey,
    bucket_m,
    get_best_config,
    make_shape_bucket,
    normalize_gpu_name,
    reset,
)

# The file lives under tests/kernels/ (it pins the kernels.autotune store
# contract) but never touches Triton or a GPU, so it opts out of the
# directory-level gpu mark and runs on CPU-only machines.
pytestmark = pytest.mark.cpu


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
# _flush failure handling
# --------------------------------------------------------------------------- #
class TestFlushFailure:
    """A failed atomic rename must propagate the original error, clean up the
    temp file, and close the descriptor (a masked error or a leaked fd leaves
    the store in a state the next put() cannot reason about)."""

    def test_replace_failure_propagates_unmasked(self, tmp_path: Path, monkeypatch):
        store = ConfigStore(cache_dir=tmp_path)
        key = TuneKey(gpu="g", op="op", shape_bucket="M16_N1_K1", dtype="fp16")

        closes = 0
        real_close = os.close

        def spy_close(fd):
            nonlocal closes
            closes += 1
            real_close(fd)

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        monkeypatch.setattr(os, "close", spy_close)

        with pytest.raises(OSError, match="disk full"):
            store.put(key, {"BLOCK_M": 16}, latency_us=1.0)

        assert closes == 1  # the mkstemp descriptor was closed, not leaked
        assert list(tmp_path.iterdir()) == []  # neither target nor .tmp remain


# --------------------------------------------------------------------------- #
# get_best_config (lookup module)
# --------------------------------------------------------------------------- #
class TestLookup:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path, monkeypatch):
        """Point the lookup module at a temporary store."""
        reset()
        monkeypatch.setenv("LITE_LLAMA_AUTOTUNE_DIR", str(tmp_path))
        monkeypatch.setenv("LITE_LLAMA_AUTOTUNE", "1")
        # Pre-populate the store
        store = ConfigStore(cache_dir=tmp_path)
        key = TuneKey(gpu="test_gpu", op="fused_moe", shape_bucket="M16_N4096_K11008", dtype="fp16")
        store.put(key, {"BLOCK_M": 16, "BLOCK_N": 128}, latency_us=35.0)
        # Patch GPU detection so lookup uses "test_gpu"
        import lite_llama.kernels.autotune.lookup as lk

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
        monkeypatch.setenv("LITE_LLAMA_AUTOTUNE", "0")
        result = get_best_config("fused_moe", m=4, n=4096, k=11008, dtype="fp16")
        assert result is None
