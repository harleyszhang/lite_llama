"""Tests for the autotune searcher — pure CPU, no GPU required.

Only the failure paths are CPU-testable (the timing path measures with CUDA
events). What is pinned here is the contract that a shape whose candidate
configs all fail raises instead of persisting a config that never ran.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lite_llama.kernels.autotune import ConfigStore
from lite_llama.kernels.autotune.searcher import AutotuneSearcher

# Same opt-out as test_autotune_store: kept under tests/kernels/ for module
# affinity, but the failure paths below never touch CUDA.
pytestmark = pytest.mark.cpu


def _always_fails(cfg: dict) -> None:
    raise RuntimeError("kernel launch failed")


class TestSearchFailure:
    def test_all_configs_failing_raises_instead_of_persisting(self, tmp_path: Path):
        store = ConfigStore(cache_dir=tmp_path)
        searcher = AutotuneSearcher(store, warmup=1, repeat=1)

        with pytest.raises(RuntimeError, match="no working config"):
            searcher.search(
                op="fused_moe",
                shape=(16, 64, 64),
                dtype="fp16",
                configs=[{"BLOCK_M": 16}, {"BLOCK_M": 32}],
                run_fn=_always_fails,
            )

        # Nothing was persisted: a later lookup must not receive a config that
        # never ran.
        assert store.load_all("fused_moe") == {}

    def test_empty_config_list_is_rejected(self, tmp_path: Path):
        searcher = AutotuneSearcher(ConfigStore(cache_dir=tmp_path), warmup=1, repeat=1)
        with pytest.raises(ValueError, match="must not be empty"):
            searcher.search(
                op="fused_moe",
                shape=(16, 64, 64),
                dtype="fp16",
                configs=[],
                run_fn=lambda cfg: None,
            )
