"""Tests for :mod:`rapid_llm.utils.env_compat` — the rename bridge.

``RAPID_LLM_*`` wins; the pre-rename ``LITE_LLAMA_*`` spelling still works
but warns once. The warning-once contract and the "no fallback for
foreign prefixes" rule are what keep deployments quiet and names honest.

Usage:
    pytest tests/utils/test_env_compat.py
"""

from __future__ import annotations

import pytest

from rapid_llm.utils import env_compat
from rapid_llm.utils.env_compat import getenv


@pytest.fixture(autouse=True)
def _clean_warned(monkeypatch):
    """Each test starts with a fresh warn-once ledger."""
    monkeypatch.setattr(env_compat, "_warned", set())


def test_new_name_wins(monkeypatch):
    monkeypatch.setenv("RAPID_LLM_PIPELINE", "1")
    monkeypatch.setenv("LITE_LLAMA_PIPELINE", "0")
    assert getenv("RAPID_LLM_PIPELINE", "0") == "1"


def test_legacy_fallback(monkeypatch):
    monkeypatch.setenv("LITE_LLAMA_OVERLAP", "0")
    assert getenv("RAPID_LLM_OVERLAP", "1") == "0"


def test_default_when_neither_set(monkeypatch):
    monkeypatch.delenv("RAPID_LLM_METRICS", raising=False)
    monkeypatch.delenv("LITE_LLAMA_METRICS", raising=False)
    assert getenv("RAPID_LLM_METRICS", "1") == "1"
    assert getenv("RAPID_LLM_METRICS") is None


def test_legacy_warns_once(monkeypatch):
    monkeypatch.setenv("LITE_LLAMA_KERNEL_TRACE", "1")
    with pytest.warns(DeprecationWarning, match="LITE_LLAMA_KERNEL_TRACE"):
        assert getenv("RAPID_LLM_KERNEL_TRACE") == "1"
    # Second read: same value, no new warning.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert getenv("RAPID_LLM_KERNEL_TRACE") == "1"


def test_new_name_silent(monkeypatch):
    monkeypatch.setenv("RAPID_LLM_PIPELINE", "1")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert getenv("RAPID_LLM_PIPELINE") == "1"


def test_no_fallback_for_foreign_prefix(monkeypatch):
    monkeypatch.setenv("LITE_LLAMA_UNRELATED", "x")
    assert getenv("UNRELATED_VAR", "d") == "d"


def test_empty_legacy_value_is_honoured(monkeypatch):
    monkeypatch.setenv("LITE_LLAMA_FORCE_BACKEND", "")
    assert getenv("RAPID_LLM_FORCE_BACKEND", "triton") == ""
