"""Device-independent policy tests for the public LLM constructor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("supports_cuda_graph", "requested", "expected"),
    [(True, True, True), (True, None, True), (False, True, False)],
)
def test_cuda_graph_policy_is_architecture_driven(
    monkeypatch, supports_cuda_graph: bool, requested: bool | None, expected: bool
):
    """Joining a TP group must not silently turn graph replay off."""
    import rapid_llm.engine.llm as llm_module

    captured: dict[str, object] = {}
    monkeypatch.setattr(llm_module, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        llm_module,
        "_resolve_spec",
        lambda _model: SimpleNamespace(
            supports_cuda_graph=supports_cuda_graph,
            is_multimodal=False,
        ),
    )

    def fake_engine_init(_self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(llm_module.LLMEngine, "__init__", fake_engine_init)
    llm_module.LLM("unused", tensor_parallel_size=2, use_cuda_graph=requested)

    assert captured["tensor_parallel_size"] == 2
    assert captured["use_cuda_graph"] is expected


def test_llm_rejects_unjoined_tensor_parallel_group(monkeypatch):
    import rapid_llm.engine.llm as llm_module

    monkeypatch.setattr(llm_module, "get_tensor_model_parallel_world_size", lambda: 1)
    with pytest.raises(ValueError, match="cannot start a tensor-parallel group"):
        llm_module.LLM("unused", tensor_parallel_size=2)
