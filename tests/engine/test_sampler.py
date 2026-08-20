"""CPU-only tests for the Sampler + SamplingParams pair."""

from __future__ import annotations

import pytest
import torch

from lite_llama.engine.sampler import Sampler, SamplingParams, sample_top_p


def test_sampling_params_rejects_negative_temperature():
    with pytest.raises(ValueError):
        SamplingParams(temperature=-0.1)


def test_sampling_params_rejects_top_p_outside_range():
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=1.5)


def test_greedy_flag():
    assert SamplingParams(temperature=0.0).is_greedy
    assert not SamplingParams(temperature=0.7).is_greedy


def test_greedy_picks_argmax_over_last_position():
    sampler = Sampler()
    logits = torch.tensor([[[0.1, 0.9, 0.0], [0.5, 0.1, 0.4]]])  # last step: token 0
    result = sampler.sample(logits, SamplingParams(temperature=0.0))
    assert result.shape == (1, 1)
    assert result.item() == 0


def test_greedy_supports_two_dim_logits():
    sampler = Sampler()
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.5, 0.1, 0.4]])
    result = sampler.sample(logits, SamplingParams(temperature=0.0))
    assert result.squeeze(-1).tolist() == [1, 0]


def test_top_p_keeps_only_the_nucleus():
    """Top-p 0.1 must keep exactly one token (the dominant one)."""
    probs = torch.tensor([[0.7, 0.2, 0.1]])
    torch.manual_seed(0)
    for _ in range(10):
        token = sample_top_p(probs.clone(), top_p=0.1).item()
        assert token == 0


def test_sampled_temperature_stays_within_vocab():
    sampler = Sampler()
    logits = torch.randn(4, 100)
    tokens = sampler.sample(logits, SamplingParams(temperature=0.8, top_p=0.9))
    assert tokens.shape == (4, 1)
    assert tokens.min() >= 0
    assert tokens.max() < 100
