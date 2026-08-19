"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _reset_torch_state():
    """Reset seeded state and clear the CUDA cache between tests."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _skip_when_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test", allow_module_level=False)


@pytest.fixture
def cuda_available():
    """Skip the test unless CUDA is available."""
    _skip_when_no_cuda()
    return True
