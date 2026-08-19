.PHONY: help install install-dev lint format test test-cpu test-gpu test-cli clean

PYTHON ?= python
UV     ?= uv

help:
	@echo "Available targets:"
	@echo "  install      Install lite_llama in editable mode"
	@echo "  install-dev  Install with dev dependencies and register pre-commit hooks"
	@echo "  lint         Run ruff check + ruff format --check"
	@echo "  format       Apply ruff formatting and autofixes"
	@echo "  test         Run the whole test suite"
	@echo "  test-cpu     Run only tests that do not need a GPU"
	@echo "  test-gpu     Run only the GPU-marked tests"
	@echo "  test-cli     End-to-end CLI smoke test over every converted checkpoint"
	@echo "  clean        Remove caches and build artifacts"

install:
	$(UV) pip install -e .

install-dev:
	$(UV) pip install -e . --group dev
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	$(PYTHON) -m pytest

test-cpu:
	$(PYTHON) -m pytest -m "not gpu and not weights"

test-gpu:
	$(PYTHON) -m pytest -m gpu

# Exercises the real CLI against every checkpoint in my_weight/ and asserts the
# generated text is non-empty and not garbled. Run this before shipping any
# change to the model, executor or engine layers.
test-cli:
	bash scripts/cli_smoke.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
