.PHONY: help install install-dev lint format test test-cpu test-gpu test-fast \
        test-weights test-golden test-eval golden-update coverage test-cli clean

PYTHON ?= python
UV     ?= uv

# Checkpoint used by the weights-gated tier. Override to test another model:
#   make test-weights MODEL_DIR=my_weight/Qwen3-0.6B
MODEL_DIR ?= my_weight/Qwen2.5-0.5B

# Eval configs the accuracy tier runs. Override for the full sweep:
#   make test-eval EVAL_CONFIGS=models-all.txt
EVAL_CONFIGS ?= models-small.txt

help:
	@echo "Setup:"
	@echo "  install       Install lite_llama in editable mode"
	@echo "  install-dev   Install with dev dependencies and register pre-commit hooks"
	@echo "  lint          ruff check + ruff format --check"
	@echo "  format        ruff --fix + ruff format"
	@echo ""
	@echo "Tests (tiers are selected by marker; each auto-skips what the machine lacks):"
	@echo "  test          Everything available on this machine"
	@echo "  test-cpu      No GPU, no checkpoint  — the tier CI runs on every PR"
	@echo "  test-gpu      Triton kernels vs torch references (needs CUDA)"
	@echo "  test-fast     Everything except the slow golden tier"
	@echo "  test-weights  End-to-end generation (needs CUDA + a converted checkpoint)"
	@echo "  test-golden   Byte-exact output regression against the recorded baseline"
	@echo "  test-eval     GSM8K accuracy against the thresholds in tests/evals/configs"
	@echo "  coverage      test-cpu with an HTML coverage report in htmlcov/"
	@echo "  test-cli      CLI smoke test over every checkpoint in my_weight/"
	@echo ""
	@echo "Maintenance:"
	@echo "  golden-update Re-record the golden baseline for MODEL_DIR (review the diff!)"
	@echo "  clean         Remove caches and build artifacts"

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

# --------------------------------------------------------------------------- #
# Test tiers
#
# Gating lives in tests/conftest.py, not here: `gpu` skips without CUDA and
# `weights` skips without a checkpoint. So every target below is safe to run
# anywhere — it reports what it skipped and why instead of failing.
# --------------------------------------------------------------------------- #
test:
	$(PYTHON) -m pytest

test-cpu:
	$(PYTHON) -m pytest -m "not gpu and not weights"

test-gpu:
	$(PYTHON) -m pytest -m gpu

test-fast:
	$(PYTHON) -m pytest -m "not slow"

test-weights:
	LITE_LLAMA_TEST_MODEL_DIR=$(MODEL_DIR) $(PYTHON) -m pytest -m weights

test-golden:
	LITE_LLAMA_TEST_MODEL_DIR=$(MODEL_DIR) $(PYTHON) -m pytest tests/golden

# Accuracy tier. Each config names its own checkpoint and skips itself when that
# checkpoint is absent, so MODEL_DIR does not apply here — pick configs instead.
test-eval:
	$(PYTHON) -m pytest -s -v tests/evals --config-list-file=$(EVAL_CONFIGS)

coverage:
	$(PYTHON) -m pytest -m "not gpu and not weights" \
		--cov=lite_llama --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

# Re-records the byte-exact baseline. The diff is the *output of the model*, so
# read it before committing: an unexpected change there is the regression this
# tier exists to catch, not noise to be accepted.
golden-update:
	$(PYTHON) scripts/golden_tokens.py --model-dir $(MODEL_DIR) \
		--save tests/golden/data/$(notdir $(MODEL_DIR)).json

# Exercises the real CLI against every checkpoint in my_weight/ and asserts the
# generated text is non-empty and not garbled. Run before shipping any change to
# the model, executor or engine layers.
test-cli:
	bash scripts/cli_smoke.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
