"""Accuracy evaluation harness.

The subpackage turns ``configs/*.yaml`` into test cases: dataset
acquisition (``dataset``), execution (``runner``) and scoring
(``gsm8k``) are shared by every benchmark.

Usage:
    pytest tests/evals/
"""
