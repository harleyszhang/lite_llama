"""Tests for tools.profiling.structure — pure CPU tree rendering.

The renderer was rewritten once already because the old recursion misplaced
grandchildren (double prefix offset, skipped depth levels); these tests pin
the exact line format so that cannot regress.
"""

from __future__ import annotations

import torch.nn as nn

from lite_llama.tools.profiling.structure import export_structure_tree


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.act = nn.ReLU()


class _Outer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.block = _Inner()
        self.head = nn.Linear(4, 5, bias=False)


def test_tree_renders_connectors_and_continuation_lines():
    tree = export_structure_tree(_Outer())
    assert tree.splitlines() == [
        "model: _Outer",
        "├── embed: Embedding [32 params, float32]",
        "├── block: _Inner",
        "│   ├── proj: Linear [16 params, float32]",
        "│   └── act: ReLU",
        "└── head: Linear [20 params, float32]",
    ]


def test_max_depth_truncates_with_a_child_count():
    tree = export_structure_tree(_Outer(), max_depth=1)
    assert tree.splitlines() == [
        "model: _Outer",
        "├── embed: Embedding [32 params, float32]",
        "├── block: _Inner",
        "│       ... (2 children)",
        "└── head: Linear [20 params, float32]",
    ]


def test_max_depth_zero_lists_only_the_root_and_a_summary():
    tree = export_structure_tree(_Outer(), max_depth=0)
    assert tree.splitlines() == [
        "model: _Outer",
        "    ... (3 children)",
    ]


def test_a_leaf_model_is_a_single_line():
    assert export_structure_tree(nn.ReLU()) == "model: ReLU"
