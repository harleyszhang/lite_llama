"""Tests for the profiling tools: the structure tree and the memory budget.

Tree rendering (branch glyphs, hidden-depth reporting, per-node
parameter accounting, dtype naming) and the static memory budget are
checked with hand-built ``nn.Module`` trees — CPU only.

Usage:
    pytest tests/tools/test_profiling.py
"""

import torch
import torch.nn as nn

from rapid_llm.tools.profiling import (
    ModelShape,
    compute_memory_budget,
    export_memory_budget,
    export_structure_tree,
)


def _model() -> nn.Module:
    """A three-level module: root -> (embed, layers -> 2 blocks -> leaves), norm."""

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Linear(8, 8, bias=False)
            self.mlp = nn.Linear(8, 16, bias=False)

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(32, 8)
            self.layers = nn.ModuleList([Block(), Block()])
            self.norm = nn.LayerNorm(8)

    return Tiny()


# --------------------------------------------------------------------------- #
# Structure tree
# --------------------------------------------------------------------------- #
def test_every_node_below_the_root_carries_a_branch_glyph():
    """The bug this catches: a level rendered with indentation but no connector.

    Indentation alone still lines up, so a subtree missing its glyphs looks like a
    tree until you try to tell two sibling branches apart.
    """
    lines = export_structure_tree(_model(), max_depth=4).splitlines()
    assert lines[0] == "model: Tiny"
    for line in lines[1:]:
        assert "├── " in line or "└── " in line, line


def test_the_last_child_closes_its_branch_and_the_others_do_not():
    lines = export_structure_tree(_model(), max_depth=4).splitlines()
    top = [line for line in lines if line.startswith(("├── ", "└── "))]
    assert [line.split(":")[0] for line in top] == [
        "├── embed",
        "├── layers",
        "└── norm",
    ]


def test_a_bar_continues_under_a_node_whose_siblings_remain():
    """`layers` has `norm` after it, so its subtree hangs under a bar, not a gap."""
    lines = export_structure_tree(_model(), max_depth=4).splitlines()
    blocks = [line for line in lines if line.endswith(": Block")]
    assert blocks == ["│   ├── 0: Block", "│   └── 1: Block"]
    # And the leaves of the *last* block sit under that block's gap, not its bar.
    assert "│       └── mlp: Linear [128 params, float32]" in lines


def test_depth_budget_reports_what_it_hid_instead_of_dropping_it():
    """Truncation has to be visible, or a tree silently lies about being complete."""
    lines = export_structure_tree(_model(), max_depth=1).splitlines()
    assert "│   ... (2 children)" in lines
    assert not any(": Block" in line for line in lines)


def test_a_node_counts_only_the_parameters_it_owns():
    """Recursive counting would report the root's total on every ancestor line."""
    lines = export_structure_tree(_model(), max_depth=4).splitlines()
    assert lines[0] == "model: Tiny"  # no param summary: Tiny owns none directly
    assert "├── embed: Embedding [256 params, float32]" in lines


def test_mixed_dtypes_are_all_named():
    layer = nn.Linear(4, 4)
    layer.bias.data = layer.bias.data.to(torch.float16)
    assert "[20 params, float16/float32]" in export_structure_tree(layer)


# --------------------------------------------------------------------------- #
# Memory budget
# --------------------------------------------------------------------------- #
def _shape(**overrides) -> ModelShape:
    fields = {
        "num_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
        "vocab_size": 1000,
        "num_kv_blocks": 100,
    }
    return ModelShape(**{**fields, **overrides})


def test_weights_match_a_hand_computed_parameter_count():
    shape = _shape()
    embed = 1000 * 64
    per_layer = 64 * (4 + 2 * 2) * 16 + 4 * 16 * 64 + 64 * 128 * 3 + 64 * 2
    expected = (embed + 2 * per_layer + embed) * 2  # untied head, fp16
    assert compute_memory_budget(shape).model_weights_bytes == expected


def test_tying_the_head_removes_exactly_one_embedding_matrix():
    """A third of a small model's weights, so guessing here is not good enough."""
    untied = compute_memory_budget(_shape()).model_weights_bytes
    tied = compute_memory_budget(_shape(tie_word_embeddings=True)).model_weights_bytes
    assert untied - tied == 1000 * 64 * 2


def test_kv_cache_scales_with_capacity_heads_and_dtype():
    base = compute_memory_budget(_shape()).kv_cache_bytes
    assert base == 2 * 2 * 2 * 16 * 100 * 2
    assert compute_memory_budget(_shape(num_kv_blocks=200)).kv_cache_bytes == 2 * base
    assert compute_memory_budget(_shape(kv_dtype="fp8")).kv_cache_bytes == base // 2


def test_an_unknown_dtype_falls_back_to_two_bytes_rather_than_raising():
    """Budgeting is a planning aid; an unfamiliar dtype name should not stop it."""
    assert compute_memory_budget(_shape(kv_dtype="nf4")).kv_cache_bytes == (
        compute_memory_budget(_shape(kv_dtype="fp16")).kv_cache_bytes
    )


def test_total_is_the_sum_of_the_parts_it_prints():
    budget = compute_memory_budget(_shape())
    assert budget.total_bytes == (
        budget.model_weights_bytes
        + budget.kv_cache_bytes
        + budget.activation_bytes
        + budget.cuda_graph_bytes
    )


def test_the_table_percentages_add_up_to_a_hundred():
    """Rounded independently, so this is the check that the rows share one total."""
    table = export_memory_budget(**vars(_shape()))
    percentages = [
        float(line.rsplit("|", 2)[1].strip().rstrip("%"))
        for line in table.splitlines()[2:-1]  # skip header, separator, total
    ]
    assert abs(sum(percentages) - 100.0) < 0.25


def test_the_table_names_the_kv_dtype_it_was_given():
    assert "| KV Cache (fp8) |" in export_memory_budget(**vars(_shape(kv_dtype="fp8")))
