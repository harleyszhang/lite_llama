"""Profiling tools: model structure tree and memory budget table.

Re-exports both halves — ``print_structure_tree`` /
``export_structure_tree`` and ``compute_memory_budget`` /
``print_memory_budget`` — one import for the whole toolkit.

Usage:
    from rapid_llm.tools.profiling import print_memory_budget
"""

from .memory import (
    MemoryBudget,
    ModelShape,
    compute_memory_budget,
    export_memory_budget,
    print_memory_budget,
)
from .structure import export_structure_tree, print_structure_tree

__all__ = [
    "MemoryBudget",
    "ModelShape",
    "compute_memory_budget",
    "export_memory_budget",
    "export_structure_tree",
    "print_memory_budget",
    "print_structure_tree",
]
