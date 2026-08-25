"""Profiling tools: model structure tree and memory budget table.

Exports readable representations of model architecture and memory allocation
for debugging, documentation, and capacity planning.

Usage:
    from lite_llama.tools.profiling import print_structure_tree, print_memory_budget
    print_structure_tree(model)
    print_memory_budget(num_layers=28, hidden_size=1024, ...)
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
