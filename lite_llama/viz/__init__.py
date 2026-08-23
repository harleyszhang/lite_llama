"""Visualization tools: structure tree and memory budget table.

Exports readable representations of model architecture and memory allocation
for debugging and documentation.

Usage:
    from lite_llama.viz import print_structure_tree, print_memory_budget
    print_structure_tree(model)
    print_memory_budget(config, kv_cache_manager)
"""

from .structure import print_structure_tree, export_structure_tree
from .memory import print_memory_budget, export_memory_budget

__all__ = [
    "export_memory_budget",
    "export_structure_tree",
    "print_memory_budget",
    "print_structure_tree",
]
