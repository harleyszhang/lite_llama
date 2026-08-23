"""viz.structure: export model architecture as a text tree (L1).

Walks a torch.nn.Module tree and renders it as an indented text tree showing
layer types, parameter shapes, and quantisation status.

Usage:
    tree_str = export_structure_tree(model)
    print_structure_tree(model)
"""

from __future__ import annotations

import torch.nn as nn


def _format_params(module: nn.Module) -> str:
    """Summarise parameters of a leaf module."""
    params = list(module.parameters(recurse=False))
    if not params:
        return ""
    total = sum(p.numel() for p in params)
    dtypes = {str(p.dtype).replace("torch.", "") for p in params}
    return f" [{total:,} params, {'/'.join(sorted(dtypes))}]"


def _tree_lines(module: nn.Module, prefix: str = "", name: str = "model") -> list[str]:
    """Recursively build tree lines."""
    lines: list[str] = []
    type_name = type(module).__name__
    param_info = _format_params(module)
    lines.append(f"{prefix}{name}: {type_name}{param_info}")

    children = list(module.named_children())
    for i, (child_name, child) in enumerate(children):
        is_last = i == len(children) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        child_lines = _tree_lines(child, prefix=prefix + extension, name=child_name)
        # Replace first line's prefix with the connector
        first = f"{prefix}{connector}{child_name}: {type(child).__name__}{_format_params(child)}"
        lines.append(first)
        lines.extend(child_lines[1:])  # skip the redundant first line from recursion

    return lines


def export_structure_tree(model: nn.Module, max_depth: int = 4) -> str:
    """Export the model structure as an indented text tree.

    Args:
        model: The PyTorch model.
        max_depth: Maximum nesting depth to display.

    Returns:
        Multi-line string of the tree.
    """
    lines: list[str] = []
    _build_tree(model, lines, prefix="", depth=0, max_depth=max_depth, name="model")
    return "\n".join(lines)


def _build_tree(module: nn.Module, lines: list[str], prefix: str,
                depth: int, max_depth: int, name: str) -> None:
    """Recursively build the tree into lines list."""
    type_name = type(module).__name__
    param_info = _format_params(module)
    lines.append(f"{prefix}{name}: {type_name}{param_info}")

    if depth >= max_depth:
        children = list(module.named_children())
        if children:
            lines.append(f"{prefix}    ... ({len(children)} children)")
        return

    children = list(module.named_children())
    for i, (child_name, child) in enumerate(children):
        is_last = i == len(children) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        lines.append(f"{prefix}{connector}{child_name}: {type(child).__name__}{_format_params(child)}")
        # Recurse into grandchildren
        grandchildren = list(child.named_children())
        for j, (gc_name, gc) in enumerate(grandchildren):
            gc_is_last = j == len(grandchildren) - 1
            gc_connector = "└── " if gc_is_last else "├── "
            gc_extension = "    " if gc_is_last else "│   "
            _build_tree(gc, lines, prefix + extension + gc_extension,
                       depth + 2, max_depth, gc_name)


def print_structure_tree(model: nn.Module, max_depth: int = 4) -> None:
    """Print the model structure tree to stdout."""
    print(export_structure_tree(model, max_depth=max_depth))
