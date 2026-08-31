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


def export_structure_tree(model: nn.Module, max_depth: int = 4) -> str:
    """Export the model structure as an indented text tree.

    Args:
        model: The PyTorch model.
        max_depth: Maximum nesting depth to display below the root.

    Returns:
        Multi-line string of the tree.
    """
    lines: list[str] = []
    _build_tree(
        model,
        lines,
        prefix="",
        depth=0,
        max_depth=max_depth,
        name="model",
        is_root=True,
        is_last=True,
    )
    return "\n".join(lines)


def _build_tree(
    module: nn.Module,
    lines: list[str],
    prefix: str,
    depth: int,
    max_depth: int,
    name: str,
    is_root: bool,
    is_last: bool,
) -> None:
    """Recursively build the tree into lines list.

    A node's connector (``├── ``/``└── ``) sits on its own line; children
    inherit ``prefix`` extended by a continuation vertical (or blank, for a
    last child). ``depth`` counts levels below the root, so ``max_depth=1``
    renders the root and its direct children only.
    """
    if not is_root:
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{name}: {type(module).__name__}{_format_params(module)}")
        prefix += "    " if is_last else "│   "
    else:
        lines.append(f"{name}: {type(module).__name__}{_format_params(module)}")

    children = list(module.named_children())
    if not children:
        return
    if depth >= max_depth:
        lines.append(f"{prefix}    ... ({len(children)} children)")
        return
    for i, (child_name, child) in enumerate(children):
        _build_tree(
            child,
            lines,
            prefix,
            depth + 1,
            max_depth,
            child_name,
            is_root=False,
            is_last=i == len(children) - 1,
        )


def print_structure_tree(model: nn.Module, max_depth: int = 4) -> None:
    """Print the model structure tree to stdout."""
    print(export_structure_tree(model, max_depth=max_depth))
