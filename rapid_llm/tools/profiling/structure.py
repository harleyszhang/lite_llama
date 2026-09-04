"""Render a model as an indented text tree: layer types, params, dtypes.

:func:`export_structure_tree` walks the ``nn.Module`` tree once, counting
each node's own parameters and naming dtypes; depth budgeting reports
what it hid instead of silently dropping subtrees.

Usage:
    print_structure_tree(model, max_depth=3)
"""

from __future__ import annotations

import torch.nn as nn

#: Box-drawing pieces: a branch, the last branch, and the bar that continues under
#: a node whose siblings are still to come.
TEE, ELBOW, BAR, GAP = "├── ", "└── ", "│   ", "    "


def _format_params(module: nn.Module) -> str:
    """Summarise the parameters this module owns directly, ignoring its children."""
    params = list(module.parameters(recurse=False))
    if not params:
        return ""
    total = sum(p.numel() for p in params)
    dtypes = {str(p.dtype).replace("torch.", "") for p in params}
    return f" [{total:,} params, {'/'.join(sorted(dtypes))}]"


def _lines(
    module: nn.Module, name: str, prefix: str, connector: str, depth: int, max_depth: int
) -> list[str]:
    """Render one node, then its subtree while the depth budget lasts.

    Args:
        module: Node to render.
        name: Attribute name this node is bound to in its parent.
        prefix: Bars and gaps inherited from every ancestor, already assembled.
        connector: This node's own branch glyph; empty for the root.
        depth: Distance from the root, in nodes.
        max_depth: Last depth whose children are expanded.
    """
    lines = [f"{prefix}{connector}{name}: {type(module).__name__}{_format_params(module)}"]
    children = list(module.named_children())
    if not children:
        return lines

    # Children hang under this node, so they inherit its prefix plus either a bar
    # (siblings still to come below it) or a gap (this node closed its branch).
    below = prefix if not connector else prefix + (GAP if connector == ELBOW else BAR)
    if depth >= max_depth:
        return [*lines, f"{below}... ({len(children)} children)"]
    for index, (child_name, child) in enumerate(children):
        last = index == len(children) - 1
        lines += _lines(child, child_name, below, ELBOW if last else TEE, depth + 1, max_depth)
    return lines


def export_structure_tree(model: nn.Module, max_depth: int = 4) -> str:
    """Return the model structure as an indented text tree.

    Args:
        model: Module to walk.
        max_depth: Last depth whose children are expanded; deeper nodes report a count.

    Returns:
        Multi-line string, one node per line, root first.
    """
    return "\n".join(_lines(model, "model", "", "", 0, max_depth))


def print_structure_tree(model: nn.Module, max_depth: int = 4) -> None:
    """Print :func:`export_structure_tree` to stdout."""
    print(export_structure_tree(model, max_depth=max_depth))
