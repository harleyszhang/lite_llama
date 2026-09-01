"""Dump per-file survey: current module docstring + top-level symbols."""
import ast
import os
import re
import sys

BASE = "/home/honggao/projects/lite_llama"
ROOTS = ["lite_llama", "examples", "tests", "benchmarks"]
pattern = re.compile(sys.argv[1])
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 16


def sym_info(tree):
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [getattr(b, "id", getattr(b, "attr", "?")) for b in node.bases]
            d = (ast.get_docstring(node) or "").strip().splitlines()
            tag = d[0][:100] if d else ""
            out.append(f"CLASS {node.name}({','.join(bases)}) :: {tag}")
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names = [a.arg for a in m.args.args if a.arg != "self"]
                    out.append(f"  - {m.name}({', '.join(names[:7])})")
                elif isinstance(m, ast.ClassDef):
                    out.append(f"  + class {m.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [a.arg for a in node.args.args]
            out.append(f"def {node.name}({', '.join(names[:7])})")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.append(f"VAR {t.id}")
    return out


for root in ROOTS:
    base_dir = os.path.join(BASE, root)
    if not os.path.isdir(base_dir):
        continue
    for dp, dn, fn in os.walk(base_dir):
        dn[:] = sorted(d for d in dn if d != "__pycache__" and not d.startswith("."))
        for f in sorted(fn):
            if not f.endswith(".py"):
                continue
            full = os.path.join(dp, f)
            rel = os.path.relpath(full, BASE)
            if not pattern.search(rel):
                continue
            src = open(full, encoding="utf-8").read()
            lines = src.splitlines()
            tree = ast.parse(src)
            n = tree.body[0] if tree.body else None
            doc = ""
            span = 0
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                doc = n.value.value
                span = n.end_lineno - n.lineno + 1
            print(f"### {rel} | total={len(lines)} | doc={span}L/{len(doc)}c | usage={'Usage:' in doc}")
            if doc:
                d = doc.splitlines()
                shown = d if len(d) <= cap else d[:cap] + [f"...(+{len(d)-cap}L)"]
                for ln in shown:
                    print("DOC| " + ln)
            for s in sym_info(tree):
                print(s)
            print()
