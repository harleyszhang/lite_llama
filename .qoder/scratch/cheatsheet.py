"""Compact cheat sheet: one entry per .py file with docstring state and symbols."""
import ast
from pathlib import Path

BASE = Path("/home/honggao/projects/lite_llama")
DIRS = ["lite_llama", "examples", "tests", "benchmarks"]
SKIP = {
    "lite_llama/utils/logger.py",
    "lite_llama/utils/path_utils.py",
    "lite_llama/utils/image_process.py",
    "lite_llama/executor/__init__.py",
    "lite_llama/executor/kv_cache_manager.py",
}

for d in DIRS:
    for p in sorted((BASE / d).rglob("*.py")):
        rel = p.relative_to(BASE).as_posix()
        if rel in SKIP or "__pycache__" in p.parts:
            continue
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception as e:  # noqa: BLE001
            print(f"### {rel} PARSE_ERROR {e}")
            continue
        doc = ast.get_docstring(tree) or ""
        first = doc.strip().splitlines()[0] if doc.strip() else "<NO_DOC>"
        has_usage = "usage" in doc.lower()
        syms = []
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                bases = ", ".join(
                    ast.unparse(b) if isinstance(b, ast.expr) else "?" for b in n.bases
                )
                syms.append(f"class {n.name}({bases})")
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in n.args.args]
                syms.append(f"def {n.name}({', '.join(args)})")
            elif isinstance(n, ast.Assign) and n.targets:
                t = n.targets[0]
                if isinstance(t, ast.Name):
                    syms.append(f"VAR {t.id} = ...")
            elif isinstance(n, ast.If) and isinstance(n.test, ast.Name):
                syms.append(f"IF {ast.unparse(n.test)}")
        sym_s = " | ".join(syms[:12])
        if len(syms) > 12:
            sym_s += f" | +{len(syms) - 12} more"
        print(f"### {rel}")
        print(f"doc[{len(doc.strip().splitlines())}L u={int(has_usage)}]: {first[:110]}")
        print(f"sym: {sym_s[:380]}")
