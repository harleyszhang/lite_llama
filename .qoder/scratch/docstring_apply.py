"""Apply new module docstrings from @@@FILE-delimited mapping files.

Each block in a mapping file is: a ``@@@FILE <relpath>`` header line, the
new docstring text (with its own triple quotes), and an ``@@@END`` line.
The docstring replaces (or inserts as) the module docstring; every
candidate file body is re-parsed with ast.parse before it is written, so
a mapping bug can never corrupt a source file.
"""
import ast
import sys
from pathlib import Path

BASE = Path("/home/honggao/projects/lite_llama")


def parse_blocks(paths):
    blocks = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for chunk in text.split("@@@FILE ")[1:]:
            head, _, body = chunk.partition("\n")
            idx = body.rfind("\n@@@END")
            if idx >= 0:
                body = body[:idx]
            else:
                body = body.rstrip()
                if body.endswith("@@@END"):
                    body = body[: -len("@@@END")].rstrip()
            blocks.append((head.strip(), body))
    return blocks


def doc_lines(body):
    lines = body.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return [ln + "\n" for ln in lines]


def apply_one(rel, body):
    path = BASE / rel
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    new_doc = doc_lines(body)
    n = tree.body[0] if tree.body else None
    is_doc = (
        n is not None
        and isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str)
    )
    lines = src.splitlines(keepends=True)
    if is_doc:
        start, end = n.lineno - 1, n.end_lineno
        lines[start:end] = new_doc
        after = start + len(new_doc)
        if after < len(lines) and lines[after].strip() != "":
            lines.insert(after, "\n")
    else:
        i = 0
        while i < len(lines) and (
            lines[i].lstrip().startswith("#") or not lines[i].strip()
        ):
            i += 1
        block = new_doc + (["\n"] if i < len(lines) else [])
        lines[i:i] = block
    new_src = "".join(lines)
    ast.parse(new_src)  # syntax gate: never write a file that no longer parses
    path.write_text(new_src, encoding="utf-8")
    return is_doc


def main():
    blocks = parse_blocks(sys.argv[1:])
    ok = patched = inserted = 0
    for rel, body in blocks:
        try:
            was_doc = apply_one(rel, body)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {rel}: {exc}")
            continue
        ok += 1
        patched += int(was_doc)
        inserted += int(not was_doc)
        span = len(doc_lines(body))
        if span > 11:
            print(f"WARN {rel}: {span} lines")
        over = [ln for ln in doc_lines(body) if len(ln.rstrip("\n")) > 88]
        if over:
            print(f"WARN {rel}: line >88 chars ({len(over)} lines)")
    print(f"done: {ok}/{len(blocks)} ok, {patched} replaced, {inserted} inserted")


if __name__ == "__main__":
    main()
