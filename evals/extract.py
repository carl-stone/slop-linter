"""Extract a repository's Python functions for labeling, without importing it."""

from __future__ import annotations

import ast
import io
import json
import os
from pathlib import Path
import textwrap
import tokenize

import slop_linter as lint


EXCLUDED_DIRECTORIES = {"__pycache__", "node_modules", "build", "dist", "vendor", "data", "outputs"}


def repository_functions(root: Path) -> list[dict]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise lint.LintError(f"not a repository directory: {root}")
    records = []
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories
                                if not name.startswith(".") and name not in EXCLUDED_DIRECTORIES
                                and name != "venv" and not (Path(folder) / name).is_symlink())
        for filename in sorted(files):
            path = Path(folder) / filename
            if not filename.endswith(".py") or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            blob = path.read_bytes()
            functions = {function.line: function for function in lint.extract_functions(relative, blob)}
            encoding, _ = tokenize.detect_encoding(io.BytesIO(blob).readline)
            tree = ast.parse(blob.decode(encoding), filename=relative)

            def visit(node, parents=()):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qualname = ".".join((*parents, child.name))
                        body = [statement for statement in child.body
                                if not (isinstance(statement, ast.Expr)
                                        and isinstance(statement.value, ast.Constant)
                                        and isinstance(statement.value.value, str))]
                        records.append({"id": f"{relative}:{child.lineno}:{qualname}",
                                        "path": relative, "qualname": qualname,
                                        "line": child.lineno, "end_line": child.end_lineno,
                                        "source": textwrap.dedent(functions[child.lineno].source),
                                        "body_statements": len(body),
                                        "lines": child.end_lineno - child.lineno + 1,
                                        "is_test": "tests" in path.relative_to(root).parts})
                        visit(child, (*parents, child.name))
                    elif isinstance(child, ast.ClassDef):
                        visit(child, (*parents, child.name))
                    else:
                        visit(child, parents)

            visit(tree)
    return sorted(records, key=lambda row: (row["path"], row["line"]))


def extract_repository(root: Path, output: Path) -> int:
    records = repository_functions(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)
