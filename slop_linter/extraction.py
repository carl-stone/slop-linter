"""Extract function source from Python files or the Git index without imports."""

from __future__ import annotations

import ast
import fnmatch
import io
import json
import os
from pathlib import Path
import subprocess
import tokenize

from .models import Function, LintError


DEFAULT_EXCLUDES = (".git", ".venv*", "venv", "__pycache__", "build", "dist")


def git(cwd: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(cwd), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LintError("could not run Git (is it installed?)") from exc
    if result.returncode:
        # Git's output may contain source or paths with terminal control characters.
        raise LintError(f"Git {args[0]} failed; check the repository and index")
    return result.stdout


def extract_functions(path: str, blob: bytes) -> list[Function]:
    label = json.dumps(path)
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(blob).readline)
        source = blob.decode(encoding)
        tree = ast.parse(source, filename=path)
    except (SyntaxError, UnicodeError, LookupError, ValueError, RecursionError) as exc:
        line = getattr(exc, "lineno", None)
        location = f"{label}:{line}" if line else label
        raise LintError(f"{location}: cannot parse staged Python source") from exc

    # Token positions locate '@' even for @(\n decorator\n), whose AST
    # decorator.lineno points inside the parentheses. Only NEWLINE ends a
    # logical statement; NL inside a multiline expression does not.
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    decorator_lines = []
    at_start = True
    try:
        for token in tokenize.generate_tokens(io.StringIO(normalized).readline):
            if token.type == tokenize.NEWLINE:
                at_start = True
            elif token.type not in (tokenize.NL, tokenize.COMMENT, tokenize.INDENT,
                                     tokenize.DEDENT, tokenize.ENDMARKER):
                if at_start and token.string == "@":
                    decorator_lines.append(token.start[0])
                at_start = False
    except (tokenize.TokenError, IndentationError) as exc:
        raise LintError(f"{label}: cannot tokenize staged Python source") from exc

    # Preserve line endings, indentation, comments, and unusual characters in
    # strings. str.splitlines() would split on non-Python newline characters.
    lines = io.StringIO(source, newline="").readlines()
    nodes = sorted(
        (node for node in ast.walk(tree)
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        key=lambda node: (node.lineno, node.col_offset),
    )
    functions = []
    for node in nodes:
        start = node.lineno
        if node.decorator_list:
            first = node.decorator_list[0].lineno
            start = next(line for line in reversed(decorator_lines) if line <= first)
        functions.append(Function(
            path=path,
            name=node.name,
            line=node.lineno,
            source="".join(lines[start - 1:node.end_lineno]),
            column=node.col_offset + 1,
        ))
    return functions


def excluded(path: str, patterns: list[str] | tuple[str, ...]) -> bool:
    parts = Path(path).parts
    normalized = Path(path).as_posix()
    return any(
        fnmatch.fnmatchcase(normalized, pattern) if "/" in pattern
        else any(fnmatch.fnmatchcase(part, pattern) for part in parts)
        for pattern in (*DEFAULT_EXCLUDES, *patterns)
    )


def staged_functions(cwd: Path, excludes: list[str] | tuple[str, ...] = ()) -> list[Function]:
    root_output = git(cwd, "rev-parse", "--show-toplevel")
    root = Path(os.fsdecode(root_output.removesuffix(b"\n")))
    changed = git(root, "diff", "--cached", "--name-only", "-z",
                  "--no-renames", "--diff-filter=ACMRTU", "--")
    paths = {path for path in changed.split(b"\0")
             if path.endswith(b".py") and not excluded(os.fsdecode(path), excludes)}
    if not paths:
        return []

    entries: dict[bytes, list[tuple[bytes, str, bytes]]] = {}
    for record in git(root, "ls-files", "--stage", "-z", "--").split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            if path in paths:
                mode, oid, stage = metadata.split()
                entries.setdefault(path, []).append((mode, oid.decode("ascii"), stage))

    functions = []
    for path in sorted(paths):
        name = os.fsdecode(path)
        indexed = entries.get(path, [])
        if len(indexed) != 1 or indexed[0][2] != b"0":
            raise LintError(f"{json.dumps(name)}: unresolved or missing index entry")
        mode, oid, _ = indexed[0]
        if mode not in (b"100644", b"100755"):
            raise LintError(f"{json.dumps(name)}: staged .py file is not a regular file")
        functions.extend(extract_functions(name, git(root, "cat-file", "blob", oid)))
    return functions


def path_functions(cwd: Path, paths: list[str], excludes: list[str]) -> list[Function]:
    files: set[Path] = set()

    def visit(path: Path) -> None:
        try:
            display = path.relative_to(cwd).as_posix()
        except ValueError:
            display = os.fspath(path)
        if excluded(display, excludes):
            return
        if path.is_symlink():
            raise LintError(f"{json.dumps(display)}: symbolic links are not scanned")
        if path.is_dir():
            try:
                for child in sorted(path.iterdir()):
                    if child.is_dir() or child.suffix == ".py":
                        visit(child)
            except OSError as exc:
                raise LintError(f"{json.dumps(display)}: cannot read directory") from exc
        elif path.is_file():
            if path.suffix == ".py":
                files.add(path)
        else:
            raise LintError(f"{json.dumps(display)}: path does not exist or is not a regular file")

    for name in paths or ["."]:
        candidate = Path(name).expanduser()
        visit(Path(os.path.abspath(cwd / candidate)))
    functions = []
    for path in sorted(files):
        try:
            display = path.relative_to(cwd).as_posix()
        except ValueError:
            display = os.fspath(path)
        try:
            blob = path.read_bytes()
        except OSError as exc:
            raise LintError(f"{json.dumps(display)}: cannot read Python file") from exc
        functions.extend(extract_functions(display, blob))
    return functions
