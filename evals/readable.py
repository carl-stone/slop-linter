"""Readable Markdown views of JSONL function datasets."""

from __future__ import annotations

import json
from pathlib import Path
import re

import slop_linter as lint


def render_file(path: Path, source_root: Path | None = None) -> Path:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError) as exc:
        raise lint.LintError(f"cannot read JSONL: {path}") from exc
    title = path.stem.replace("_", " ").title()
    parts = [f"# {title} examples", "", f"{len(rows)} examples. Generated from `{path.name}`.",
             "", "**Flag** = needless helper; **Keep** = useful; **Review** = undecided.", ""]
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not isinstance(row.get("source"), str):
            raise lint.LintError(f"example {index} needs function source: {path}")
        name = row.get("qualname", row.get("id", f"Example {index}"))
        if "label" not in row:
            label = "Unlabeled (test/support)" if row.get("is_test") else "Unlabeled"
        else:
            label = {True: "Flag", False: "Keep", None: "Review"}.get(row["label"], "Unlabeled")
        parts += [f"## {index}. {name}", "", f"**{label}**", ""]
        location = row.get("path")
        if location and not location.startswith("<"):
            line = row.get("line", 1)
            target = source_root / location if source_root else None
            if target is not None and target.is_file():
                parts += [f"[{location}:{line}]({target.resolve()}:{line})", ""]
            else:
                parts += [f"`{location}:{line}`", ""]
        elif location == "<synthetic-example>":
            parts += ["Constructed counterpart.", ""]
        elif location == "<user-example>":
            parts += ["User-supplied example.", ""]
        if row.get("rationale"):
            parts += [row["rationale"], ""]
        if row.get("paired_with"):
            parts += [f"Paired with `{row['paired_with']}`.", ""]
        source = row["source"].rstrip()
        fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", source)), default=0))
        parts += [f"{fence}python", source, fence, ""]
    output = path.with_suffix(".md")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


def render_paths(paths: list[Path], source_root: Path | None = None) -> list[Path]:
    source_root = source_root.expanduser() if source_root else None
    files = set()
    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            files.update(path.rglob("*.jsonl"))
        elif path.is_file() and path.suffix == ".jsonl":
            files.add(path)
        else:
            raise lint.LintError(f"expected a JSONL file or dataset directory: {path}")
    return [render_file(path, source_root) for path in sorted(files)]
