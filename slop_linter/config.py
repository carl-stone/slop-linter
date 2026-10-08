"""Project settings and deployable optimized rule candidates."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tomllib

from .models import LintError
from .rules import RULES, check_instruction
from .transport import DEFAULT_MODEL


DEFAULT_THRESHOLD = 0.5
DEFAULT_TIMEOUT = 30.0


def load_config(cwd: Path, explicit: Path | None, disabled: bool) -> dict:
    settings = {"select": list(RULES), "ignore": [], "exclude": [],
                "model": DEFAULT_MODEL, "threshold": DEFAULT_THRESHOLD,
                "timeout": DEFAULT_TIMEOUT, "candidate": None}
    if disabled:
        return settings
    candidates = [cwd / explicit.expanduser()] if explicit is not None else [
        folder / "pyproject.toml" for folder in (cwd, *cwd.parents)
    ]
    for path in candidates:
        if explicit is None and not path.is_file():
            continue
        try:
            with path.open("rb") as stream:
                document = tomllib.load(stream)
        except (OSError, ValueError) as exc:
            raise LintError(f"{json.dumps(os.fspath(path))}: cannot read TOML configuration") from exc
        tool = document.get("tool", {})
        if not isinstance(tool, dict):
            raise LintError("TOML tool configuration must be a table")
        table = tool.get("slop-linter")
        if table is None:
            if explicit is not None:
                raise LintError("configuration needs a [tool.slop-linter] table")
            continue
        if not isinstance(table, dict) or set(table) - set(settings):
            raise LintError("[tool.slop-linter] contains unsupported configuration keys")
        settings.update(table)
        if settings["candidate"] is not None:
            candidate = settings["candidate"]
            if not isinstance(candidate, str) or not candidate.strip():
                raise LintError("candidate must be a nonempty file path")
            settings["candidate"] = path.parent / Path(candidate).expanduser()
        break
    return settings


def validate_settings(settings: dict) -> None:
    model, threshold, timeout = (settings.get(key) for key in ("model", "threshold", "timeout"))
    if not isinstance(model, str) or not model.strip() or any(char.isspace() for char in model):
        raise LintError("model must be a nonempty model identifier without whitespace")
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise LintError("threshold must be a finite number in [0, 1]")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise LintError("timeout must be a positive finite number of seconds")


def validate_prompt(prompt: dict) -> None:
    if not isinstance(prompt, dict) or not isinstance(prompt.get("rule"), str):
        raise LintError("candidate needs a rule code and saved prompt configuration")
    if prompt["rule"] not in RULES:
        raise LintError(f"unknown candidate rule {json.dumps(prompt['rule'])}")
    check_instruction(prompt.get("instruction"))
    validate_settings(prompt)
    demos = prompt.get("demos", [])
    if not isinstance(demos, list) or any(
        not isinstance(demo, dict) or set(demo) != {"source", "label"}
        or not isinstance(demo["source"], str) or not demo["source"].strip()
        or type(demo["label"]) is not bool for demo in demos
    ):
        raise LintError("demonstrations must contain function source and a Boolean label")


def load_candidate(path: Path) -> dict:
    """Load either an optimizer result or its plain candidate prompt object."""
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise LintError(f"{json.dumps(os.fspath(path))}: cannot read candidate JSON") from exc
    if not isinstance(artifact, dict):
        raise LintError("candidate JSON must contain an object")
    if "candidate" in artifact:
        if type(artifact.get("schema_version")) is not int or artifact["schema_version"] != 1:
            raise LintError("unsupported candidate artifact schema")
        prompt = artifact["candidate"]
    else:
        prompt = artifact
    validate_prompt(prompt)
    return {key: prompt[key] for key in ("rule", "instruction", "model", "threshold", "timeout", "demos")
            if key in prompt}
