"""Check Python functions with small, explicit lint rules."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

from .config import (DEFAULT_THRESHOLD, DEFAULT_TIMEOUT, load_candidate,
                     load_config, validate_settings)
from .extraction import path_functions, staged_functions
from .models import LintError
from .rules import RULES, selected_rules
from .transport import DEFAULT_MODEL, judge


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval":
        from evals.cli import main as eval_main
        return eval_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="slop-linter 0.2.0")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("rules", help="list implemented lint rules")
    commands.add_parser("eval", help="evaluate datasets and optimize rules with DSPy")
    check = commands.add_parser("check", help="check Python files or staged index contents")
    check.add_argument("paths", nargs="*", help="files/directories (default: current directory)")
    check.add_argument("--staged", action="store_true", help="check staged Git index contents")
    check.add_argument("--repo", type=Path, default=Path.cwd(), help="base directory for paths and Git")
    config = check.add_mutually_exclusive_group()
    config.add_argument("--config", type=Path, help="explicit pyproject.toml configuration")
    config.add_argument("--no-config", action="store_true", help="use defaults and CLI options only")
    check.add_argument("--select", help="comma-separated rule codes or names")
    check.add_argument("--ignore", help="comma-separated rule codes or names to disable")
    check.add_argument("--exclude", action="append", help="additional exclusion glob; repeatable")
    check.add_argument("--model", help=f"OpenRouter decision model (default: {DEFAULT_MODEL})")
    check.add_argument("--candidate", type=Path, help="saved optimized candidate JSON")
    check.add_argument("--threshold", type=float,
                       help=f"flag P(yes) >= this value (default: {DEFAULT_THRESHOLD})")
    check.add_argument("--timeout", type=float,
                       help=f"HTTP socket timeout per request (default: {DEFAULT_TIMEOUT:g}s)")
    check.add_argument("--dry-run", action="store_true",
                       help="print extracted functions as JSON without API calls")
    args = parser.parse_args(argv)
    if args.command == "rules":
        for rule in RULES.values():
            print(f"{rule.code} {rule.name}: {rule.predicate}")
        return 0
    if args.staged and args.paths:
        parser.error("paths cannot be combined with --staged")
    try:
        cwd = args.repo.expanduser().resolve()
        if not cwd.is_dir():
            raise LintError("--repo must be an existing directory")
        settings = load_config(cwd, args.config, args.no_config)
        candidate_path = cwd / args.candidate.expanduser() if args.candidate else settings["candidate"]
        candidate = load_candidate(candidate_path) if candidate_path is not None else None
        if candidate:
            settings.update({key: candidate[key] for key in ("model", "threshold", "timeout")})
        for key in ("model", "threshold", "timeout"):
            value = getattr(args, key)
            if value is not None:
                settings[key] = value
        for key in ("select", "ignore"):
            value = getattr(args, key)
            if value is not None:
                settings[key] = [name.strip() for name in value.split(",") if name.strip()]
        patterns = settings["exclude"]
        if not isinstance(patterns, list) or any(not isinstance(pattern, str) or not pattern
                                                 for pattern in patterns):
            raise LintError("exclude must be a list of nonempty glob strings")
        patterns = patterns + (args.exclude or [])
        model, threshold, timeout = (settings[key] for key in ("model", "threshold", "timeout"))
        validate_settings(settings)
        rules = selected_rules(settings["select"], settings["ignore"])
        if candidate:
            if candidate["rule"] not in {rule.code for rule in rules}:
                raise LintError("candidate rule must be selected and not ignored")
            rules = [replace(rule, predicate=candidate["instruction"],
                             demos=tuple(candidate.get("demos", [])))
                     if rule.code == candidate["rule"] else rule for rule in rules]
        if not rules:
            print("[]" if args.dry_run else "slop-linter: no rules selected")
            return 0
        functions = (staged_functions(cwd, patterns) if args.staged
                     else path_functions(cwd, args.paths, patterns))
        if args.dry_run:
            print(json.dumps([asdict(function) for function in functions], indent=2))
            return 0
        if not functions:
            print("slop-linter: no Python functions to check")
            return 0
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key or any(character.isspace() for character in api_key):
            raise LintError("set OPENROUTER_API_KEY to a valid API key")
        violations = 0
        models = set()
        for function in functions:
            label = function.path
            if any(ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF
                   for char in label):
                label = json.dumps(label)
            location = f"{label}:{function.line}:{function.column}"
            for rule in rules:
                try:
                    probability, resolved_model = judge(function, api_key, model, timeout, rule)
                except LintError as exc:
                    raise LintError(f"{location}: {rule.code}: {exc}") from exc
                models.add(resolved_model)
                if probability >= threshold:
                    violations += 1
                    print(f"{location}: {rule.code} {rule.name}: "
                          f"{json.dumps(function.name)}: {rule.message} "
                          f"(P(yes)={probability:.4f} >= {threshold:g}; "
                          f"model={json.dumps(resolved_model)})")
        print(f"slop-linter: checked {len(functions)} functions; {violations} violations; "
              f"models={json.dumps(sorted(models))}")
        return 1 if violations else 0
    except LintError as exc:
        print(f"slop-linter: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("slop-linter: error: interrupted", file=sys.stderr)
        return 2
