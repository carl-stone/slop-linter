import argparse
from pathlib import Path
import sys

from .core import (DATA_ROOT, DecisionsService, baseline_prompt, digest, evaluate,
                   load_split, manifest, metadata, read_json, save_json, validate_dataset, validate_prompt)
import slop_linter as lint


def main(argv=None):
    parser = argparse.ArgumentParser(prog="slop-linter eval",
                                     description="Evaluate and refine slop-linter rules.")
    parser.add_argument("--data", type=Path, default=DATA_ROOT / "SL001" / "digital_readiness-v3")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate frozen examples and split grouping")
    extract = commands.add_parser("extract", help="extract Python functions for labeling")
    extract.add_argument("repository", type=Path)
    extract.add_argument("--out", type=Path, required=True)
    render = commands.add_parser("render", help="create readable Markdown views of JSONL examples")
    render.add_argument("paths", nargs="*", type=Path, help="JSONL files or directories (default: current dataset)")
    render.add_argument("--source-root", type=Path, help="source repository for clickable locations")
    for name in ("baseline", "optimize", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--mock", dest="live", action="store_false",
                             help="use fixture predictions without model calls")
        command.add_argument("--out", type=Path, required=True, help="new JSON result path; never overwrite")
        if name == "compare":
            command.add_argument("candidate", type=Path)
        else:
            command.add_argument("--model")
            command.add_argument("--threshold", type=float)
        if name == "baseline":
            command.add_argument("--split", choices=("train", "dev", "test"), default="dev")
        if name == "optimize":
            command.add_argument("--optimizer", choices=("mipro", "gepa"), default="mipro",
                                 help="joint question/example search (MIPRO, default) or question search (GEPA)")
            command.add_argument("--candidate", type=Path, help="saved starting question and examples")
            command.add_argument("--reflection-backend", choices=("codex", "openrouter"), default="codex",
                                 help="reflection provider (default: Codex with ChatGPT login)")
            command.add_argument("--reflection-model",
                                 help="Codex model override, or required OpenRouter model identifier")
            budget = command.add_mutually_exclusive_group()
            budget.add_argument("--auto", choices=("light", "medium", "heavy"),
                                help="DSPy's standard search budget (default: medium)")
            budget.add_argument("--max-calls", type=int,
                                help="GEPA metric-call budget instead of --auto")
            command.add_argument("--trials", type=int, help="MIPRO search trials instead of --auto")
            command.add_argument("--num-candidates", type=int, help="MIPRO candidate pool size with --trials")
            command.add_argument("--log-dir", type=Path,
                                 help="decision-cache directory; GEPA also resumes its search here")
            command.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            from .readable import render_paths
            paths = render_paths(args.paths or [args.data], args.source_root)
            print(f"Created {len(paths)} readable Markdown files.")
            return 0
        if args.command == "extract":
            from .extract import extract_repository
            count = extract_repository(args.repository, args.out)
            print(f"Extracted {count} functions: {args.out}")
            return 0
        if args.command == "validate":
            print(validate_dataset(args.data))
            return 0
        if args.out.exists():
            raise lint.LintError("output already exists; choose a new result path")
        rule = manifest(args.data)["rule"]
        if args.command == "baseline":
            prompt = baseline_prompt(rule, args.model, lint.DEFAULT_THRESHOLD if args.threshold is None else args.threshold)
            cases = load_split(args.data, args.split)
            service = DecisionsService(cases, prompt, live=args.live, max_calls=len(cases))
            result = {"schema_version": 1, **metadata(args.data, "live" if args.live else "mock"),
                      "split": args.split, "baseline": evaluate(cases, prompt, service),
                      "decision_calls": service.calls}
        elif args.command == "optimize":
            if args.max_calls is not None and args.max_calls <= 0:
                raise lint.LintError("max-calls must be positive")
            from .optimize import optimize
            prompt = (lint.load_candidate(args.candidate) if args.candidate else
                      baseline_prompt(rule, args.model, lint.DEFAULT_THRESHOLD if args.threshold is None else args.threshold))
            if args.model is not None:
                prompt["model"] = args.model
            if args.threshold is not None:
                prompt["threshold"] = args.threshold
            if prompt["rule"] != rule:
                raise lint.LintError("starting candidate rule does not match dataset")
            # Test and review files are never loaded by optimization.
            train, dev = load_split(args.data, "train"), load_split(args.data, "dev")
            result, _, _ = optimize(args.data, train, dev,
                prompt, live=args.live,
                reflection_model=args.reflection_model, reflection_backend=args.reflection_backend,
                max_calls=args.max_calls, auto=args.auto, seed=args.seed,
                log_dir=args.log_dir or args.out.with_suffix("." + args.optimizer),
                optimizer_name=args.optimizer, num_trials=args.trials, num_candidates=args.num_candidates)
        else:
            artifact = read_json(args.candidate)
            if artifact.get("schema_version") != 1:
                raise lint.LintError("invalid candidate artifact")
            if artifact.get("manifest_sha256") != digest(args.data / "manifest.json"):
                raise lint.LintError("candidate dataset version does not match")
            if (artifact.get("mode") == "live") != args.live:
                raise lint.LintError("candidate and comparison must use the same live/mock mode")
            baseline, candidate = artifact["baseline"], artifact["candidate"]
            validate_prompt(baseline)
            validate_prompt(candidate)
            if {k: v for k, v in baseline.items() if k not in {"instruction", "demos"}} != {
                    k: v for k, v in candidate.items() if k not in {"instruction", "demos"}}:
                raise lint.LintError("comparison must hold model, threshold, and rule fixed")
            cases = load_split(args.data, "test")
            service = DecisionsService(cases, baseline, live=args.live, max_calls=2 * len(cases))
            result = {"schema_version": 1, **metadata(args.data, "live" if args.live else "mock"),
                      "split": "test",
                      "baseline": evaluate(cases, baseline, service),
                      "candidate": evaluate(cases, candidate, service),
                      "decision_calls": service.calls}
        save_json(args.out, result)
        print(f"Saved {args.command} result: {args.out} ({result['mode']})")
        return 0
    except (lint.LintError, ImportError, OSError, KeyError, ValueError) as exc:
        print(f"slop-linter-eval: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
