# Repository Guidelines

## Project Structure & Module Organization

`slop_linter/` contains the CLI, extraction, configuration, and OpenRouter transport; `rules.py` defines the shared `RULES` registry. `evals/` contains dataset tooling, metrics, and the optional DSPy optimizer, accessible through `slop-linter eval`. Training datasets live in `evals/data/<RULE_CODE>/<VERSION>/`; extracted corpora live in `datasets/`. Local experiment outputs belong in ignored `evals/runs/`. Tests live in `tests/`; hook templates are in `examples/pre-commit` and `.pre-commit-hooks.yaml`.

## Build, Test, and Development Commands

Use Python 3.11+ and run commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/slop-linter check slop_linter evals --dry-run
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip wheel --no-deps . -w dist
```

These create an environment, install editable CLI entry points, inspect extraction without API calls, run tests, and build a wheel. For evaluation work, create `.venv-eval`, install `-e '.[eval]'`, then run `.venv-eval/bin/slop-linter eval validate`. See `evals/README.md` for optimization and comparison commands. Apply saved questions and examples with `check --candidate FILE`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and `UPPER_CASE` constants. Keep runtime dependencies in the standard library; DSPy belongs in the eval extra. Register rules in `RULES` with unique codes such as `SL002` and kebab-case names. Default Noul polarity: **Yes = bad behavior (flag); No = acceptable behavior (keep)**. Keep question wording natural.

## Testing Guidelines

Use `unittest`, `unittest.mock`, and temporary directories or Git repositories. Name files `test_*.py` and methods `test_<behavior>`. Mock HTTP requests. Add tests for plausible behavioral failures, not implementation details or routine documentation edits. Run the suite with `.venv-eval/bin/python` to include DSPy tests. Keep related evaluation examples within one split and reserve test data for final comparisons.

## Commit & Pull Request Guidelines

Use concise imperative commit subjects, such as `Add SL002 rule selection`. Keep changes focused. PR descriptions should explain the behavior change and any material limitations; include example diagnostics when useful. Update affected documentation.

## Security & Configuration

Read Jev credentials from `OPENROUTER_API_KEY`; never commit keys. Codex reflection reuses the CLI's ChatGPT login. Evaluation commands make model calls by default; use `--mock` explicitly for fixtures. `check --dry-run` prints extraction without model calls.

## Agent Instructions

Handle verification, provenance, and uncertainty internally. Surface them briefly only when they affect the result or require a decision. Do not add verification reports, audit checklists, or process commentary.
