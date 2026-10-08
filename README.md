# slop-linter

Evaluation and DSPy refinement live in [evals/README.md](evals/README.md), with
an SL001 dataset drawn from `digital_readiness` and a separate ambiguous-case
review file.

A small Python linter for needless helpers. Run it beside Ruff, mypy, or other
tools in your Python project.

| Code | Name | Judgment |
| --- | --- | --- |
| `SL001` | `needless-helper` | Flags helpers that only rename, forward, preset arguments, or shallowly assemble values without adding useful behavior. |

Only `SL001` is implemented. Each function gets one Noul question via
OpenRouter Decisions. The package includes the best-scoring DSPy question;
the selected program uses no few-shot examples. Findings use
`P(yes) >= threshold`, with a configurable default of `0.5`. Normal checks use
these defaults without a candidate file.

## Install and run

Requires Python 3.11+. Git is needed for installation from GitHub and staged
checks. Runtime dependencies are Python's standard library.

Install from GitHub in your project's Python environment:

```bash
python -m pip install "slop-linter @ git+https://github.com/carl-stone/slop-linter.git@main"
slop-linter check .
```

For a shared project, add the following to `requirements-dev.txt` and install
with `python -m pip install -r requirements-dev.txt`:

```text
slop-linter @ git+https://github.com/carl-stone/slop-linter.git@main
```

After changes are pushed, refresh an existing installation with:

```bash
python -m pip install --upgrade --force-reinstall \
  "slop-linter @ git+https://github.com/carl-stone/slop-linter.git@main"
```

The update command installs the current `main` revision, even if its package
version is unchanged. Each teammate needs `OPENROUTER_API_KEY` set.

To work on the linter itself, use an editable local install:

```bash
cd ~/projects/slop-linter
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

From the Python project you want to lint:

```bash
~/projects/slop-linter/.venv/bin/slop-linter check . --dry-run
~/projects/slop-linter/.venv/bin/slop-linter check src tests
~/projects/slop-linter/.venv/bin/slop-linter check --staged
~/projects/slop-linter/.venv/bin/slop-linter rules
```

Activate that environment to use the short `slop-linter` command, or install into
your project environment with `python -m pip install -e ~/projects/slop-linter`.
You can also use `python -m slop_linter check .` in that environment.

`check` defaults to the current directory and recursively scans `.py` files.
Paths use working tree contents; `--staged` uses Git index blobs and accepts no
paths. `--repo DIR` sets the base directory. `--dry-run` prints extracted
functions as JSON and makes no API calls.

The API key is read **only** from `OPENROUTER_API_KEY`. Keep it in your environment,
out of source, config, and hooks. Inference transmits function source to
OpenRouter and its provider and may incur charges. Tests use mocked HTTP only.

## Project config

Add this to the project being linted's `pyproject.toml`:

```toml
[tool.slop-linter]
select = ["SL001"]
ignore = []
exclude = ["generated", "vendor/*"]
threshold = 0.5
model = "typesafe/jev-1.13"
timeout = 30
```

The nearest ancestor `pyproject.toml` with `[tool.slop-linter]`, starting at the
base directory, supplies config for the entire run. Other tools' tables are
independent. Use `--config FILE` to choose a file or `--no-config` to skip
discovery. Command options override config:

```bash
slop-linter check . --select SL001 --threshold 0.9
slop-linter check . --ignore SL001
slop-linter check . --select needless-helper --exclude '*_generated.py'
```

Selections and ignores accept exact rule codes or names, separated by commas
on the command line. Ignores win. `--ignore ''` clears configured ignores;
`--select ''` selects no rules. With no enabled rules, inference is skipped.
Unknown rules/config keys are errors. All implemented rules are enabled by
default unless `select` explicitly limits them.

Configured exclusions add to `.git`, `.venv*`, `venv`, `__pycache__`, `build`,
and `dist`. A glob containing `/` matches the entire relative path; otherwise
it matches path components. Paths are relative to the base directory in file
mode and the Git root in staged mode. Repeated `--exclude` options add patterns.
Excluded files are skipped before parsing, even when explicitly named.

`threshold` is in `[0, 1]`. `timeout` is a positive per-request socket timeout
in seconds; default 30. It is not a deadline for the whole scan.
`model` must name an OpenRouter decision model.

## Findings and statuses

```text
src/client.py:12:1: SL001 needless-helper: "fetch": needless forwarding helper (P(yes)=0.9500 >= 0.5; model="typesafe/jev-1.13-20260917")
```

- `0`: clean, no functions/enabled rules, or successful dry run.
- `1`: at least one finding.
- `2`: configuration, input, parse, Git, network, or response error.

Errors take precedence over findings; earlier diagnostics may already be
printed. Paths with control characters are JSON quoted. Lines refer to `def`;
columns are one-based AST byte offsets.

All functions are checked, including methods, async functions, and nested
functions. Lambdas are excluded. Source keeps decorators, formatting, comments,
encoding cookies, and line endings. Nested source also appears inside its
enclosing function. Selected files are parsed before inference, without
importing or executing them. Symlinks and unresolved staged conflicts are
errors; staged deletions are skipped. Staged files are checked in full.

## Add a rule

Add a `Rule` entry to `RULES` in `slop_linter/rules.py` with a unique code, name, short
yes/no predicate, and diagnostic message. The runner supplies function source,
calls Noul, validates the answer, applies selection/ignores and threshold,
and emits that rule's code and message. No plugin framework is involved.
An editable install picks up changes immediately.

Default Noul polarity is **Yes = bad behavior (flag)** and
**No = acceptable behavior (keep)**. Labels use the same convention:
`true` means flag and `false` means keep. Keep questions natural rather than
distorting their wording to force this shape.

Add representative inputs and mocked responses to `tests/test_slop_linter.py`.
The registry supports per-function Noul rules. A future deterministic or
whole-file rule would require a new execution path in the runner. Evaluate
new predicates on labeled examples with `evals/`.

## Optimize a rule

DSPy searches questions and labeled few-shot example sets for Jev.
The optimizer and linter share rule definitions and the same decision transport;
optimized questions and examples are saved together as JSON, then loaded directly:

```bash
slop-linter eval validate
slop-linter check src --candidate evals/runs/sl001-candidate.json
```

Install `.[eval]` to use DSPy. See [evals/README.md](evals/README.md) for the
training, comparison, and review workflow.

## Hooks

For the pre-commit framework, add this repository to your existing config:

```yaml
repos:
  - repo: https://github.com/carl-stone/slop-linter
    rev: main
    hooks:
      - id: slop-linter
```

Run `pre-commit install` in the target project to enable the hook.
`.pre-commit-hooks.yaml` defines the hosted hook.
`examples/pre-commit` is a Python native Git hook template for an environment
with slop-linter installed. Preserve existing hooks when integrating it.
Nothing installs hooks automatically.

## API and tests

The [official OpenRouter reference](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
describes `POST https://openrouter.ai/api/alpha/decisions`. Calls use
Bearer authentication, `model`, source as `state`, and one named `noul` question.
The runner validates `answers[rule.code].type` and `answers[rule.code].noul`,
rejecting Booleans, nonfinite/out-of-range values, malformed responses,
redirects, and responses over 1 MiB. The returned model is reported.

Default model `typesafe/jev-1.13` has a 32,000-token budget shared by source and
question. Context errors exit `2`; source is never truncated. Requests are
sequential with no automatic retries. Function-only context may not show why
callers need an abstraction.

```bash
cd ~/projects/slop-linter
.venv/bin/python -m unittest discover -s tests -v
```
