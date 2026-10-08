# Optimize SL001 with DSPy

The default dataset is `data/SL001/digital_readiness-v3/`: **20 train, 21 dev,
and 12 test examples**. Read [train](data/SL001/digital_readiness-v3/train.md),
[dev](data/SL001/digital_readiness-v3/dev.md), and the
[split design](data/SL001/digital_readiness-v3/README.md). Markdown views display
the same functions as JSONL with normal line breaks.

**Yes / true = flag a needless helper; No / false = keep it.** The target input
is only function source, including decorators and docstrings. Selected training
functions and their labels can accompany it as few-shot demonstrations. Target
labels and rationales never go to Jev; rationales inform the proposal model.

The current winning question is built into the package with no demonstrations
and a default cutoff of `0.5`. Normal `slop-linter check PATH` calls use it;
`--candidate FILE` applies a different saved program for future experiments.

## Setup

```bash
python3 -m venv .venv-eval
.venv-eval/bin/python -m pip install -e '.[eval]'
source .venv-eval/bin/activate
slop-linter eval validate
```

Jev uses `OPENROUTER_API_KEY`. Proposals default to `codex exec` through the
CLI's ChatGPT login and configured model. Select a model explicitly with
`--reflection-model MODEL`, or another provider with
`--reflection-backend openrouter --reflection-model PROVIDER/MODEL`.
Calls are live by default; `--mock` explicitly selects fixture scores.

## Joint instruction and example search

```bash
slop-linter eval optimize --out evals/runs/sl001-candidate.json
slop-linter eval compare evals/runs/sl001-candidate.json \
  --out evals/runs/sl001-test.json
slop-linter check ~/digital-planet/digital_readiness/core \
  --candidate evals/runs/sl001-candidate.json --threshold 0.5
```

[DSPy's MIPROv2](https://dspy.ai/current/api/optimizers/MIPROv2/) proposes
instructions from the training data and searches combinations of instructions
and demonstration sets using development scores. This integration uses
`LabeledFewShot` with reviewed Boolean labels, rather than generating answers
or treating Jev's predictions as gold. Related variants stay together in each
set, and demonstration count varies over the available training groups.
The original program remains a candidate.

The selection metric is `1 - (P(Yes) - label)**2`; higher is better. Every trial
uses the complete dev split. The runtime threshold does not affect selection.
Saved candidates contain both the question and demonstration sources/labels;
the runtime transport sends the same configuration used during optimization.

The default is MIPRO's standard `--auto medium` budget. Choose `light` or
`heavy`, or supply `--trials N --num-candidates N` for an explicit search.
There are no package-imposed limits on question length, tokens, trials, or
example count. `--candidate FILE` starts from an existing saved program.
Decision caches live beside the output in `.mipro/`, with example sets included
in cache keys. MIPRO starts a new search on each invocation; this cache is not
an optimizer checkpoint.

## Question refinement with GEPA

```bash
slop-linter eval optimize --optimizer gepa --out evals/runs/sl001-gepa.json
slop-linter eval optimize --optimizer gepa --auto heavy \
  --log-dir evals/runs/sl001-gepa.gepa --out evals/runs/sl001-gepa-continued.json
```

[GEPA](https://dspy.ai/current/getting-started/gepa-optimization/) revises the
question from training feedback, keeping any starting demonstrations fixed.
Use `--auto` or an explicit `--max-calls N` metric-call budget. Metric calls
include scored cached predictions, not just HTTP requests. GEPA checkpoints
resume the search when the same log directory is reused. Both optimizers ask
for concise Jev decision questions and stop without saving a candidate when
model calls fail.

## Data and other rules

Select on **dev**, freeze the result, then score **test** once. V3 retains the
v2 train/dev splits and replaces its already inspected test cases with fresh
pairs. Ambiguous examples belong in `review.jsonl`. Create a new dataset version
when scored data changes; full extraction remains in
`datasets/digital_readiness/functions.jsonl`.

```bash
slop-linter eval render
slop-linter eval extract ~/path/to/repository --out functions.jsonl
slop-linter eval --data evals/data/SL001/v1 baseline --mock --out evals/runs/mock.json
.venv-eval/bin/python -m unittest discover -s tests
```

For another rule, register it in `slop_linter/rules.py`, add the same dataset
layout, and pass `--data PATH` before the eval command. DSPy remains an optional
eval dependency; runtime candidates also work from `[tool.slop-linter]`.
