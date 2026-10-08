# digital_readiness functions for SL001

`functions.jsonl` contains all 2,571 functions extracted from the current
`~/digital-planet/digital_readiness` Python source, including methods and nested
functions. Each record includes its path, qualified name, line, and source.
Environment, dependency, data, and generated-output directories are excluded.
The readable version is [functions.md](functions.md).

All 577 production functions were inspected and labeled: 14 needless helpers
and 563 useful functions. The 1,994 test/support functions
remain in the inventory but are excluded from optimization.

The compact training dataset is
[`evals/data/SL001/digital_readiness-v1`](../../evals/data/SL001/digital_readiness-v1/).
It uses all 14 positives, the user's 3 `NormalDist` examples, 18 selected
useful helpers, and 2 synthetic useful counterparts. These negatives include protocol callbacks, shape
adapters, validation, clipping, escaping, and real mathematical transforms.
Function length does not determine the label.

Two matched contrasts keep names and callees similar: useless `NormalDist`
forwarding versus numeric-input normalization, and `_canonical_level`
forwarding versus cleaning missing text labels. Each contrast stays in one split.

| Split | Needless | Useful | Total |
| --- | ---: | ---: | ---: |
| Train | 12 | 10 | 22 |
| Dev | 3 | 5 | 8 |
| Test | 2 | 5 | 7 |

Related distribution aliases and observation-binding forwarders stay together
within their splits. Development positives cover registry resets, fixed export
options, and fixed result selection. All reviewed cases are now labeled and
included in the scored dataset.
[`review.md`](../../evals/data/SL001/digital_readiness-v1/review.md) is currently
empty; future ambiguous cases belong there until their labels are settled.

See [`evals/README.md`](../../evals/README.md) for the DSPy workflow.
