# SL001 examples and splits

This set targets needless helpers using **only function source**, including
decorators and docstrings. True means needless; false means acceptable. The
labels express the distinction agreed in this project: naming, argument
presets, callable identity, and compatibility aliases alone do not justify a
wrapper. Validation, actual transformation, cleanup, and required interfaces
can justify small functions.

| Split | Flag | Keep | Purpose |
| --- | ---: | ---: | --- |
| Train | 8 | 12 | Feedback for question revisions |
| Dev | 9 | 12 | Selection of the best question |
| Test | 5 | 5 | Final evaluation of the frozen selection |

Training covers Python membership and direct, fixed-argument, and multi-statement aliases alongside
numeric policy, input adaptation, shape checks, parameter construction, and
substantial computations. Development covers callback and compatibility aliases,
attribute forwarding, fixed selectors and options, registry reset, validation,
protocols, normalization, and mathematical helpers. Both contain positives and
small useful functions; neither assigns a label based on line count.

Pairs contrast almost the same operation with and without useful behavior:
normal-distribution construction versus input validation; float casting versus
finite-number policy; registry clearing versus resource cleanup; workbook writing
versus atomic replacement; sample selection versus statistical summaries; direct
forwarding to sigmoid versus computing the mapping; an ordinary path alias versus
the required filesystem path protocol.

The ten test examples are five fresh pairs covering graph endpoint validation,
the Python length protocol, optional string normalization, YAML loading, and
contextual error handling. Each pair contains an unused repository function and
a constructed needless counterpart. All pairs and near-duplicate variants stay
in one split. Three redundant aliases from v1 were omitted, while their labels
remain in the full inventory and v1 dataset.

The earlier seven test examples now belong to development because their scores
influenced prompt selection. Test cases should be scored only after selecting
and freezing the winner on development data.
