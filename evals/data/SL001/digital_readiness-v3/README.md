# SL001 examples and splits

Jev judges only the target function source, including decorators and docstrings.
True means flag a needless helper; false means keep it. Training labels can also
be supplied as demonstration functions at inference time.

| Split | Flag | Keep | Purpose |
| --- | ---: | ---: | --- |
| Train | 8 | 12 | Instruction proposals and labeled demonstration selection |
| Dev | 9 | 12 | Selection by continuous probability error |
| Test | 6 | 6 | Final evaluation after freezing the selection |

Train and dev are unchanged from v2. Related functions and synthetic counterparts
remain grouped, including the NormalDist and finite-number contrasts. The fresh
test contains six pairs covering workbook loading, tagged parsing, CSV decoding,
registry fallback, optional-array equality, and directory containment. Each pair
includes a repository function and a constructed counterpart.

V2 test scores were already inspected while considering a different optimization
method, so those cases are not reused as independent final evidence here.
Ambiguous cases remain outside scored splits. Read the examples in train.md,
dev.md, and test.md; JSONL is the machine-readable source.
