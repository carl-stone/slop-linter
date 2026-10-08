# Test examples

12 examples. Generated from `test.jsonl`.

**Flag** = needless helper; **Keep** = useful; **Review** = undecided.

## 1. read_tracker

**Keep**

`inference/highlights.py:264`

Checks required columns, renames the data schema, and removes banner rows after reading the workbook.

Paired with `synthetic:tracker-loading-counterpart`.

```python
def read_tracker(path: Path | str | None = None) -> pd.DataFrame:
    """The tracker's national sheet, cut to :data:`TRACKER_COLUMNS`.

    Rows with no indicator code are dropped: the sheet interleaves category
    banner rows among the indicator rows, and they carry a category and nothing
    else.
    """
    path = Path(path) if path is not None else latest_tracker()
    frame = pd.read_excel(path, sheet_name=TRACKER_SHEET, header=TRACKER_HEADER_ROW)
    missing = [name for name in TRACKER_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            f"tracker {path.name!r} sheet {TRACKER_SHEET!r} has no column(s) "
            f"{missing!r}; the highlight indexes on them, so a renamed column "
            "is a silently empty sheet"
        )
    frame = frame[list(TRACKER_COLUMNS)].rename(columns=TRACKER_COLUMNS)
    return frame[frame["indicator"].notna()].reset_index(drop=True)
```

## 2. synthetic:tracker-loading-counterpart

**Flag**

Only presets the worksheet and header arguments for pd.read_excel.

Paired with `inference/highlights.py:264:read_tracker`.

```python
def read_tracker(path: Path | str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=TRACKER_SHEET, header=TRACKER_HEADER_ROW)
```

## 3. parse_level

**Keep**

`inference/labeling.py:116`

Validates the supported type tag, gives a contextual error, and dispatches the corresponding parser.

Paired with `synthetic:level-type-dispatch-counterpart`.

```python
def parse_level(tag: str, text: str) -> Any:
    """Invert ``level_text``."""
    if tag not in _UNTAG:
        raise ValueError(
            f"level type {tag!r} is not one this reader knows "
            f"({', '.join(sorted(_UNTAG))})"
        )
    return _UNTAG[tag](text)
```

## 4. synthetic:level-type-dispatch-counterpart

**Flag**

Only forwards tag and text to an existing parser.

Paired with `inference/labeling.py:116:parse_level`.

```python
def parse_level(tag: str, text: str) -> Any:
    return parse_tagged_level(tag, text)
```

## 5. _read_losses

**Keep**

`inference/serialization.py:308`

Manages the file handle, reads CSV records, and extracts and converts the loss column into a typed array.

Paired with `synthetic:loss-csv-decoding-counterpart`.

```python
def _read_losses(path: Path, dtype: np.dtype) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))[1:]
    return np.asarray([dtype.type(row[1]) for row in rows], dtype=dtype)
```

## 6. synthetic:loss-csv-decoding-counterpart

**Flag**

Only presets CSV-reader options for np.loadtxt; adds no parsing or result handling.

Paired with `inference/serialization.py:308:_read_losses`.

```python
def _read_losses(path: Path, dtype: np.dtype) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=1, dtype=dtype)
```

## 7. registry_entry_for

**Keep**

`io/config/desugar.py:641`

Resolves explicit constructors and implements a unique-constructor fallback while handling missing families.

Paired with `synthetic:registry-constructor-policy-counterpart`.

```python
def registry_entry_for(cpd_def):
    """The registry ``ConstructorEntry`` for a node's CPD, or ``None``.

    Resolves ``cpd.family`` (+ optional ``cpd.constructor``) against the CPD
    registry. An omitted constructor is admitted when the family registers
    exactly one (e.g. a segmentation anchor declaring only its family).
    """
    family_table = REGISTRY.get(cpd_def.family)
    if not family_table:
        return None
    if cpd_def.constructor is not None:
        return family_table.get(cpd_def.constructor)
    if len(family_table) == 1:
        return next(iter(family_table.values()))
    return None
```

## 8. synthetic:registry-constructor-policy-counterpart

**Flag**

Only renames an existing registry resolver.

Paired with `io/config/desugar.py:641:registry_entry_for`.

```python
def registry_entry_for(cpd_def):
    return resolve_registry_entry(cpd_def)
```

## 9. _same_array

**Keep**

`inference/result.py:532`

Handles absent arrays, checks shape and dtype, and applies NaN equality only to floating arrays.

Paired with `synthetic:optional-array-equality-counterpart`.

```python
def _same_array(a: Any, b: Any) -> bool:
    """Two optional arrays, equal bitwise, with ``nan`` equal to itself."""
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))
```

## 10. synthetic:optional-array-equality-counterpart

**Flag**

Only wraps the existing array equality operation and presets its NaN option.

Paired with `inference/result.py:532:_same_array`.

```python
def _same_array(a: Any, b: Any) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))
```

## 11. default_integrated_data_path

**Flag**

`io/observation_loader.py:724`

Only hides a fixed base directory behind an ordinary path-joining operation.

Paired with `synthetic:canonical-data-path-counterpart`.

```python
def default_integrated_data_path(
    filename: str = "national_long_27countries_20260724.xlsx",
) -> Path:
    """Return the canonical path to a file under ``data/03_Integrated_Data/``."""
    return DEFAULT_INTEGRATED_DATA_DIR / filename
```

## 12. synthetic:canonical-data-path-counterpart

**Keep**

Resolves the path and rejects directory traversal before returning it.

Paired with `io/observation_loader.py:724:default_integrated_data_path`.

```python
def default_integrated_data_path(
    filename: str = "national_long_27countries_20260724.xlsx",
) -> Path:
    root = DEFAULT_INTEGRATED_DATA_DIR.resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("data filename must stay inside the integrated-data directory")
    return path
```
