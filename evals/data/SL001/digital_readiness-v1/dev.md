# Dev examples

8 examples. Generated from `dev.jsonl`.

**Flag** = needless helper; **Keep** = useful; **Review** = undecided.

## 1. SiteSpec.broadcast_params

**Keep**

[core/site_spec.py:155](/Users/carlstone/digital-planet/digital_readiness/core/site_spec.py:155)

Aligns every parameter to the shared declared cell shape.

```python
def broadcast_params(self) -> dict[str, "jnp.ndarray"]:
    """``params`` with every field materialized at exactly ``cell_shape``.

    The single place a parameter field is expanded onto the grid, so the
    solo path and the bundled path cannot expand it differently.
    """
    shape = self.cell_shape
    return {
        key: jnp.broadcast_to(jnp.asarray(value), shape)
        for key, value in self.params.items()
    }
```

## 2. NodeDef._cpd_has_family

**Keep**

[io/config/schema.py:1123](/Users/carlstone/digital-planet/digital_readiness/io/config/schema.py:1123)

A required Pydantic validator invokes shared validation with the actual authored node id.

```python
@model_validator(mode="after")
def _cpd_has_family(self) -> "NodeDef":
    _require_cpd_family(self.cpd, where=f"node {self.id!r}")
    return self
```

## 3. _clear_registry

**Flag**

[core/cpd.py:925](/Users/carlstone/digital-planet/digital_readiness/core/cpd.py:925)

Only renames REGISTRY.clear(); the test-reset name adds no behavior or required interface.

```python
def _clear_registry() -> None:
    """Wipe the global registry. Intended only for tests."""

    REGISTRY.clear()
```

## 4. _beta_from_logit

**Keep**

[model/cpds/observations/grid.py:41](/Users/carlstone/digital-planet/digital_readiness/model/cpds/observations/grid.py:41)

Transforms location and scale into valid Beta parameters with floors.

```python
def _beta_from_logit(loc: Any, scale: Any) -> Any:
    """Beta with the mean the logit location implies and κ from the scale."""
    mu = 1.0 / (1.0 + jnp.exp(-loc))
    kappa = jnp.maximum(1.0 / jnp.square(scale), 1.0)
    return dist.Beta(
        jnp.maximum(mu * kappa, 1e-3), jnp.maximum((1.0 - mu) * kappa, 1e-3)
    )
```

## 5. write_sheet

**Flag**

[inference/highlights.py:151](/Users/carlstone/digital-planet/digital_readiness/inference/highlights.py:151)

Only calls DataFrame.to_excel with index=False and returns the supplied path; the direct call makes the export option explicit.

```python
def write_sheet(frame: pd.DataFrame, path) -> Any:
    """Write one highlight frame to ``path`` as a single-sheet workbook."""
    frame.to_excel(path, index=False)
    return path
```

## 6. InferenceResult.segment_shares

**Flag**

[inference/result.py:271](/Users/carlstone/digital-planet/digital_readiness/inference/result.py:271)

Only fixes the segment_shares node name in samples and forwards country and year; callers can use the existing samples API directly.

```python
def segment_shares(self, country: Any, year: Any):
    """Segment shares for one ``(country, year)`` -- the K-simplex vector.

    Named wrapper over ``samples("segment_shares", country=..., year=...)``;
    equals ``self.posterior.segment_shares.sel(country=..., year=...)``.
    """
    return self.samples("segment_shares", country=country, year=year)
```

## 7. truncated_site.site.prepared

**Keep**

[model/cpds/temporal.py:683](/Users/carlstone/digital-planet/digital_readiness/model/cpds/temporal.py:683)

Handles absent bounds and aligns bound dtype and shape to the current location.

```python
def prepared(value: Any) -> Any:
    if value is None:
        return None
    return jnp.broadcast_to(jnp.asarray(value, dtype=loc.dtype), loc.shape)
```

## 8. inline_javascript

**Keep**

[scripts/plot_v4_config.py:959](/Users/carlstone/digital-planet/digital_readiness/scripts/plot_v4_config.py:959)

Escapes closing script text case-insensitively so bundled JavaScript cannot terminate its HTML element.

```python
def inline_javascript(source: str) -> str:
    """Prevent script-like text inside JavaScript from closing its HTML element."""
    return re.sub(
        r"</script",
        lambda match: match[0].replace("</", r"<\/", 1),
        source,
        flags=re.IGNORECASE,
    )
```
