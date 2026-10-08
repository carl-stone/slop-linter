# Dev examples

21 examples. Generated from `dev.jsonl`.

**Flag** = needless helper; **Keep** = useful; **Review** = undecided.

## 1. match_level

**Keep**

`core/structured_array.py:246`

Matches exact levels and case-insensitive string spellings.

```python
def match_level(value: Any, levels: Sequence[Any]) -> Any | None:
    """The member of ``levels`` matching ``value``, or ``None`` if none does.

    Strings match case-insensitively (by ``casefold``); everything else matches
    by equality. This bridges the gender convention split -- the loader writes
    lowercase ``"f"``/``"m"``, a config's gender axis may declare ``"F"``/``"M"``
    -- and returns the *declared* spelling, so a caller can index or pin without
    either side having to win the case war.

    Lives here rather than beside its callers because the desugar pass that
    chooses a grid's levels and the CPD that reads the latent at them must agree
    on what counts as the same level [08-18 10:28].
    """
    for level in levels:
        if level == value:
            return level
        if (
            isinstance(level, str)
            and isinstance(value, str)
            and level.casefold() == value.casefold()
        ):
            return level
    return None
```

## 2. _canonical_level

**Flag**

`model/cpds/observations/_common.py:86`

Only extracts axis.levels and forwards to match_level; callers can invoke the existing matching helper directly.

```python
def _canonical_level(value: Any, axis: Any) -> Any | None:
    """``value`` as ``axis`` spells it, or ``None`` if the axis has no such level.

    The rule itself is :func:`~core.structured_array.match_level`; this is the
    axis-shaped call of it, which is what every caller here has in hand. The
    bridge it provides -- the loader writes lowercase ``"f"``/``"m"``, a config's
    gender axis may declare ``"F"``/``"M"`` -- is described there.
    """
    return match_level(value, axis.levels)
```

## 3. _canonical_level

**Keep**

Constructed counterpart.

Unlike the attribute-forwarding alias, this trims incoming labels and converts explicit missing-value tokens to absence before matching.

Paired with `model/cpds/observations/_common.py:86:_canonical_level`.

```python
def _canonical_level(value, axis):
    """Clean external text labels before matching declared levels."""
    if value is None:
        return None
    value = str(value).strip()
    if value.casefold() in {"", "unknown", "n/a"}:
        return None
    return match_level(value, axis.levels)
```

## 4. _beta_from_logit

**Keep**

`model/cpds/observations/grid.py:41`

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

## 5. _resolve_beta_observation_bindings

**Flag**

`model/cpds/observations/beta.py:142`

Forwards unchanged arguments to resolve_observation_bindings, which already has the required callback signature.

```python
def _resolve_beta_observation_bindings(
    *,
    child: NodeMeta,
    parents: tuple[NodeMeta, ...],
    hyperparameters: Mapping[str, Any],
) -> Mapping[str, tuple[str, ...]]:
    return resolve_observation_bindings(
        child=child, parents=parents, hyperparameters=hyperparameters
    )
```

## 6. literal

**Flag**

`core/observations.py:207`

Simply forwards the same site, value, and mask to Literal with no policy or adaptation.

```python
def literal(site: str, value: Any, mask: Any = None) -> Literal:
    """A rule conditioning one ``site`` on ``value`` exactly as given."""
    return Literal(site=site, value=value, mask=mask)
```

## 7. filesystem_path

**Flag**

Constructed counterpart.

An ordinary method only renames os.fspath(self.path); unlike __fspath__, it implements no required protocol.

Paired with `synthetic:filesystem-path-protocol`.

```python
def filesystem_path(self):
    return os.fspath(self.path)
```

## 8. __fspath__

**Keep**

Constructed counterpart.

Implements the required os.PathLike protocol so the object can be passed to filesystem APIs.

Paired with `synthetic:filesystem-path-alias`.

```python
def __fspath__(self):
    return os.fspath(self.path)
```

## 9. write_sheet

**Flag**

`inference/highlights.py:151`

Only calls DataFrame.to_excel with index=False and returns the supplied path; the direct call makes the export option explicit.

Paired with `synthetic:write-sheet-atomic`.

```python
def write_sheet(frame: pd.DataFrame, path) -> Any:
    """Write one highlight frame to ``path`` as a single-sheet workbook."""
    frame.to_excel(path, index=False)
    return path
```

## 10. write-sheet-atomic

**Keep**

Constructed counterpart.

Coordinates temporary-file creation, atomic replacement, and cleanup; these are observable guarantees beyond forwarding to to_excel.

Paired with `inference/highlights.py:151:write_sheet`.

```python
def write_sheet(frame: pd.DataFrame, path) -> Any:
    """Replace a workbook atomically and remove temporary files on failure."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_excel(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
```

## 11. InferenceResult.segment_shares

**Flag**

`inference/result.py:271`

Only fixes the segment_shares node name in samples and forwards country and year; callers can use the existing samples API directly.

Paired with `synthetic:segment-shares-summary`.

```python
def segment_shares(self, country: Any, year: Any):
    """Segment shares for one ``(country, year)`` -- the K-simplex vector.

    Named wrapper over ``samples("segment_shares", country=..., year=...)``;
    equals ``self.posterior.segment_shares.sel(country=..., year=...)``.
    """
    return self.samples("segment_shares", country=country, year=year)
```

## 12. segment-shares-summary

**Keep**

Constructed counterpart.

Computes a posterior summary from draws instead of merely returning a fixed-node selection.

Paired with `inference/result.py:271:InferenceResult.segment_shares`.

```python
def segment_shares(self, country: Any, year: Any):
    """Return the 5th percentile, median, and 95th percentile of segment shares."""
    samples = self.samples("segment_shares", country=country, year=year)
    return np.quantile(samples, (0.05, 0.5, 0.95), axis=0)
```

## 13. has_explicit_observation_nodes

**Flag**

`io/pipeline.py:194`

Only renames has_declared_observation_nodes; the legacy alias adds no behavior and has no production callers in this repository.

```python
def has_explicit_observation_nodes(config: GraphConfig) -> bool:
    """Backward-compatible alias for :func:`has_declared_observation_nodes`.

    Prefer the more specific helpers for new code: build routing uses
    ``has_declared_observation_nodes``; Fork-A fit detection uses
    ``graph_has_observation_v0_sites``.
    """
    return has_declared_observation_nodes(config)
```

## 14. _normalize_value

**Keep**

`io/observation_loader.py:731`

Converts percentage units to fractions and applies the beta-support boundary policy.

```python
def _normalize_value(unit: str, raw: float) -> float:
    if unit in _PCT_UNITS:
        v = float(raw) / 100.0
        # TODO: When we settle on the observation distributions we want to use,
        # remove this clipping logic. Clipping for use with beta works for now,
        # but eventually we'll want a distribution whose support contains our
        # actual observations.
        return float(np.clip(v, 1e-6, 1.0 - 1e-6))
    return float(raw)
```

## 15. NodeDef._cpd_has_family

**Keep**

`io/config/schema.py:1123`

A required Pydantic validator invokes shared validation with the actual authored node id.

```python
@model_validator(mode="after")
def _cpd_has_family(self) -> "NodeDef":
    _require_cpd_family(self.cpd, where=f"node {self.id!r}")
    return self
```

## 16. _sigmoid

**Keep**

`model/cpds/observations/beta.py:48`

Computes the nonlinear sigmoid mapping rather than forwarding a call; replacing it with a library primitive is a separate concern.

Paired with `synthetic:sigmoid-library-alias`.

```python
def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))
```

## 17. sigmoid-library-alias

**Flag**

Constructed counterpart.

Only renames the existing sigmoid callable. The paired real function computes the nonlinear mapping; library replacement of that computation is a separate concern.

Paired with `model/cpds/observations/beta.py:48:_sigmoid`.

```python
def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(x)
```

## 18. _clear_registry

**Flag**

`core/cpd.py:925`

Only renames REGISTRY.clear(); the test-reset name adds no behavior or required interface.

Paired with `synthetic:registry-resource-cleanup`.

```python
def _clear_registry() -> None:
    """Wipe the global registry. Intended only for tests."""

    REGISTRY.clear()
```

## 19. registry-resource-cleanup

**Keep**

Constructed counterpart.

Owns resource cleanup before registry removal; a direct REGISTRY.clear() call would omit the cleanup.

Paired with `core/cpd.py:925:_clear_registry`.

```python
def _clear_registry() -> None:
    """Close registered resources before removing them."""
    for resource in REGISTRY.values():
        resource.close()
    REGISTRY.clear()
```

## 20. _validate_kappa

**Keep**

`model/cpds/temporal.py:183`

Coerces and validates the convex-combination adjustment speed.

```python
def _validate_kappa(raw: Any, label: str) -> float:
    """kappa, checked to be in [0, 1].

    Outside that range the self-bounding guarantee breaks: the update stops being
    a convex combination, so the mean can leave [low, high] and the truncation
    starts doing real work (silently distorting the dynamics) instead of merely
    being redundant.
    """
    kappa = float(raw)
    if not 0.0 <= kappa <= 1.0:
        raise ValueError(
            f"{label}: kappa must be in "
            f"[0, 1] (the update is a convex combination); got {kappa}."
        )
    return kappa
```

## 21. truncated_site.site.prepared

**Keep**

`model/cpds/temporal.py:683`

Handles absent bounds and aligns bound dtype and shape to the current location.

```python
def prepared(value: Any) -> Any:
    if value is None:
        return None
    return jnp.broadcast_to(jnp.asarray(value, dtype=loc.dtype), loc.shape)
```
