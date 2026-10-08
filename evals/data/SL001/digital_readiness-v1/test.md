# Test examples

7 examples. Generated from `test.jsonl`.

**Flag** = needless helper; **Keep** = useful; **Review** = undecided.

## 1. literal

**Flag**

[core/observations.py:207](/Users/carlstone/digital-planet/digital_readiness/core/observations.py:207)

Simply forwards the same site, value, and mask to Literal with no policy or adaptation.

```python
def literal(site: str, value: Any, mask: Any = None) -> Literal:
    """A rule conditioning one ``site`` on ``value`` exactly as given."""
    return Literal(site=site, value=value, mask=mask)
```

## 2. match_level

**Keep**

[core/structured_array.py:246](/Users/carlstone/digital-planet/digital_readiness/core/structured_array.py:246)

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

## 3. _normalize_value

**Keep**

[io/observation_loader.py:731](/Users/carlstone/digital-planet/digital_readiness/io/observation_loader.py:731)

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

## 4. _canonical_level

**Flag**

[model/cpds/observations/_common.py:86](/Users/carlstone/digital-planet/digital_readiness/model/cpds/observations/_common.py:86)

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

## 5. _sigmoid

**Keep**

[model/cpds/observations/beta.py:48](/Users/carlstone/digital-planet/digital_readiness/model/cpds/observations/beta.py:48)

Computes the nonlinear sigmoid mapping rather than forwarding a call; replacing it with a library primitive is a separate concern.

```python
def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))
```

## 6. _validate_kappa

**Keep**

[model/cpds/temporal.py:183](/Users/carlstone/digital-planet/digital_readiness/model/cpds/temporal.py:183)

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

## 7. _canonical_level

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
