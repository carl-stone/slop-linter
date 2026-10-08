# Train examples

20 examples. Generated from `train.jsonl`.

**Flag** = needless helper; **Keep** = useful; **Review** = undecided.

## 1. _resolve_standardize_bindings

**Keep**

`model/cpds/standardize.py:135`

Translates graph parents to the value role required by the registry callback.

```python
def _resolve_standardize_bindings(
    *,
    child: NodeMeta,
    parents: tuple[NodeMeta, ...],
    hyperparameters: Mapping[str, Any],
) -> Mapping[str, tuple[str, ...]]:
    """Classify every parent into the single ``value`` role."""
    del child, hyperparameters
    return {"value": tuple(p.node_id for p in parents)}
```

## 2. binomial_observation

**Keep**

`model/cpds/binomial.py:69`

Implements the CPD interface by obtaining a rate parent, sampling, and producing count metadata.

```python
def binomial_observation(
    resolved_bindings: Mapping[str, tuple[StructuredArray, ...]],
    *,
    sample_fn: SampleFn,
    deterministic_fn: DeterministicFn,
    hyperparameters: Mapping[str, Any],
    name: str,
) -> StructuredArray:
    """Binomial(N, p) with success probability supplied by the ``rate`` parent.

    Reads the single parent ``StructuredArray`` out of
    ``resolved_bindings["rate"]``, extracts its raw array for the success
    probability, constructs a NumPyro Binomial, calls ``sample_fn`` (so
    the wrapper can apply observe / intervene overrides), and returns
    the sampled count wrapped in a ``StructuredArray``. The arity-1
    contract on ``rate`` (declared in the schema, enforced by
    ``validate_resolution``) is what lets the ``[0]`` index here be safe.
    """
    del deterministic_fn  # stochastic CPD, no deterministic calls
    rate_parent = resolved_bindings["rate"][0]
    p = rate_parent.array
    n = hyperparameters["total_count"]
    value = sample_fn(name, dist.Binomial(total_count=n, probs=p))
    # Scalar wrapper: 0-d count. Indicator metadata is decorative for
    # the toy-conjugate-pair usage -- see the module-level TODO above.
    return StructuredArray(
        array=value,
        axes=(),
        indicator=IndicatorMeta(
            indicator=name, family="count", unit="successes"
        ),
        population=None,
    )
```

## 3. _build_normal

**Flag**

`model/cpds/normal.py:75`

Forwards unchanged keyword arguments to dist.Normal; the distribution class can be the callable directly.

```python
def _build_normal(**params):
    """The family closure a ``SiteSpec`` carries (see ``core/site_spec.py``).

    Module-level so its identity is stable: ``bundle_layout`` uses ``build``
    identity to refuse cross-family bundles.
    """
    return dist.Normal(**params)
```

## 4. NormalDist

**Keep**

Constructed counterpart.

Unlike the alias, this accepts numeric strings and rejects nonfinite parameters or nonpositive scale before constructing the distribution.

Paired with `user:NormalDist-direct`.

```python
def NormalDist(mu, sigma):
    """Accept finite scalar numbers or numeric strings with positive scale."""
    mu = float(mu)
    sigma = float(sigma)
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Normal parameters must be finite and scale positive")
    return dist.Normal(mu, sigma)
```

## 5. NormalDist

**Flag**

User-supplied example.

Only renames the Normal constructor while forwarding the same parameters.

```python
def NormalDist(mu, sigma):
    return dist.Normal(mu, sigma)
```

## 6. NormalDist

**Flag**

User-supplied example.

Hides a zero-mean argument in a constructor alias without expressing a distinct useful policy.

```python
def NormalDist(sigma):
    return dist.Normal(0, sigma)
```

## 7. NormalDist

**Flag**

User-supplied example.

Local aliases add no behavior; this is the same unnecessary constructor forwarding across several lines.

```python
def NormalDist(mu, sigma):
    m = mu
    sig = sigma
    return dist.Normal(m, sig)
```

## 8. StructuredArray.with_node_id

**Flag**

`core/structured_array.py:550`

Only wraps dataclasses.replace to set node_id; the attribution name adds no behavior beyond the direct replacement.

```python
def with_node_id(self, node_id: str | None) -> "StructuredArray":
    """This value, attributed to (or detached from) a producing node.

    The stamping seam. ``node_id=None`` detaches, which is what bundle
    projection wants: a ``(consumer, role)`` block is not a node's output.
    """
    return replace(self, node_id=node_id)
```

## 9. _synthesized_theta_weight_loc

**Keep**

`io/config/build.py:517`

Transforms AxisMeta objects into the axis-name and level pairs accepted by shared inference.

```python
def _synthesized_theta_weight_loc(
    config: GraphConfig, axes: tuple[AxisMeta, ...]
) -> float | list:
    """Adapter: ``desugar.synthesized_weight_loc`` over resolved ``AxisMeta``."""
    return synthesized_weight_loc(config, [(ax.name, ax.levels) for ax in axes])
```

## 10. load_demo_cross_section

**Flag**

`io/observation_loader.py:1094`

Constructs a year-window filter and delegates to the existing loader. It adds no validation, parsing, normalization, or result handling beyond presetting call arguments.

```python
def load_demo_cross_section(
    path: str | Path | None = None,
    *,
    year: int = 2024,
    flt: ObservationLoadFilter | None = None,
) -> LoadedObservations:
    """Load a single-year national cross-section through the production loader.

    When a graph config is available, use
    :func:`digital_readiness.io.pipeline.load_observations_for_config` instead
    so indicator set and ``axis_breakdown`` come from the config. This helper
    only adds a year window on top of ``flt`` (default filter includes
    ``gender → {pop}``).
    """
    base = flt or ObservationLoadFilter()
    return load_observations_from_national_long(
        path,
        flt=ObservationLoadFilter(
            **{**base.__dict__, "year_min": year, "year_max": year},
        ),
    )
```

## 11. load_pct_smartphone_slice

**Flag**

`io/observation_loader.py:1117`

Presets a demo indicator, source, units, status, and year in a filter, then forwards to the existing loader; it adds no loading behavior beyond arranging arguments.

```python
def load_pct_smartphone_slice(
    *,
    path: str | Path | None = None,
    year: int = 2020,
    countries: Sequence[str] | None = None,
    source: str = "GSMAINT",
) -> LoadedObservations:
    """Convenience loader for the v0 demo slice (``pct_smartphone`` actuals)."""
    return load_observations_from_national_long(
        path,
        flt=ObservationLoadFilter(
            indicator_codes=("pct_smartphone",),
            source_codes=(source,),
            country_iso3=tuple(countries) if countries is not None else None,
            year_min=year,
            year_max=year,
            units=("pct",),
            obs_status=("A",),
        ),
    )
```

## 12. _select

**Keep**

`io/observation_loader.py:842`

Checks that an explicitly filtered column exists and supplies a contextual error before membership selection.

```python
def _select(
    out: pd.DataFrame, column: str, allowed: Sequence[Any]
) -> pd.DataFrame:
    """Keep rows whose ``column`` is in ``allowed``. An absent column errors [10:39]."""
    if column not in out.columns:
        raise KeyError(
            f"filter names column {column!r}, which the frame does not have; "
            f"it has {sorted(map(str, out.columns))}"
        )
    return out[out[column].isin(list(allowed))]
```

## 13. _finite_number

**Keep**

`inference/nuts_runner.py:112`

Normalizes numerical diagnostics to finite floats or JSON-safe absence.

Paired with `synthetic:finite-number-cast-alias`.

```python
def _finite_number(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None
```

## 14. finite-number-cast-alias

**Flag**

Constructed counterpart.

Only wraps float(value). Unlike the finite-number policy, it performs no finiteness check or conversion of nonfinite values to absence.

Paired with `inference/nuts_runner.py:112:_finite_number`.

```python
def _finite_number(value: Any) -> float:
    return float(value)
```

## 15. Literal.mask_for

**Keep**

`core/observations.py:141`

Checks mask shape against the actual sampled site before returning it.

```python
def mask_for(self, site: str, shape: tuple[int, ...] | None = None) -> Any:
    """The stored mask, checked against the site the same way the value is."""
    if self.mask is None or shape is None:
        return self.mask
    _check_reaches(
        "observation mask",
        site,
        self.mask,
        shape,
        "Supply one flag per cell.",
    )
    return self.mask
```

## 16. ordered_thresholds_from_deltas

**Keep**

`model/cpds/ordered_threshold.py:64`

Transforms unconstrained increments into ordered thresholds with minimum gaps.

```python
def ordered_thresholds_from_deltas(
    deltas: jnp.ndarray, *, min_gap: MinGap = 0.0
) -> jnp.ndarray:
    """Map unconstrained ``deltas`` (last axis length T=K-1) to increasing ``τ``.

    ``τ_0 = δ_0`` (free location); for ``k ≥ 1``,
    ``τ_k = τ_0 + Σ_{j=1..k} (floor_j + softplus(δ_j))``.

    ``min_gap`` may be a scalar (same floor on every softplus gap) or a sequence
    of length ``T-1`` (per-gap floors: index 0 = first middle segment / Under,
    index 1 = next / Small, ...).
    """
    first = deltas[..., :1]
    if deltas.shape[-1] == 1:
        return first
    n_gaps = int(deltas.shape[-1] - 1)
    floors = _as_gap_floors(min_gap, n_gaps)
    # Broadcast floors onto leading dims of deltas.
    lead = deltas.ndim - 1
    floors_b = floors.reshape((1,) * lead + (n_gaps,))
    gaps = floors_b + jax.nn.softplus(deltas[..., 1:])
    return jnp.concatenate([first, first + jnp.cumsum(gaps, axis=-1)], axis=-1)
```

## 17. lognormal_prior_site

**Keep**

`model/cpds/lognormal.py:109`

Builds a SiteSpec with normalized parameters, output axes, family, and population policy.

```python
def lognormal_prior_site(
    resolved_bindings: Mapping[str, tuple[StructuredArray, ...]],
    *,
    hyperparameters: Mapping[str, Any],
    name: str,
) -> SiteSpec:
    """Phase 1 of ``lognormal.prior_v0``: describe the site without drawing it.

    The lognormal twin of ``normal.normal_prior_site``, and the reason the
    contract is duplication *removal* rather than opt-in overhead: the cell
    shaping the two modules each carried a verbatim copy of is now the
    framework's, and what remains here is only what is genuinely lognormal --
    the family closure and the level metadata.
    """
    del resolved_bindings  # root-style; the schema declares no roles
    return SiteSpec(
        build=_build_lognormal,
        params={
            "loc": jnp.asarray(hyperparameters["loc"]),
            "scale": jnp.asarray(hyperparameters["scale"]),
        },
        axes=tuple(hyperparameters["axes"]),
        indicator=IndicatorMeta(
            indicator=name,
            family=hyperparameters.get("family", "count"),
            unit="level",
        ),
        population=hyperparameters.get("population"),
    )
```

## 18. BayesGraph.__contains__

**Keep**

`core/graph.py:126`

Implements Python membership protocol for the domain graph.

```python
def __contains__(self, node_id: str) -> bool:
    return node_id in self._g
```

## 19. inline_javascript

**Keep**

`scripts/plot_v4_config.py:959`

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

## 20. SiteSpec.broadcast_params

**Keep**

`core/site_spec.py:155`

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
