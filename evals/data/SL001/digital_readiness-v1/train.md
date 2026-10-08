# Train examples

22 examples. Generated from `train.jsonl`.

**Flag** = needless helper; **Keep** = useful; **Review** = undecided.

## 1. BayesGraph.__contains__

**Keep**

[core/graph.py:126](/Users/carlstone/digital-planet/digital_readiness/core/graph.py:126)

Implements Python membership protocol for the domain graph.

```python
def __contains__(self, node_id: str) -> bool:
    return node_id in self._g
```

## 2. Literal.mask_for

**Keep**

[core/observations.py:141](/Users/carlstone/digital-planet/digital_readiness/core/observations.py:141)

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

## 3. _finite_number

**Keep**

[inference/nuts_runner.py:112](/Users/carlstone/digital-planet/digital_readiness/inference/nuts_runner.py:112)

Normalizes numerical diagnostics to finite floats or JSON-safe absence.

```python
def _finite_number(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None
```

## 4. _synthesized_theta_weight_loc

**Keep**

[io/config/build.py:517](/Users/carlstone/digital-planet/digital_readiness/io/config/build.py:517)

Transforms AxisMeta objects into the axis-name and level pairs accepted by shared inference.

```python
def _synthesized_theta_weight_loc(
    config: GraphConfig, axes: tuple[AxisMeta, ...]
) -> float | list:
    """Adapter: ``desugar.synthesized_weight_loc`` over resolved ``AxisMeta``."""
    return synthesized_weight_loc(config, [(ax.name, ax.levels) for ax in axes])
```

## 5. _select

**Keep**

[io/observation_loader.py:842](/Users/carlstone/digital-planet/digital_readiness/io/observation_loader.py:842)

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

## 6. binomial_observation

**Keep**

[model/cpds/binomial.py:69](/Users/carlstone/digital-planet/digital_readiness/model/cpds/binomial.py:69)

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

## 7. lognormal_prior_site

**Keep**

[model/cpds/lognormal.py:109](/Users/carlstone/digital-planet/digital_readiness/model/cpds/lognormal.py:109)

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

## 8. _build_lognormal

**Flag**

[model/cpds/lognormal.py:98](/Users/carlstone/digital-planet/digital_readiness/model/cpds/lognormal.py:98)

Forwards unchanged keyword arguments to dist.LogNormal; the distribution class can be the callable directly.

```python
def _build_lognormal(**params):
    """The family closure a ``SiteSpec`` carries (see ``core/site_spec.py``).

    Module-level so its identity is stable: ``bundle_layout`` uses ``build``
    identity to refuse cross-family bundles -- which is what keeps a lognormal
    parameter field out of a normal bundle even though both declare
    ``{loc, scale}``.
    """
    return dist.LogNormal(**params)
```

## 9. _build_normal

**Flag**

[model/cpds/normal.py:75](/Users/carlstone/digital-planet/digital_readiness/model/cpds/normal.py:75)

Forwards unchanged keyword arguments to dist.Normal; the distribution class can be the callable directly.

```python
def _build_normal(**params):
    """The family closure a ``SiteSpec`` carries (see ``core/site_spec.py``).

    Module-level so its identity is stable: ``bundle_layout`` uses ``build``
    identity to refuse cross-family bundles.
    """
    return dist.Normal(**params)
```

## 10. ordered_thresholds_from_deltas

**Keep**

[model/cpds/ordered_threshold.py:64](/Users/carlstone/digital-planet/digital_readiness/model/cpds/ordered_threshold.py:64)

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

## 11. _resolve_standardize_bindings

**Keep**

[model/cpds/standardize.py:135](/Users/carlstone/digital-planet/digital_readiness/model/cpds/standardize.py:135)

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

## 12. NormalDist

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

## 13. NormalDist

**Flag**

User-supplied example.

Only renames the Normal constructor while forwarding the same parameters.

```python
def NormalDist(mu, sigma):
    return dist.Normal(mu, sigma)
```

## 14. NormalDist

**Flag**

User-supplied example.

Hides a zero-mean argument in a constructor alias without expressing a distinct useful policy.

```python
def NormalDist(sigma):
    return dist.Normal(0, sigma)
```

## 15. NormalDist

**Flag**

User-supplied example.

Local aliases add no behavior; this is the same unnecessary constructor forwarding across several lines.

```python
def NormalDist(mu, sigma):
    m = mu
    sig = sigma
    return dist.Normal(m, sig)
```

## 16. StructuredArray.with_node_id

**Flag**

[core/structured_array.py:550](/Users/carlstone/digital-planet/digital_readiness/core/structured_array.py:550)

Only wraps dataclasses.replace to set node_id; the attribution name adds no behavior beyond the direct replacement.

```python
def with_node_id(self, node_id: str | None) -> "StructuredArray":
    """This value, attributed to (or detached from) a producing node.

    The stamping seam. ``node_id=None`` detaches, which is what bundle
    projection wants: a ``(consumer, role)`` block is not a node's output.
    """
    return replace(self, node_id=node_id)
```

## 17. load_demo_cross_section

**Flag**

[io/observation_loader.py:1094](/Users/carlstone/digital-planet/digital_readiness/io/observation_loader.py:1094)

Only sets a same-year filter before calling the existing loader; this demo convenience adds no loading behavior.

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

## 18. load_pct_smartphone_slice

**Flag**

[io/observation_loader.py:1117](/Users/carlstone/digital-planet/digital_readiness/io/observation_loader.py:1117)

Only presets indicator, source, year, unit, and status filters for a specific demo slice before forwarding to the existing loader; it adds no general loading behavior.

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

## 19. has_explicit_observation_nodes

**Flag**

[io/pipeline.py:194](/Users/carlstone/digital-planet/digital_readiness/io/pipeline.py:194)

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

## 20. _resolve_beta_observation_bindings

**Flag**

[model/cpds/observations/beta.py:142](/Users/carlstone/digital-planet/digital_readiness/model/cpds/observations/beta.py:142)

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

## 21. _resolve_lognormal_observation_bindings

**Flag**

[model/cpds/observations/lognormal.py:118](/Users/carlstone/digital-planet/digital_readiness/model/cpds/observations/lognormal.py:118)

Forwards unchanged arguments to resolve_observation_bindings, which already has the required callback signature.

```python
def _resolve_lognormal_observation_bindings(
    *,
    child: NodeMeta,
    parents: tuple[NodeMeta, ...],
    hyperparameters: Mapping[str, Any],
) -> Mapping[str, tuple[str, ...]]:
    return resolve_observation_bindings(
        child=child, parents=parents, hyperparameters=hyperparameters
    )
```

## 22. _resolve_normal_observation_bindings

**Flag**

[model/cpds/observations/normal.py:99](/Users/carlstone/digital-planet/digital_readiness/model/cpds/observations/normal.py:99)

Forwards unchanged arguments to resolve_observation_bindings, which already has the required callback signature.

```python
def _resolve_normal_observation_bindings(
    *,
    child: NodeMeta,
    parents: tuple[NodeMeta, ...],
    hyperparameters: Mapping[str, Any],
) -> Mapping[str, tuple[str, ...]]:
    return resolve_observation_bindings(
        child=child, parents=parents, hyperparameters=hyperparameters
    )
```
