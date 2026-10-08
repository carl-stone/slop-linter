# Test examples

10 examples. Generated from `test.jsonl`.

**Flag** = needless helper; **Keep** = useful; **Review** = undecided.

## 1. _axis_pos

**Keep**

[model/cpds/dirichlet.py:280](/Users/carlstone/digital-planet/digital_readiness/model/cpds/dirichlet.py:280)

Adds an actionable error naming the node, parent role, missing axis, and available axes.

Paired with `synthetic:axis-position-forwarder`.

```python
def _axis_pos(sa: StructuredArray, axis: str, node: str, role: str) -> int:
    """Index of ``axis`` in ``sa``, or a clear error naming what's available."""
    try:
        return sa.axis_names.index(axis)
    except ValueError:
        raise ValueError(
            f"dirichlet.segment_v0 ({node!r}): {role} parent has no axis "
            f"{axis!r}; axes are {list(sa.axis_names)}."
        ) from None
```

## 2. axis-position-forwarder

**Flag**

Constructed counterpart.

Only extracts axis_names and forwards index lookup; the node and role arguments add no behavior.

Paired with `model/cpds/dirichlet.py:280:_axis_pos`.

```python
def _axis_pos(sa: StructuredArray, axis: str, node: str, role: str) -> int:
    return sa.axis_names.index(axis)
```

## 3. BayesGraph.add_edge

**Keep**

[core/graph.py:93](/Users/carlstone/digital-planet/digital_readiness/core/graph.py:93)

Checks that both nodes exist before adding an edge, producing a useful error instead of silently creating unknown nodes.

Paired with `synthetic:graph-edge-forwarder`.

```python
def add_edge(self, parent: str, child: str) -> None:
    for n in (parent, child):
        if n not in self._g:
            raise KeyError(f"unknown node: {n}")
    self._g.add_edge(parent, child)
```

## 4. graph-edge-forwarder

**Flag**

Constructed counterpart.

Only forwards the edge addition. Unlike the real method, it adds no endpoint validation.

Paired with `core/graph.py:93:BayesGraph.add_edge`.

```python
def add_edge(self, parent: str, child: str) -> None:
    self._g.add_edge(parent, child)
```

## 5. BayesGraph.__len__

**Keep**

[core/graph.py:129](/Users/carlstone/digital-planet/digital_readiness/core/graph.py:129)

Implements the required Python length protocol; the special method makes len(graph) work.

Paired with `synthetic:graph-node-count-alias`.

```python
def __len__(self) -> int:
    return self._g.number_of_nodes()
```

## 6. graph-node-count-alias

**Flag**

Constructed counterpart.

An ordinary method merely renames number_of_nodes; unlike __len__, it implements no Python protocol.

Paired with `core/graph.py:129:BayesGraph.__len__`.

```python
def node_count(self) -> int:
    return self._g.number_of_nodes()
```

## 7. _coerce_str_tuple

**Keep**

[core/observation_cell.py:20](/Users/carlstone/digital-planet/digital_readiness/core/observation_cell.py:20)

Preserves None and existing tuples and treats a string as one element rather than splitting it into characters.

Paired with `synthetic:string-tuple-cast-alias`.

```python
def _coerce_str_tuple(value: Any) -> tuple[str, ...] | None:
    """Normalise a unit or aggregation_geo field to a tuple of strings."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        return (value,)
    return tuple(value)
```

## 8. string-tuple-cast-alias

**Flag**

Constructed counterpart.

Only wraps the tuple constructor and provides none of the paired normalization policy.

Paired with `core/observation_cell.py:20:_coerce_str_tuple`.

```python
def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    return tuple(value)
```

## 9. load_graph_config

**Keep**

[io/pipeline.py:53](/Users/carlstone/digital-planet/digital_readiness/io/pipeline.py:53)

Owns file opening and closing, parses YAML, and validates the resulting configuration.

Paired with `synthetic:graph-config-validation-alias`.

```python
def load_graph_config(path: str | Path) -> GraphConfig:
    """Load and validate a graph YAML file."""
    config_path = Path(path)
    with config_path.open() as f:
        raw = YAML().load(f)
    return GraphConfig.model_validate(raw)
```

## 10. graph-config-validation-alias

**Flag**

Constructed counterpart.

Only forwards a raw configuration to its existing validator; it performs no loading or other adaptation.

Paired with `io/pipeline.py:53:load_graph_config`.

```python
def load_graph_config(raw) -> GraphConfig:
    return GraphConfig.model_validate(raw)
```
