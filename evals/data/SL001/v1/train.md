# Train examples

8 examples. Generated from `train.jsonl`.

**Flag** = needless helper; **Keep** = useful; **Review** = undecided.

## 1. direct-alias-1

**Flag**

Only renames the same call and preserves its arguments.

```python
def invoke(action, item):
    return action(item)
```

## 2. direct-alias-2

**Flag**

Only renames the same call and preserves its arguments.

```python
def run(action, *args, **kwargs):
    return action(*args, **kwargs)
```

## 3. fixed-argument-1

**Flag**

Only supplies fixed arguments to an existing callable.

```python
def reverse_sort(sorter, values):
    return sorter(values, reverse=True)
```

## 4. fixed-argument-2

**Flag**

Only supplies fixed arguments to an existing callable.

```python
def pretty_encode(encoder, value):
    return encoder(value, indent=2)
```

## 5. validation-1

**Keep**

Checks an invariant and rejects invalid input before delegating.

```python
def positive_transfer(sender, amount):
    if amount <= 0:
        raise ValueError("amount must be positive")
    return sender(amount)
```

## 6. validation-2

**Keep**

Checks an invariant and rejects invalid input before delegating.

```python
def nonempty_submit(submit, text):
    if not text.strip():
        raise ValueError("empty text")
    return submit(text)
```

## 7. normalization-1

**Keep**

Transforms data into a consistent useful representation.

```python
def canonical_tag(text):
    return "-".join(text.strip().lower().split())
```

## 8. normalization-2

**Keep**

Transforms data into a consistent useful representation.

```python
def clean_keys(mapping):
    return {key.strip().lower(): value for key, value in mapping.items()}
```
