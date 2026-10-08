# Review examples

4 examples. Generated from `review.jsonl`.

**Flag** = needless helper; **Keep** = useful; **Review** = undecided.

## 1. public-facade-1

**Review**

A public compatibility boundary may be meaningful, but the snippet lacks caller/version context.

```python
def public_load(implementation, key):
    """Stable public loading API."""
    return implementation(key)
```

## 2. public-facade-2

**Review**

A public compatibility boundary may be meaningful, but the snippet lacks caller/version context.

```python
def legacy_submit(new_submit, value):
    """Compatibility entry point for older callers."""
    return new_submit(value)
```

## 3. policy-default-1

**Review**

A fixed argument may encode an important policy; usefulness cannot be decided from this snippet alone.

```python
def safe_connect(connect, address):
    return connect(address, verify_certificate=True)
```

## 4. policy-default-2

**Review**

A fixed argument may encode an important policy; usefulness cannot be decided from this snippet alone.

```python
def tenant_read(reader, tenant, key):
    return reader(key, tenant=tenant)
```
