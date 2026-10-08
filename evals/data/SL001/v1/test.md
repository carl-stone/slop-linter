# Test examples

8 examples. Generated from `test.jsonl`.

**Flag** = needless helper; **Keep** = useful; **Review** = undecided.

## 1. argument-unpacking-1

**Flag**

Only unpacks existing arguments into one forwarded call.

```python
def call_pair(action, pair):
    return action(*pair)
```

## 2. argument-unpacking-2

**Flag**

Only unpacks existing arguments into one forwarded call.

```python
def call_options(action, options):
    return action(**options)
```

## 3. method-forwarding-1

**Flag**

The shown method only forwards to an existing object operation.

```python
class Sender:
    def send(self, value):
        return self.backend.send(value)
```

## 4. method-forwarding-2

**Flag**

The shown method only forwards to an existing object operation.

```python
class Reader:
    def fetch(self, key):
        return self.storage.fetch(key)
```

## 5. buffer-interface-1

**Keep**

Adapts a read operation into the readinto buffer/count protocol.

```python
def readinto(stream, buffer):
    chunk = stream.read(len(buffer))
    buffer[:len(chunk)] = chunk
    return len(chunk)
```

## 6. buffer-interface-2

**Keep**

Adapts a read operation into the readinto buffer/count protocol.

```python
def copy_into(reader, target):
    data = reader(len(target))
    target[:len(data)] = data
    return len(data)
```

## 7. decorator-behavior-1

**Keep**

The decorator adds caching or a context-manager interface beyond the forwarded call.

```python
from functools import lru_cache
@lru_cache(maxsize=32)
def lookup(loader, key):
    return loader(key)
```

## 8. decorator-behavior-2

**Keep**

The decorator adds caching or a context-manager interface beyond the forwarded call.

```python
from contextlib import contextmanager
@contextmanager
def opened(opener, path):
    resource = opener(path)
    try:
        yield resource
    finally:
        resource.close()
```
