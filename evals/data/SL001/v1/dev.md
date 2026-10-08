# Dev examples

8 examples. Generated from `dev.jsonl`.

**Flag** = needless helper; **Keep** = useful; **Review** = undecided.

## 1. async-alias-1

**Flag**

Only awaits and returns the corresponding asynchronous call.

```python
async def fetch(fetcher, key):
    return await fetcher(key)
```

## 2. async-alias-2

**Flag**

Only awaits and returns the corresponding asynchronous call.

```python
async def push(sender, value):
    return await sender(value)
```

## 3. constructor-wrapper-1

**Flag**

Only forwards construction with a preset argument.

```python
def make_rectangle(rectangle_type, width, height):
    return rectangle_type(width, height)
```

## 4. constructor-wrapper-2

**Flag**

Only forwards construction with a preset argument.

```python
def make_blue_widget(widget_type, size):
    return widget_type(size=size, color="blue")
```

## 5. exception-adapter-1

**Keep**

Translates an implementation exception into the caller-facing error contract.

```python
def load_record(loader, key):
    try:
        return loader(key)
    except KeyError as error:
        raise ValueError("record missing") from error
```

## 6. exception-adapter-2

**Keep**

Translates an implementation exception into the caller-facing error contract.

```python
def parse_number(parser, text):
    try:
        return parser(text)
    except ValueError as error:
        raise TypeError("invalid number") from error
```

## 7. resource-lifetime-1

**Keep**

Owns and closes a resource around the delegated operation.

```python
def read_bytes(path):
    with open(path, "rb") as stream:
        return stream.read()
```

## 8. resource-lifetime-2

**Keep**

Owns and closes a resource around the delegated operation.

```python
def read_text(path):
    with open(path, encoding="utf-8") as stream:
        return stream.read()
```
