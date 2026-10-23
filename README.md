# heiz

A Python project using a `src` layout with a namespace package:

- namespace: `hm.egt`
- package: `hm.egt.heiz`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m hm.egt.heiz
```

## Example usage

```python
from hm.egt.heiz import __version__

print(__version__)
```
