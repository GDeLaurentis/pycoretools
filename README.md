# pycoretools

[![CI Lint](https://github.com/GDeLaurentis/pycoretools/actions/workflows/ci_lint.yml/badge.svg)](https://github.com/GDeLaurentis/pycoretools/actions/workflows/ci_lint.yml)
[![CI Test](https://github.com/GDeLaurentis/pycoretools/actions/workflows/ci_test.yml/badge.svg)](https://github.com/GDeLaurentis/pycoretools/actions/workflows/ci_test.yml)
[![Coverage](https://img.shields.io/badge/Coverage-58%25-orange?labelColor=2a2f35)](https://github.com/GDeLaurentis/pycoretools/actions)
[![PyPI](https://img.shields.io/pypi/v/pycoretools?label=PyPI)](https://pypi.org/project/pycoretools/)
[![Python](https://img.shields.io/pypi/pyversions/pycoretools?label=Python)](https://pypi.org/project/pycoretools/)

`pycoretools` is a lightweight collection of **generic, low-level utilities** that are reused across multiple Python projects.
It is designed to provide commonly needed building blocks without introducing heavy dependencies or project-specific assumptions.

This package exists primarily to **break dependency cycles** between higher-level libraries and to centralise small but widely useful tools. 

---

## Provided utilities

### Concurrency helpers

- `mapThreads`: a flexible `map`-like interface supporting
  - multiprocessing or threading
  - nested parallelism
  - progress reporting
  - optional checkpointing via pickled caches
- `filterThreads`: parallelised filtering built on top of `mapThreads`

These helpers are designed to work robustly in complex environments (e.g. nested pools, daemonic processes).

---

### Context management

- `TemporarySetting`: temporarily override a module attribute or namespace entry, e.g.

```python
import syngular
from pycoretools import TemporarySetting

with TemporarySetting(syngular, "TIMEOUT", 3600):
    ...
```

Useful for safely modifying global or module-level settings locally.

---

### Decorators

- `retry`: retry a function call on specified exceptions
- `with_cm`: apply a context manager as a decorator

---

### Iterable utilities

- `flatten`: controlled flattening of nested iterables, with support for
    - list subclasses
    - NumPy arrays
    - SymPy matrices
    - limited recursion depth
- `crease`: inverse operation to flatten, reconstructing nested structure from a template
- `chunks`: iterate over fixed-size chunks
- `all_non_empty_subsets`: generate all non-empty subsets of an iterable

---

### Sentinels

- `NaI` (“Not an Integer”): a singleton sentinel value that propagates through arithmetic operations, akin to NaN (“Not a Number“) for foats.

```
from pycoretools import NaI

NaI + 3   # NaI
NaI * 10  # NaI
```

---

### Installation

The package is available on the Python Package Index:

```
pip install pycoretools
```

For development:

```
git clone https://github.com/GDeLaurentis/pycoretools.git
pip install -e pycoretools
```

---

### Requirements

pycoretools itself depends only on the Python standard library.

Optional behaviour in some utilities may interact with:

- `numpy`
- `sympy`

These are not required dependencies.
