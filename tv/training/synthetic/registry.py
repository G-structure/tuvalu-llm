"""Dataset registry: maps dataset names to loader functions."""

from __future__ import annotations

from typing import Any, Callable, Iterator

# Loader signature: (split, limit) -> Iterator[dict]
LoaderFn = Callable[..., Iterator[dict[str, Any]]]

_REGISTRY: dict[str, LoaderFn] = {}


def register(name: str) -> Callable[[LoaderFn], LoaderFn]:
    """Decorator to register a dataset loader under *name*."""

    def _wrap(fn: LoaderFn) -> LoaderFn:
        if name in _REGISTRY:
            raise ValueError(f"Duplicate dataset registration: {name!r}")
        _REGISTRY[name] = fn
        return fn

    return _wrap


def get_loader(name: str) -> LoaderFn:
    """Return the loader function for *name*, or raise KeyError."""
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown dataset {name!r}. Available: {available}")


def list_datasets() -> list[str]:
    """Return sorted list of registered dataset names."""
    return sorted(_REGISTRY)
