"""Compatibility shim for ``tv.training.common`` -> ``tv.common``."""

from importlib import import_module as _import_module

_TARGET = _import_module("tv.common")

__doc__ = _TARGET.__doc__
__all__ = getattr(_TARGET, "__all__", [])
__path__ = list(getattr(_TARGET, "__path__", []))

if __spec__ is not None:
    __spec__.submodule_search_locations = __path__


def __getattr__(name: str):
    return getattr(_TARGET, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_TARGET)))
