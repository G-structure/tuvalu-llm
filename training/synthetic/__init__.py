"""Synthetic data generation: registry, loaders, selective translation."""

from .registry import get_loader, list_datasets, register

# Import loaders to trigger registration via @register decorators
from . import loaders as _loaders  # noqa: F401

__all__ = ["get_loader", "list_datasets", "register"]
