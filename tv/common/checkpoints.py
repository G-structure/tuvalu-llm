"""Checkpoint save/resume wrappers for Tinker training."""

from __future__ import annotations

import logging
from typing import Any

from .tinker_runtime import ensure_cookbook_on_path

logger = logging.getLogger(__name__)


def save_checkpoint(
    training_client: Any,
    name: str,
    log_path: str,
    kind: str = "both",
    loop_state: dict[str, Any] | None = None,
    ttl_seconds: int | None = 7 * 24 * 3600,
) -> dict[str, str]:
    """Save a Tinker checkpoint (state, weights, or both). Returns paths dict.

    Args:
        ttl_seconds: Time-to-live for the checkpoint. None = no expiry (permanent).
    """
    ensure_cookbook_on_path()
    from tinker_cookbook import checkpoint_utils  # type: ignore

    result = checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=name,
        log_path=log_path,
        kind=kind,
        loop_state=loop_state or {},
        ttl_seconds=ttl_seconds,
    )
    logger.info("Saved checkpoint '%s' (kind=%s) to %s", name, kind, log_path)
    return result


def get_last_checkpoint(log_path: str, **kwargs: Any) -> dict[str, Any] | None:
    """Get the last checkpoint info, or None if no checkpoint exists."""
    ensure_cookbook_on_path()
    from tinker_cookbook import checkpoint_utils  # type: ignore

    # Map our 'key' kwarg to tinker_cookbook's 'required_key' parameter
    if "key" in kwargs:
        kwargs["required_key"] = kwargs.pop("key")

    return checkpoint_utils.get_last_checkpoint(log_path, **kwargs)
