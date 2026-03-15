"""Tinker client setup, cookbook path management, renderer selection."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from .config import get_repo_root

logger = logging.getLogger(__name__)


def ensure_cookbook_on_path() -> None:
    """Add vendor/tinker-cookbook to sys.path if it exists."""
    cookbook_root = get_repo_root() / "vendor" / "tinker-cookbook"
    if cookbook_root.exists():
        path_str = str(cookbook_root)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def require_tinker_api_key() -> str:
    """Return TINKER_API_KEY or raise SystemExit."""
    key = os.environ.get("TINKER_API_KEY")
    if not key:
        raise SystemExit("TINKER_API_KEY must be set in the environment.")
    return key


def get_renderer(model_name: str) -> tuple[Any, Any, str]:
    """Return (tokenizer, renderer, renderer_name) for a model.

    Imports tinker_cookbook lazily so the module works without it installed
    (for testing / data-only workflows).
    """
    ensure_cookbook_on_path()
    from tinker_cookbook import model_info, renderers  # type: ignore
    from tinker_cookbook.tokenizer_utils import get_tokenizer  # type: ignore

    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Using renderer %s for model %s", renderer_name, model_name)
    return tokenizer, renderer, renderer_name


def create_service_client(base_url: str | None = None) -> Any:
    """Create a tinker.ServiceClient."""
    ensure_cookbook_on_path()
    import tinker  # type: ignore

    return tinker.ServiceClient(base_url=base_url)


def create_lora_training_client(
    service_client: Any,
    model_name: str,
    lora_rank: int = 32,
) -> Any:
    """Create a LoRA training client."""
    logger.info("Creating LoRA training client for %s (rank=%d)", model_name, lora_rank)
    return service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank,
    )


def resume_training_client(
    service_client: Any,
    state_path: str,
) -> tuple[Any, int]:
    """Resume a training client from checkpoint state.

    Returns (training_client, start_step).
    """
    ensure_cookbook_on_path()
    from tinker_cookbook import checkpoint_utils  # type: ignore

    # First try nested checkpoint lookup (for compound paths)
    info = checkpoint_utils.get_last_checkpoint(state_path)
    if info:
        logger.info("Resuming from nested checkpoint: %s", info["state_path"])
        client = service_client.create_training_client_from_state_with_optimizer(
            info["state_path"]
        )
        start_step = int(info.get("step", info.get("batch", 0)))
        return client, start_step

    # Fall back to using the state_path directly
    logger.info("Resuming directly from state path: %s", state_path)
    client = service_client.create_training_client_from_state_with_optimizer(state_path)
    return client, 0


def create_sampling_client(
    service_client: Any,
    *,
    model_path: str | None = None,
    base_model: str | None = None,
) -> Any:
    """Create a sampling client from either a model_path or base_model."""
    if model_path:
        logger.info("Creating sampling client from model path: %s", model_path)
        return service_client.create_sampling_client(model_path=model_path)
    if base_model:
        logger.info("Creating sampling client from base model: %s", base_model)
        return service_client.create_sampling_client(base_model=base_model)
    raise ValueError("Provide either model_path or base_model")


def get_adam_params(learning_rate: float) -> Any:
    """Create tinker.AdamParams with standard defaults."""
    ensure_cookbook_on_path()
    import tinker  # type: ignore

    return tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )


def get_sampling_params(
    renderer: Any,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Any:
    """Create tinker.SamplingParams with renderer stop sequences."""
    ensure_cookbook_on_path()
    import tinker  # type: ignore

    return tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )
