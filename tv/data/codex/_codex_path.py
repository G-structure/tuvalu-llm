"""Add the sibling ``rl-agent-work/src/*`` packages to sys.path.

The codex harness substrate (``codex_orchestrate``, ``codex_control``,
``codex_env``, ``codex_proxy``) lives outside this repo. Importing it
without installing as wheels keeps the dependency chain explicit and
matches the Phase 4.5 design where the harness is in a sibling
workspace.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_codex_packages_on_path() -> Path:
    """Insert the rl-agent-work codex packages into sys.path. Returns the
    workspace root. Idempotent."""
    # tuvalu-llm/tv/data/codex/_codex_path.py
    # → tuvalu-llm/tv/data/codex/  → tv/data/codex/  → tv/data/  → tv/  → tuvalu-llm/
    repo_root = Path(__file__).resolve().parents[3]
    workspace = repo_root.parent          # parent of tuvalu-llm == rl-agent-work
    src = workspace / "src"
    for pkg in (
        "codex-control",
        "codex-proxy",
        "codex-env",
        "codex-orchestrate",
        "codex-train",
    ):
        p = src / pkg / "src"
        if p.is_dir():
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)
    return workspace


__all__ = ["ensure_codex_packages_on_path"]
