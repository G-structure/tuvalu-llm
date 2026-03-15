#!/usr/bin/env python3
"""CLI wrapper for the corpus split pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tv.corpus.splits import main


if __name__ == "__main__":
    main()
