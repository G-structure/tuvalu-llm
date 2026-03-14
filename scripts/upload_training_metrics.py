#!/usr/bin/env python3
"""Upload training metrics from local JSONL logs to Cloudflare D1.

Reads metrics from logs/tinker/stage_b_llama8b/metrics.jsonl and uploads
new entries to the talafutipolo D1 database via the Cloudflare HTTP API.

Usage:
    uv run python scripts/upload_training_metrics.py --watch       # poll every 30s (default)
    uv run python scripts/upload_training_metrics.py --once        # upload once and exit
    uv run python scripts/upload_training_metrics.py --init-schema # create tables first
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Cloudflare D1 config
ACCOUNT_ID = "8f86f0b518afefff58d515fe2a253b33"
DATABASE_ID = "7087ac6b-6417-48a4-9c7f-1d108057cd51"
D1_API_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}"
    f"/d1/database/{DATABASE_ID}/query"
)

# Local paths
METRICS_PATH = REPO_ROOT / "logs" / "tinker" / "stage_b_llama8b" / "metrics.jsonl"
MIX_STATS_PATH = REPO_ROOT / "data" / "finetune" / "stage_b_mix" / "stats.json"
UPLOAD_STATE_PATH = REPO_ROOT / ".upload_state.json"

RUN_ID = "stage_b_llama8b"
POLL_INTERVAL = 30  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# D1 HTTP helpers
# ---------------------------------------------------------------------------


def get_api_token() -> str:
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        sys.exit("ERROR: CLOUDFLARE_API_TOKEN env var not set")
    return token


def d1_query(sql_statements: list[dict], token: str) -> list[dict]:
    """Execute one or more SQL statements against D1.

    Each element should be {"sql": "...", "params": [...]}.
    The D1 API takes a single {sql, params} object per request,
    so we send each statement individually.
    """
    results = []
    for stmt in sql_statements:
        payload = json.dumps(stmt).encode()
        req = urllib.request.Request(
            D1_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            log.error("D1 API HTTP %d: %s", e.code, error_body)
            raise
        except urllib.error.URLError as e:
            log.error("D1 API network error: %s", e.reason)
            raise

        if not body.get("success"):
            errors = body.get("errors", [])
            log.error("D1 API returned errors: %s", errors)
            raise RuntimeError(f"D1 query failed: {errors}")

        results.extend(body.get("result", []))
    return results


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------


SCHEMA_SQL = [
    {
        "sql": (
            "CREATE TABLE IF NOT EXISTS training_metrics ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  run_id TEXT NOT NULL DEFAULT 'stage_b_llama8b',"
            "  step INTEGER NOT NULL,"
            "  metric_type TEXT NOT NULL,"
            "  value_json TEXT NOT NULL,"
            "  created_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        ),
        "params": [],
    },
    {
        "sql": "CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON training_metrics(run_id, step)",
        "params": [],
    },
    {
        "sql": "CREATE INDEX IF NOT EXISTS idx_metrics_type ON training_metrics(run_id, metric_type)",
        "params": [],
    },
    {
        "sql": (
            "CREATE TABLE IF NOT EXISTS training_config ("
            "  key TEXT PRIMARY KEY,"
            "  value_json TEXT NOT NULL,"
            "  updated_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        ),
        "params": [],
    },
]


def init_schema(token: str) -> None:
    log.info("Initializing D1 schema (%d statements)...", len(SCHEMA_SQL))
    d1_query(SCHEMA_SQL, token)
    log.info("Schema ready.")


# ---------------------------------------------------------------------------
# Metric classification
# ---------------------------------------------------------------------------


def classify_metric(entry: dict) -> str | None:
    """Return the metric_type for a log entry, or None if not a metric."""
    if "gen_eval_chrf_pp" in entry:
        return "gen_eval"
    if "validation_mean_nll" in entry:
        return "val_nll"
    if "train_nll" in entry or "train_mean_nll" in entry:
        return "train_nll"
    return None


# ---------------------------------------------------------------------------
# Upload state persistence
# ---------------------------------------------------------------------------


def load_upload_state() -> dict:
    if UPLOAD_STATE_PATH.exists():
        try:
            return json.loads(UPLOAD_STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt upload state file, starting fresh.")
    return {"last_line_count": 0, "last_mix_stats_mtime": 0}


def save_upload_state(state: dict) -> None:
    tmp = UPLOAD_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(UPLOAD_STATE_PATH)


# ---------------------------------------------------------------------------
# Read metrics log
# ---------------------------------------------------------------------------


def read_metrics_lines() -> list[str]:
    """Read all lines from the metrics JSONL, handling partial writes."""
    if not METRICS_PATH.exists():
        return []
    try:
        text = METRICS_PATH.read_text()
    except OSError as e:
        log.warning("Could not read metrics file: %s", e)
        return []
    lines = text.splitlines()
    # Drop the last line if it looks incomplete (no closing brace)
    if lines and not lines[-1].rstrip().endswith("}"):
        log.debug("Dropping incomplete last line")
        lines = lines[:-1]
    return lines


def parse_metric_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        log.debug("Skipping malformed JSON line: %.80s...", line)
        return None


# ---------------------------------------------------------------------------
# Upload logic
# ---------------------------------------------------------------------------


def upload_new_metrics(token: str, state: dict) -> dict:
    """Upload any new metric lines since the last upload. Returns updated state."""
    lines = read_metrics_lines()
    last = state.get("last_line_count", 0)

    if len(lines) <= last:
        log.debug("No new metric lines (%d total, %d already uploaded)", len(lines), last)
        return state

    new_lines = lines[last:]
    log.info("Found %d new metric lines (total %d, previously %d)", len(new_lines), len(lines), last)

    # Build batch of INSERT statements
    batch: list[dict] = []
    for line in new_lines:
        entry = parse_metric_line(line)
        if entry is None:
            continue
        metric_type = classify_metric(entry)
        if metric_type is None:
            # Still upload as a config/progress entry if it has step info
            if "current_step" in entry or "total_steps" in entry:
                _upload_run_info(token, entry)
            continue

        step = entry.get("step", entry.get("current_step", 0))
        batch.append(
            {
                "sql": (
                    "INSERT INTO training_metrics (run_id, step, metric_type, value_json) "
                    "VALUES (?, ?, ?, ?)"
                ),
                "params": [RUN_ID, step, metric_type, json.dumps(entry)],
            }
        )

    if batch:
        # D1 HTTP API doesn't support batch, so we use multi-row INSERT
        # SQLite supports INSERT INTO ... VALUES (...), (...), ...
        CHUNK_SIZE = 25  # 4 params each = 100 vars, D1 HTTP API is stricter
        for i in range(0, len(batch), CHUNK_SIZE):
            chunk = batch[i : i + CHUNK_SIZE]
            # Build a single multi-row INSERT
            placeholders = ", ".join(["(?, ?, ?, ?)"] * len(chunk))
            params = []
            for stmt in chunk:
                params.extend(stmt["params"])
            sql = (
                "INSERT INTO training_metrics (run_id, step, metric_type, value_json) "
                f"VALUES {placeholders}"
            )
            log.info(
                "Uploading rows %d-%d of %d...",
                i + 1,
                min(i + CHUNK_SIZE, len(batch)),
                len(batch),
            )
            d1_query([{"sql": sql, "params": params}], token)

    state["last_line_count"] = len(lines)
    save_upload_state(state)
    log.info("Upload state saved: %d lines processed", len(lines))
    return state


def _upload_run_info(token: str, entry: dict) -> None:
    """Upload run-level info (model name, total_steps, etc.) to training_config."""
    info = {}
    for key in ("model_name", "current_step", "total_steps", "run_id", "job_id"):
        if key in entry:
            info[key] = entry[key]
    if not info:
        return
    d1_query(
        [
            {
                "sql": (
                    "INSERT INTO training_config (key, value_json, updated_at) "
                    "VALUES ('run_info', ?, datetime('now')) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, "
                    "updated_at=datetime('now')"
                ),
                "params": [json.dumps(info)],
            }
        ],
        token,
    )


def upload_mix_stats(token: str, state: dict) -> dict:
    """Upload mix stats from stats.json to training_config if changed."""
    if not MIX_STATS_PATH.exists():
        log.debug("Mix stats file not found: %s", MIX_STATS_PATH)
        return state

    try:
        mtime = MIX_STATS_PATH.stat().st_mtime
    except OSError:
        return state

    if mtime <= state.get("last_mix_stats_mtime", 0):
        log.debug("Mix stats unchanged (mtime=%.0f)", mtime)
        return state

    try:
        stats = json.loads(MIX_STATS_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Could not read mix stats: %s", e)
        return state

    log.info("Uploading mix stats to training_config...")
    d1_query(
        [
            {
                "sql": (
                    "INSERT INTO training_config (key, value_json, updated_at) "
                    "VALUES ('mix_stats', ?, datetime('now')) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, "
                    "updated_at=datetime('now')"
                ),
                "params": [json.dumps(stats)],
            }
        ],
        token,
    )

    state["last_mix_stats_mtime"] = mtime
    save_upload_state(state)
    log.info("Mix stats uploaded.")
    return state


def upload_run_info_from_config(token: str) -> None:
    """Upload run-level info from the training config file."""
    config_path = REPO_ROOT / "configs" / "stage_b_agent_llama1b.json"
    if not config_path.exists():
        return
    try:
        cfg = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    # Read total steps from log or metrics
    lines = read_metrics_lines()
    max_step = 0
    for line in reversed(lines):
        entry = parse_metric_line(line)
        if entry and "step" in entry:
            max_step = max(max_step, entry["step"])
            break
    info = {
        "model_name": cfg.get("model", {}).get("name", "Qwen/Qwen3-30B-A3B"),
        "total_steps": 45561,
        "current_step": max_step,
        "run_id": RUN_ID,
    }
    d1_query(
        [
            {
                "sql": (
                    "INSERT INTO training_config (key, value_json, updated_at) "
                    "VALUES ('run_info', ?, datetime('now')) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, "
                    "updated_at=datetime('now')"
                ),
                "params": [json.dumps(info)],
            }
        ],
        token,
    )
    log.info("Run info uploaded: step %d, model %s", max_step, info["model_name"])


def run_upload_cycle(token: str, state: dict) -> dict:
    """Run one full upload cycle: metrics + mix stats + run info."""
    try:
        state = upload_new_metrics(token, state)
    except Exception as e:
        log.error("Failed to upload metrics: %s", e)

    try:
        state = upload_mix_stats(token, state)
    except Exception as e:
        log.error("Failed to upload mix stats: %s", e)

    try:
        upload_run_info_from_config(token)
    except Exception as e:
        log.error("Failed to upload run info: %s", e)

    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload training metrics to Cloudflare D1"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Upload once and exit")
    mode.add_argument(
        "--watch",
        action="store_true",
        default=True,
        help="Poll every 30s for new entries (default)",
    )
    parser.add_argument(
        "--init-schema",
        action="store_true",
        help="Create D1 tables before uploading",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset upload state (re-upload everything)",
    )
    args = parser.parse_args()

    token = get_api_token()

    if args.init_schema:
        init_schema(token)

    if args.reset:
        if UPLOAD_STATE_PATH.exists():
            UPLOAD_STATE_PATH.unlink()
            log.info("Upload state reset.")

    state = load_upload_state()

    if args.once:
        state = run_upload_cycle(token, state)
        log.info("Done (--once mode).")
        return

    # Watch mode
    log.info(
        "Watching %s (poll every %ds, Ctrl-C to stop)...",
        METRICS_PATH,
        POLL_INTERVAL,
    )
    try:
        while True:
            state = run_upload_cycle(token, state)
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
