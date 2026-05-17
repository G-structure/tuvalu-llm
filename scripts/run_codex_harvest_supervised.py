"""Supervised codex harvest with stall detection and auto-resume.

Wraps ``scripts/run_codex_harvest.py``. While the child harvest runs,
watch the per-rollout sink ``<out>/traces/traces.jsonl`` for growth.
If it hasn't grown in ``--stall-minutes`` minutes, terminate the child
+ any orphaned ``codex app-server`` subprocesses, count how many rows
landed before the stall, and re-spawn for the remaining tasks under a
new seed. Repeat until the total accepted row target is met or
``--max-resumes`` is exhausted.

Why this exists: the 2026-05-15 nc500 harvest hung at row 127 because
``codex_orchestrate.pools.codex_session`` doesn't propagate per-call
timeouts down to the StdioTransport, so a stuck rollout drops the
child PPID to 1 and the Python parent sleeps forever waiting on EOF.
A real fix lives in codex_orchestrate; this supervisor is the
mechanical workaround so we can leave a 5000-row harvest running
overnight without losing it to one stuck rollout.

Usage::

    uv run python scripts/run_codex_harvest_supervised.py \\
        --task-family native_chat \\
        --use-personas \\
        --bucket low --bucket med \\
        --n 1000 \\
        --out runs/codex_harvest_persona_1000 \\
        --stall-minutes 15 --max-resumes 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _kill_descendants(pid: int) -> list[int]:
    """Return PIDs of *all* descendants (including orphaned codex
    processes whose parent has been reparented to PID 1) we'd want to
    take down with the child harvest. Conservative — only finds nodes
    we can match by command name."""
    candidates: list[int] = []
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,comm="],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return candidates
    rows: list[tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            p, pp = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        rows.append((p, pp, parts[2]))

    # Walk down from pid to all descendants.
    by_pp: dict[int, list[tuple[int, str]]] = {}
    for p, pp, comm in rows:
        by_pp.setdefault(pp, []).append((p, comm))
    seen: set[int] = set()
    stack = [pid]
    while stack:
        cur = stack.pop()
        for child, _comm in by_pp.get(cur, []):
            if child not in seen:
                seen.add(child)
                candidates.append(child)
                stack.append(child)

    # Also catch any orphaned `codex` processes (parent=1).
    for p, pp, comm in rows:
        if pp == 1 and ("codex" in comm or "VLLM" in comm):
            # Only if recent — heuristic: if no other harvest is running,
            # any codex orphan is ours. The supervisor is single-tenant
            # by design.
            if p not in seen:
                seen.add(p)
                candidates.append(p)
    return candidates


def _terminate_tree(pid: int, *, log: logging.Logger) -> None:
    """Politely-then-forcefully kill the harvest tree."""
    descendants = _kill_descendants(pid)
    pids_to_kill = [pid, *descendants]
    log.warning("terminating: %s", pids_to_kill)
    for p in pids_to_kill:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            log.warning("permission denied terminating %d", p)
    deadline = time.time() + 6.0
    while time.time() < deadline:
        alive = []
        for p in pids_to_kill:
            try:
                os.kill(p, 0)
                alive.append(p)
            except ProcessLookupError:
                pass
        if not alive:
            return
        time.sleep(0.5)
    for p in pids_to_kill:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _count_rows(traces: Path) -> int:
    if not traces.exists():
        return 0
    try:
        with traces.open("rb") as fp:
            return sum(1 for _ in fp)
    except OSError:
        return 0


def _spawn_harvest(
    *,
    task_family: str,
    n: int,
    out: Path,
    seed: int,
    teacher_model: str,
    use_personas: bool,
    buckets: list[str] | None,
    max_density: float | None,
    direction: str | None,
    extra_args: list[str],
    log: logging.Logger,
) -> subprocess.Popen[str]:
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "run_codex_harvest.py"),
        "--task-family", task_family,
        "--n", str(n),
        "--out", str(out),
        "--teacher-model", teacher_model,
        "--seed", str(seed),
    ]
    if use_personas:
        cmd.append("--use-personas")
    for b in (buckets or []):
        cmd.extend(["--bucket", b])
    if max_density is not None:
        cmd.extend(["--max-religious-density", str(max_density)])
    if direction:
        cmd.extend(["--direction", direction])
    cmd.extend(extra_args)
    log.info("spawning: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(ROOT),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-family",
        choices=["native_chat", "hard_translation", "qa_grounded", "reframe"],
        required=True,
    )
    parser.add_argument("--n", type=int, required=True,
                        help="total accepted-row target across all resumes")
    parser.add_argument("--out", type=Path, required=True,
                        help="root output dir; per-attempt dirs land at <out>/attempt_NN/")
    parser.add_argument("--teacher-model", default="gpt-5.3-codex")
    parser.add_argument("--use-personas", action="store_true")
    parser.add_argument("--bucket", action="append", default=None,
                        choices=["low", "med", "high"])
    parser.add_argument("--max-religious-density", type=float, default=None)
    parser.add_argument("--direction", choices=["tvl2en", "en2tvl"], default=None)
    parser.add_argument(
        "--stall-minutes", type=float, default=15.0,
        help="if traces.jsonl hasn't grown in this many minutes, kill+resume",
    )
    parser.add_argument(
        "--max-resumes", type=int, default=10,
        help="hard cap on resume attempts (1 attempt + N resumes)",
    )
    parser.add_argument(
        "--seed-base", type=int, default=1,
        help="seed for attempt 0; attempt K uses seed_base + K",
    )
    parser.add_argument(
        "--poll-seconds", type=float, default=30.0,
        help="how often to check traces.jsonl growth",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s sup: %(message)s",
    )
    log = logging.getLogger("supervisor")
    log.info(
        "target=%d task_family=%s stall_minutes=%.1f max_resumes=%d",
        args.n, args.task_family, args.stall_minutes, args.max_resumes,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    combined_traces = args.out / "traces" / "traces.jsonl"
    combined_traces.parent.mkdir(parents=True, exist_ok=True)
    # Start fresh — supervisor owns the combined sink.
    if combined_traces.exists():
        log.info("removing existing combined sink %s", combined_traces)
        combined_traces.unlink()

    total_accepted = 0
    total_rows = 0
    attempt = 0
    extra_args: list[str] = []

    while total_accepted < args.n and attempt <= args.max_resumes:
        remaining_target = args.n - total_accepted
        # Over-request a bit because some rows will be rejected.
        attempt_n = int(remaining_target * 1.05) + 1
        attempt_dir = args.out / f"attempt_{attempt:02d}"
        attempt_traces = attempt_dir / "traces" / "traces.jsonl"
        seed = args.seed_base + attempt

        log.info(
            "attempt %d: targeting %d more accepted rows (raw n=%d, seed=%d)",
            attempt, remaining_target, attempt_n, seed,
        )

        proc = _spawn_harvest(
            task_family=args.task_family,
            n=attempt_n,
            out=attempt_dir,
            seed=seed,
            teacher_model=args.teacher_model,
            use_personas=args.use_personas,
            buckets=args.bucket,
            max_density=args.max_religious_density,
            direction=args.direction,
            extra_args=extra_args,
            log=log,
        )

        # Drain child stdout into the supervisor log + a per-attempt file.
        attempt_log = attempt_dir / "child.log"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        stdout_thread_target = attempt_log

        # Monitor loop. We poll every `poll_seconds` and check:
        # 1. Did child exit cleanly? If yes, break.
        # 2. Did traces.jsonl grow within the stall window? If no, kill.
        last_row_count = 0
        last_growth = time.time()
        rows_added_this_attempt = 0
        with attempt_log.open("w", encoding="utf-8") as logfp:
            while True:
                # Drain stdout (non-blocking — we set bufsize=1 so this
                # gets at most a few lines per poll cycle).
                if proc.stdout is not None:
                    proc.stdout.flush()
                # Has child exited?
                rc = proc.poll()
                rows_now = _count_rows(attempt_traces)
                if rows_now > last_row_count:
                    last_row_count = rows_now
                    last_growth = time.time()
                if rc is not None:
                    log.info(
                        "attempt %d child exited rc=%d, final rows=%d",
                        attempt, rc, rows_now,
                    )
                    break
                stall = time.time() - last_growth
                if stall > args.stall_minutes * 60.0:
                    log.warning(
                        "attempt %d STALLED — %.1f min since last row (have %d). Killing.",
                        attempt, stall / 60.0, rows_now,
                    )
                    _terminate_tree(proc.pid, log=log)
                    # Give the OS a moment to release file handles.
                    time.sleep(2.0)
                    break
                time.sleep(args.poll_seconds)

            # Drain whatever child stdout we have buffered (the log
            # already captured per-line via subprocess buffering).
            try:
                tail, _ = proc.communicate(timeout=5.0)
                if tail:
                    logfp.write(tail)
            except subprocess.TimeoutExpired:
                _terminate_tree(proc.pid, log=log)
                tail, _ = proc.communicate(timeout=5.0)
                if tail:
                    logfp.write(tail)

        # Count accepted rows landed this attempt.
        attempted_rows = []
        if attempt_traces.exists():
            with attempt_traces.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    attempted_rows.append(row)
        attempt_accepted = sum(1 for r in attempted_rows if r.get("accepted"))
        log.info(
            "attempt %d: wrote %d rows, %d accepted",
            attempt, len(attempted_rows), attempt_accepted,
        )

        # Merge into the combined sink.
        if attempted_rows:
            with combined_traces.open("a", encoding="utf-8") as fp_out:
                for row in attempted_rows:
                    fp_out.write(json.dumps(row, ensure_ascii=False) + "\n")

        total_rows += len(attempted_rows)
        total_accepted += attempt_accepted
        attempt += 1
        if attempt_accepted == 0 and attempt > 1:
            log.warning(
                "attempt %d yielded zero accepted rows — stopping early to avoid burn",
                attempt - 1,
            )
            break

    log.info(
        "supervisor done: %d total accepted (target %d) across %d attempts",
        total_accepted, args.n, attempt,
    )
    summary = {
        "target_accepted": args.n,
        "total_accepted": total_accepted,
        "total_rows_attempted": total_rows,
        "attempts": attempt,
        "combined_traces": str(combined_traces),
    }
    (args.out / "supervisor_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if total_accepted >= args.n else 2


if __name__ == "__main__":
    raise SystemExit(main())
