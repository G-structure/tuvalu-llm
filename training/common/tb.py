"""Thin TensorBoard logging wrapper for training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TBLogger:
    """Lazy-initialised TensorBoard SummaryWriter.

    Writes scalars alongside the existing JSONL metrics files so that
    ``tensorboard --logdir logs/`` (or ``out/``) just works.

    Uses tensorboard's own SummaryWriter (no torch dependency required).
    """

    def __init__(self, log_dir: str | Path) -> None:
        self._log_dir = Path(log_dir)
        self._writer: Any | None = None

    def _ensure_writer(self) -> Any:
        if self._writer is None:
            from tensorboard.summary.writer.event_file_writer import EventFileWriter  # type: ignore
            from tensorboard.compat.proto.summary_pb2 import Summary  # type: ignore
            from tensorboard.compat.proto.event_pb2 import Event  # type: ignore

            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = EventFileWriter(str(self._log_dir))
            self._Summary = Summary
            self._Event = Event
        return self._writer

    def log_scalars(self, metrics: dict[str, float | int], step: int) -> None:
        """Write all numeric values in *metrics* as TensorBoard scalars."""
        w = self._ensure_writer()
        values = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                values.append(self._Summary.Value(tag=key, simple_value=float(value)))
        if values:
            summary = self._Summary(value=values)
            event = self._Event(step=step, summary=summary)
            import time
            event.wall_time = time.time()
            w.add_event(event)
            w.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
