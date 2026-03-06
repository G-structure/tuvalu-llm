"""Thin TensorBoard logging wrapper for training runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class TBLogger:
    """Lazy-initialised TensorBoard SummaryWriter.

    Writes scalars alongside the existing JSONL metrics files so that
    ``tensorboard --logdir logs/`` (or ``out/``) just works.

    Uses tensorboard's own SummaryWriter (no torch dependency required).
    Supports use as a context manager.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self._log_dir = Path(log_dir)
        self._writer: Any | None = None

    def __enter__(self) -> "TBLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

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

    def log_scalars(self, metrics: dict[str, Any], step: int) -> None:
        """Write numeric values in *metrics* as TensorBoard scalars.

        Non-numeric values (strings, None, lists, dicts, etc.) are
        silently skipped.
        """
        w = self._ensure_writer()
        values = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(self._Summary.Value(tag=key, simple_value=float(value)))
        if values:
            summary = self._Summary(value=values)
            event = self._Event(step=step, summary=summary)
            event.wall_time = time.time()
            w.add_event(event)
            w.flush()

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Write a text value as a TensorBoard text summary."""
        w = self._ensure_writer()
        from tensorboard.compat.proto.summary_pb2 import SummaryMetadata  # type: ignore
        from tensorboard.compat.proto.tensor_pb2 import TensorProto  # type: ignore
        from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto  # type: ignore

        tensor = TensorProto(
            dtype=7,  # DT_STRING
            string_val=[text.encode("utf-8")],
            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
        )
        metadata = SummaryMetadata(
            plugin_data=SummaryMetadata.PluginData(plugin_name="text"),
        )
        value = self._Summary.Value(tag=tag, metadata=metadata, tensor=tensor)
        summary = self._Summary(value=[value])
        event = self._Event(step=step, summary=summary)
        event.wall_time = time.time()
        w.add_event(event)
        w.flush()

    def close(self) -> None:
        """Close the writer. Safe to call multiple times."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
