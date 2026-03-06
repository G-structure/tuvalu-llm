"""Test TBLogger robustness."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.tb import TBLogger


def test_numeric_scalar_filtering():
    """Verify non-numeric values are silently skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tb = TBLogger(tmpdir)
        # Should not raise - strings, None, bools, lists are skipped
        tb.log_scalars({"loss": 0.5, "name": "test", "flag": True, "items": [1, 2]}, step=0)
        tb.log_scalars({"lr": 1e-4, "none_val": None}, step=1)
        tb.close()


def test_close_idempotent():
    """close() twice doesn't error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tb = TBLogger(tmpdir)
        tb.log_scalars({"x": 1.0}, step=0)
        tb.close()
        tb.close()
        tb.close()


def test_no_exception_on_mixed_metrics():
    """Dict with strings, None, lists, nested dicts doesn't crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tb = TBLogger(tmpdir)
        mixed = {
            "loss": 0.123,
            "accuracy": 95,
            "model_name": "qwen-30b",
            "config": {"lr": 0.001},
            "tags": ["a", "b"],
            "nothing": None,
            "is_best": True,
        }
        tb.log_scalars(mixed, step=0)
        tb.close()


def test_context_manager():
    """with TBLogger() as tb: ... works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with TBLogger(tmpdir) as tb:
            tb.log_scalars({"val": 42.0}, step=0)
            tb.log_text("info", "hello world", step=0)
        # After exiting context, writer should be closed (no error on re-close)
        tb.close()


def test_log_text():
    """log_text writes without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with TBLogger(tmpdir) as tb:
            tb.log_text("model", "Qwen/Qwen3-30B-A3B-Base", step=0)
            tb.log_text("config", '{"lr": 0.001}', step=1)
