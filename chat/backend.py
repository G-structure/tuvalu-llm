#!/usr/bin/env python3
"""Tinker sampling backend for TVL Chat.

Connects to the latest Stage B sampler weights and serves a chat API.
Auto-refreshes the model when new sampler checkpoints appear.

Usage:
    uv run python chat/backend.py
    uv run python chat/backend.py --port 8787
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.common.tinker_runtime import (
    create_service_client,
    ensure_cookbook_on_path,
    get_renderer,
    get_sampling_params,
    require_tinker_api_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("chat-backend")

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
CHECKPOINTS_FILE = REPO_ROOT / "logs" / "tinker" / "stage_b_llama8b" / "checkpoints.jsonl"


class ModelState:
    """Thread-safe model state that auto-refreshes from checkpoints."""

    def __init__(self):
        self.service = None
        self.sampling_client = None
        self.renderer = None
        self.sampling_params = None
        self.sampler_path: str | None = None
        self.step: str = "?"
        self.lock = threading.Lock()

    def init(self):
        require_tinker_api_key()
        _tokenizer, self.renderer, name = get_renderer(MODEL_NAME)
        logger.info("Renderer: %s", name)
        self.service = create_service_client()
        self.refresh()

    def get_latest_sampler(self) -> tuple[str | None, str]:
        if not CHECKPOINTS_FILE.exists():
            return None, "?"
        sampler_path = None
        step = "?"
        for line in CHECKPOINTS_FILE.read_text().strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            if "sampler_path" in entry:
                sampler_path = entry["sampler_path"]
                step = entry.get("name", "?")
        return sampler_path, step

    def refresh(self) -> bool:
        path, step = self.get_latest_sampler()
        if not path:
            logger.warning("No sampler checkpoint found")
            return False
        if path == self.sampler_path:
            return False

        with self.lock:
            logger.info("Loading sampler: %s (step %s)", path, step)
            self.sampling_client = self.service.create_sampling_client(model_path=path)
            self.sampling_params = get_sampling_params(
                self.renderer, max_tokens=1024, temperature=0.3
            )
            self.sampler_path = path
            self.step = step
            logger.info("Model ready at step %s", step)
        return True

    def sample(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
        with self.lock:
            if not self.sampling_client:
                raise RuntimeError("Model not loaded")

            params = get_sampling_params(
                self.renderer, max_tokens=max_tokens, temperature=temperature
            )
            prompt = self.renderer.build_generation_prompt(messages)
            future = self.sampling_client.sample(prompt, sampling_params=params, num_samples=1)

        result = future.result()
        output_tokens = result.sequences[0].tokens
        response_message, _ok = self.renderer.parse_response(output_tokens)
        if isinstance(response_message, dict):
            return str(response_message.get("content", ""))
        return str(response_message)


state = ModelState()


def refresh_loop():
    """Periodically check for new sampler checkpoints."""
    while True:
        time.sleep(30)
        try:
            state.refresh()
        except Exception as e:
            logger.warning("Refresh failed: %s", e)


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/chat":
            self._handle_chat()
        else:
            self._respond(404, {"error": "Not found"})

    def do_GET(self):
        if self.path == "/api/model-info":
            self._handle_model_info()
        elif self.path == "/api/training-stats":
            self._handle_training_stats()
        else:
            self._respond(404, {"error": "Not found"})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _handle_chat(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            self._respond(400, {"error": "Invalid JSON"})
            return

        messages = body.get("messages", [])
        if not messages:
            self._respond(400, {"error": "No messages"})
            return

        temperature = body.get("temperature", 0.3)
        max_tokens = body.get("max_tokens", 1024)

        try:
            content = state.sample(
                [{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._respond(200, {
                "content": content,
                "model_info": {
                    "sampler_path": state.sampler_path,
                    "step": state.step,
                },
            })
        except Exception as e:
            logger.exception("Sample failed")
            self._respond(500, {"error": str(e)})

    def _handle_model_info(self):
        # Also read latest step from metrics
        metrics_file = REPO_ROOT / "logs" / "tinker" / "stage_b_llama8b" / "metrics.jsonl"
        latest_train_step = 0
        latest_train_nll = None
        latest_val_nll = None
        if metrics_file.exists():
            for line in metrics_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                entry = json.loads(line)
                s = entry.get("step", 0)
                if ("train_nll" in entry or "train_mean_nll" in entry) and s > latest_train_step:
                    latest_train_step = s
                    latest_train_nll = entry.get("train_nll") or entry.get("train_mean_nll")
                if "validation_mean_nll" in entry:
                    latest_val_nll = entry["validation_mean_nll"]

        self._respond(200, {
            "sampler_path": state.sampler_path or "",
            "step": state.step,
            "run": "stage_b_llama8b",
            "status": "training" if state.sampling_client else "offline",
            "latest_train_step": latest_train_step,
            "latest_train_nll": latest_train_nll,
            "latest_val_nll": latest_val_nll,
        })

    def _handle_training_stats(self):
        metrics_file = REPO_ROOT / "logs" / "tinker" / "stage_b_llama8b" / "metrics.jsonl"
        stats_file = REPO_ROOT / "data" / "finetune" / "stage_b_mix" / "stats.json"
        ckpt_file = CHECKPOINTS_FILE

        metrics = []
        if metrics_file.exists():
            for line in metrics_file.read_text().strip().split("\n"):
                if line.strip():
                    metrics.append(json.loads(line))

        mix_stats = {}
        if stats_file.exists():
            mix_stats = json.loads(stats_file.read_text())

        checkpoints = []
        if ckpt_file.exists():
            for line in ckpt_file.read_text().strip().split("\n"):
                if line.strip():
                    checkpoints.append(json.loads(line))

        # Parse training log for current progress
        log_file = REPO_ROOT / "logs" / "stage_b_train.log"
        current_step = 0
        total_steps = 45561
        if log_file.exists():
            for line in reversed(log_file.read_text().split("\n")):
                if "step=" in line and "/45561" in line:
                    try:
                        part = line.split("step=")[1]
                        current_step = int(part.split("/")[0])
                        total_steps = int(part.split("/")[1].split()[0])
                    except (ValueError, IndexError):
                        pass
                    break

        self._respond(200, {
            "metrics": metrics,
            "mix_stats": mix_stats,
            "checkpoints": checkpoints,
            "current_step": current_step,
            "total_steps": total_steps,
            "progress_pct": round(100 * current_step / max(total_steps, 1), 2),
            "model_name": MODEL_NAME,
            "sampler_path": state.sampler_path or "",
            "sampler_step": state.step,
        })

    def _respond(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        logger.info("%s %s", self.client_address[0], format % args)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TVL Chat backend")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    state.init()

    # Start refresh thread
    t = threading.Thread(target=refresh_loop, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", args.port), ChatHandler)
    logger.info("Chat backend on http://localhost:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
