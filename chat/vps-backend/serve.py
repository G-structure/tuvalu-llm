#!/usr/bin/env python3
"""Tinker chat backend for VPS deployment.

Connects to Tinker's sampling API and serves a simple chat endpoint.
Model weights live on Tinker infrastructure — this just does prompt
formatting and API calls.

Usage:
    uv run python serve.py
    SAMPLER_PATH=tinker://... uv run python serve.py --port 8787
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("tvl-chat")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
DEFAULT_SAMPLER_PATH = os.environ.get(
    "SAMPLER_PATH",
    "tinker://cd962896-a3c4-5b9f-94d7-b89e2a0cba17:train:0/sampler_weights/gen_eval_003000",
)


def init_tinker():
    """Initialize Tinker SDK components."""
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        sys.exit("TINKER_API_KEY must be set")

    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Renderer: %s for %s", renderer_name, MODEL_NAME)

    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(model_path=DEFAULT_SAMPLER_PATH)
    logger.info("Sampling client ready: %s", DEFAULT_SAMPLER_PATH)

    return renderer, sampling_client


class ChatState:
    def __init__(self):
        self.renderer = None
        self.sampling_client = None
        self.lock = threading.Lock()

    def init(self):
        self.renderer, self.sampling_client = init_tinker()

    def sample(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
        import tinker

        with self.lock:
            params = tinker.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=self.renderer.get_stop_sequences(),
            )
            prompt = self.renderer.build_generation_prompt(messages)
            future = self.sampling_client.sample(prompt, sampling_params=params, num_samples=1)

        result = future.result()
        output_tokens = result.sequences[0].tokens
        response_message, _ok = self.renderer.parse_response(output_tokens)
        if isinstance(response_message, dict):
            return str(response_message.get("content", ""))
        return str(response_message)


state = ChatState()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/chat":
            self._handle_chat()
        else:
            self._respond(404, {"error": "Not found"})

    def do_GET(self):
        if self.path == "/api/health":
            self._respond(200, {"status": "ok", "model": MODEL_NAME, "sampler": DEFAULT_SAMPLER_PATH})
        else:
            self._respond(404, {"error": "Not found"})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
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

        try:
            content = state.sample(
                [{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=body.get("temperature", 0.3),
                max_tokens=body.get("max_tokens", 1024),
            )
            self._respond(200, {
                "content": content,
                "model_info": {"sampler_path": DEFAULT_SAMPLER_PATH},
            })
        except Exception as e:
            logger.exception("Sample failed")
            self._respond(500, {"error": str(e)})

    def _respond(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt, *args):
        logger.info("%s %s", self.client_address[0], fmt % args)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    state.init()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    logger.info("TVL Chat backend on http://127.0.0.1:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
