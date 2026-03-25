#!/usr/bin/env python3
"""TVL Bridge backend — chat with any LLM in Tuvaluan.

Translates Tuvaluan → English via Tinker, queries an LLM via OpenRouter,
then translates the English response back to Tuvaluan via Tinker.

Usage:
    SAMPLER_PATH=tinker://... OPENROUTER_API_KEY=sk-... uv run python serve.py
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import threading
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("tvl-bridge")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
DEFAULT_SAMPLER_PATH = os.environ.get("SAMPLER_PATH")
if not DEFAULT_SAMPLER_PATH:
    sys.exit("SAMPLER_PATH must be set in the environment")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    sys.exit("OPENROUTER_API_KEY must be set in the environment")

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4.1-nano")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_DIR = pathlib.Path(os.environ.get("PROMPT_DIR", "/app/prompts"))
SYSTEM_PROMPT_FILE = PROMPT_DIR / "system.txt"


def load_system_prompt() -> str:
    """Load system prompt from file, re-read each call so edits take effect."""
    try:
        return SYSTEM_PROMPT_FILE.read_text().strip()
    except FileNotFoundError:
        return "You are a helpful assistant."


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


class TinkerTranslator:
    """Translates between Tuvaluan and English using the Tinker model."""

    def __init__(self):
        self.renderer = None
        self.sampling_client = None
        self.lock = threading.Lock()

    def init(self):
        self.renderer, self.sampling_client = init_tinker()

    def _sample(self, messages: list[dict], max_tokens: int = 1024) -> str:
        import tinker

        with self.lock:
            params = tinker.SamplingParams(
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.95,
                stop=self.renderer.get_stop_sequences(),
            )
            prompt = self.renderer.build_generation_prompt(messages)
            future = self.sampling_client.sample(
                prompt, sampling_params=params, num_samples=1
            )

        result = future.result()
        output_tokens = result.sequences[0].tokens
        response_message, _ok = self.renderer.parse_response(output_tokens)
        if isinstance(response_message, dict):
            return str(response_message.get("content", ""))
        return str(response_message)

    def tvl_to_en(self, text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    "Translate this Tuvaluan text to English. "
                    "Do NOT answer the question or write code — ONLY translate the words to English.\n\n"
                    f"Tuvaluan: {text}\n"
                    "English:"
                ),
            },
        ]
        return self._sample(messages, max_tokens=2048)

    def en_to_tvl(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an English-to-Tuvaluan translator. "
                    "Translate the following English text to Tuvaluan accurately. "
                    "Keep code blocks, technical terms, and programming syntax exactly as-is — do not translate them. "
                    "Output only the Tuvaluan translation, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ]
        return self._sample(messages, max_tokens=2048)


def query_openrouter(messages: list[dict]) -> str:
    """Send messages to OpenRouter and return the assistant response."""
    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }).encode()

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/g-structure/tv",
        },
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    return data["choices"][0]["message"]["content"]


translator = TinkerTranslator()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/bridge":
            self._handle_bridge()
        else:
            self._respond(404, {"error": "Not found"})

    def do_GET(self):
        if self.path == "/api/health":
            self._respond(200, {
                "status": "ok",
                "service": "tvl-bridge",
                "model": MODEL_NAME,
                "sampler": DEFAULT_SAMPLER_PATH,
                "openrouter_model": OPENROUTER_MODEL,
            })
        else:
            self._respond(404, {"error": "Not found"})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _handle_bridge(self):
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
            # Step 1: Translate each user message TVL → EN
            en_messages = []
            system_prompt = load_system_prompt()
            en_messages.append({"role": "system", "content": system_prompt})

            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    en_content = translator.tvl_to_en(content)
                    logger.info("TVL→EN: %s → %s", content[:80], en_content[:80])
                    en_messages.append({"role": "user", "content": en_content})
                elif role == "assistant":
                    # Prior assistant messages — pass through as English
                    # (the client should send the English intermediates if maintaining history)
                    en_messages.append({"role": "assistant", "content": content})

            # Step 2: Query OpenRouter
            en_response = query_openrouter(en_messages)
            logger.info("LLM response: %s", en_response[:120])

            # Step 3: Translate EN → TVL
            tvl_response = translator.en_to_tvl(en_response)
            logger.info("EN→TVL: %s", tvl_response[:120])

            self._respond(200, {
                "content": tvl_response,
                "english": en_response,
                "model_info": {
                    "sampler_path": DEFAULT_SAMPLER_PATH,
                    "openrouter_model": OPENROUTER_MODEL,
                },
            })
        except Exception as e:
            logger.exception("Bridge request failed")
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
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()

    translator.init()

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    logger.info("TVL Bridge on http://0.0.0.0:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
