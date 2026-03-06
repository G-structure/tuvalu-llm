"""Translate football articles from English to Tuvaluan via Tinker API.

Picks untranslated articles from the database, translates title + body + OG
description paragraph-by-paragraph, and stores results in the translations table.

Usage:
    uv run python scripts/translate_football.py                 # translate all untranslated
    uv run python scripts/translate_football.py --limit 10      # translate up to 10
    uv run python scripts/translate_football.py --article ID    # translate a specific article
"""

import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import httpx
from tqdm import tqdm

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football" / "football.db"

# Tinker API config
TINKER_BASE = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
TINKER_MODEL = "tinker://a6453cc0-d0d8-5168-996a-c9b9ee3b8582:train:0/sampler_weights/final"

SYSTEM_PROMPT = (
    "You are a careful translator between Tuvaluan and English. Translate "
    "faithfully. Preserve names, numbers, punctuation, line breaks, and structure "
    "when possible. Output only the translation."
)

USER_PROMPT_TEMPLATE = (
    "Convert this English text to natural Tuvaluan while keeping the original "
    "structure when possible.\n\n{text}"
)

MAX_TOKENS = 512
TEMPERATURE = 0.0
STOP_SEQUENCES = ["\n\nUser:"]
REQUEST_DELAY = 0.5  # seconds between API calls


def get_api_key() -> str:
    """Load Tinker API key from environment or .env file."""
    key = os.environ.get("TINKER_API_KEY", "")
    if not key:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("TINKER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise RuntimeError("TINKER_API_KEY not found in environment or .env file")
    return key


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def split_paragraphs(body: str) -> list[str]:
    """Split article body into paragraphs for individual translation."""
    if "<p" in body:
        matches = re.findall(r"<p[^>]*>([\s\S]*?)</p>", body, re.IGNORECASE)
        if matches:
            paragraphs = [re.sub(r"<[^>]+>", "", m).strip() for m in matches]
            return [p for p in paragraphs if len(p) > 0]
    return [p.strip() for p in body.split("\n\n") if p.strip()]


def translate_text(
    client: httpx.Client, api_key: str, text: str, max_retries: int = 3
) -> str | None:
    """Translate a single piece of text via Tinker completions API."""
    prompt = (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"User: {USER_PROMPT_TEMPLATE.format(text=text)}\n\n"
        f"Assistant:"
    )

    for attempt in range(max_retries):
        try:
            resp = client.post(
                f"{TINKER_BASE}/completions",
                headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
                json={
                    "model": TINKER_MODEL,
                    "prompt": prompt,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "stop": STOP_SEQUENCES,
                },
                timeout=60,
            )

            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                continue

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return None

            translated = choices[0].get("text", "").strip()
            # Clean up any trailing stop sequences
            for stop in STOP_SEQUENCES:
                if translated.endswith(stop.strip()):
                    translated = translated[: -len(stop.strip())].strip()

            return translated if translated else None

        except (httpx.HTTPError, json.JSONDecodeError) as e:
            print(f"  Request error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

    return None


def translate_article(
    client: httpx.Client, api_key: str, article: sqlite3.Row
) -> dict | None:
    """Translate an entire article (title + body + OG description)."""
    # Translate title
    title_tvl = translate_text(client, api_key, article["title_en"])
    time.sleep(REQUEST_DELAY)

    if not title_tvl:
        print(f"  Failed to translate title, skipping article")
        return None

    # Translate OG description if present
    og_desc_tvl = None
    if article["og_description_en"]:
        og_desc_tvl = translate_text(client, api_key, article["og_description_en"])
        time.sleep(REQUEST_DELAY)

    # Translate body paragraphs
    paragraphs = split_paragraphs(article["body_en"])
    translated_paragraphs = []
    failed_count = 0

    for para in paragraphs:
        if len(para) < 5:
            translated_paragraphs.append(para)
            continue

        tvl = translate_text(client, api_key, para)
        time.sleep(REQUEST_DELAY)

        if tvl:
            translated_paragraphs.append(tvl)
        else:
            # Keep original on failure so paragraph alignment is preserved
            translated_paragraphs.append(para)
            failed_count += 1

    body_tvl = "\n\n".join(translated_paragraphs)

    return {
        "title_tvl": title_tvl,
        "body_tvl": body_tvl,
        "og_description_tvl": og_desc_tvl,
        "paragraph_count": len(paragraphs),
        "failed_paragraphs": failed_count,
    }


def get_untranslated_articles(
    conn: sqlite3.Connection, limit: int | None = None, article_id: str | None = None
) -> list[sqlite3.Row]:
    """Get articles that don't have translations yet."""
    if article_id:
        return list(
            conn.execute(
                "SELECT * FROM articles WHERE id = ?", (article_id,)
            ).fetchall()
        )

    query = """
        SELECT a.* FROM articles a
        LEFT JOIN translations t ON t.article_id = a.id
        WHERE t.id IS NULL
        ORDER BY a.published_at DESC
    """
    if limit:
        query += f" LIMIT {limit}"
    return list(conn.execute(query).fetchall())


def save_translation(conn: sqlite3.Connection, article_id: str, translation: dict):
    """Insert or replace translation in the database."""
    conn.execute(
        """INSERT OR REPLACE INTO translations
           (article_id, title_tvl, body_tvl, og_description_tvl,
            model_path, paragraph_count, failed_paragraphs)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            article_id,
            translation["title_tvl"],
            translation["body_tvl"],
            translation["og_description_tvl"],
            TINKER_MODEL,
            translation["paragraph_count"],
            translation["failed_paragraphs"],
        ),
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Translate football articles to Tuvaluan")
    parser.add_argument("--limit", type=int, help="Max articles to translate")
    parser.add_argument("--article", type=str, help="Translate a specific article ID")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}. Run init_football_db.py first.")
        return

    api_key = get_api_key()
    conn = get_db()

    articles = get_untranslated_articles(conn, args.limit, args.article)
    if not articles:
        print("No untranslated articles found.")
        return

    print(f"Found {len(articles)} articles to translate")

    client = httpx.Client(http2=True)
    translated = 0
    failed = 0

    for article in tqdm(articles, desc="Translating"):
        title_preview = article["title_en"][:60]
        tqdm.write(f"  Translating: {title_preview}...")

        translation = translate_article(client, api_key, article)

        if translation:
            save_translation(conn, article["id"], translation)
            translated += 1
            if translation["failed_paragraphs"] > 0:
                tqdm.write(
                    f"  Warning: {translation['failed_paragraphs']}/{translation['paragraph_count']} paragraphs failed"
                )
        else:
            failed += 1
            tqdm.write(f"  Failed to translate article")

    client.close()
    conn.close()

    print(f"\nDone! {translated} translated, {failed} failed")


if __name__ == "__main__":
    main()
