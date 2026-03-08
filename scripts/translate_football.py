"""Translate football articles from English to Tuvaluan via Tinker API.

Picks untranslated articles from the database, translates title + body + OG
description paragraph-by-paragraph, and stores results in the translations table.

Features:
- 3-attempt retry with escalating temperature (0.0 → 0.3 → 0.7) on collapse
- Model collapse detection (n-gram repetition ratio)
- All attempts recorded in translation_attempts table for RL training
- Model ID tracking for each translation

Usage:
    uv run python scripts/translate_football.py                 # translate all untranslated
    uv run python scripts/translate_football.py --limit 10      # translate up to 10
    uv run python scripts/translate_football.py --article ID    # translate a specific article
    uv run python scripts/translate_football.py --retry-collapsed  # retry previously collapsed
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

from detect_collapse import is_collapsed, collapse_score

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football" / "football.db"

# Tinker API config
TINKER_BASE = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
TINKER_MODEL = "tinker://a6453cc0-d0d8-5168-996a-c9b9ee3b8582:train:0/sampler_weights/final"
TINKER_MODEL_ID = "a6453cc0-d0d8-5168-996a-c9b9ee3b8582"

SYSTEM_PROMPT = (
    "You are a careful translator between Tuvaluan and English. Translate "
    "faithfully. Preserve names, numbers, punctuation, line breaks, and structure "
    "when possible. Output only the translation.\n"
    "Keep proper nouns (person names, place names, team names) exactly as they "
    "appear in English — do not transliterate them. For football/sports terms "
    "with no Tuvaluan equivalent (penalty, offside, midfielder, VAR, Premier "
    "League, Champions League, etc.), keep the English term as a loanword."
)

USER_PROMPT_TEMPLATE = (
    "Convert this English text to natural Tuvaluan while keeping the original "
    "structure when possible.\n\n{text}"
)

MAX_TOKENS = 1024
STOP_SEQUENCES = ["\n\nUser:", "\nUser:", "User:"]
REQUEST_DELAY = 0.5  # seconds between API calls
MAX_PARAGRAPH_WORDS = 150  # sub-chunk paragraphs longer than this

# Temperature escalation for retry on collapse
TEMPERATURE_SCHEDULE = [0.0, 0.3, 0.7]


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


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_paragraphs(body: str) -> list[str]:
    """Split article body into paragraphs for individual translation."""
    if "<p" in body:
        matches = re.findall(r"<p[^>]*>([\s\S]*?)</p>", body, re.IGNORECASE)
        if matches:
            paragraphs = [re.sub(r"<[^>]+>", "", m).strip() for m in matches]
            return [p for p in paragraphs if len(p) > 0]
    return [p.strip() for p in body.split("\n\n") if p.strip()]


def sub_chunk_paragraph(text: str, max_words: int = MAX_PARAGRAPH_WORDS) -> list[str]:
    """Split a long paragraph into sentence-boundary chunks.

    If the paragraph is short enough, returns it as a single-element list.
    For long paragraphs, groups sentences into chunks of ~max_words.
    """
    word_count = len(text.split())
    if word_count <= max_words:
        return [text]

    sentences = _SENTENCE_BOUNDARY.split(text)
    if len(sentences) <= 1:
        return [text]  # can't split further

    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        current.append(sentence)
        current_words += s_words

        if current_words >= max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

    if current:
        # Merge short tail into last chunk to avoid tiny fragments
        if chunks and current_words < max_words // 3:
            chunks[-1] += " " + " ".join(current)
        else:
            chunks.append(" ".join(current))

    return chunks


def translate_text(
    client: httpx.Client, api_key: str, text: str,
    temperature: float = 0.0, max_retries: int = 3
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
                    "temperature": temperature,
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
            # Clean up any trailing stop sequences and repeated prompts
            for stop in STOP_SEQUENCES:
                stop_clean = stop.strip()
                if translated.endswith(stop_clean):
                    translated = translated[: -len(stop_clean)].strip()
            # Also strip any "User:" that appears mid-text (model looping)
            if "\nUser:" in translated:
                translated = translated.split("\nUser:")[0].strip()

            return translated if translated else None

        except (httpx.HTTPError, json.JSONDecodeError) as e:
            print(f"  Request error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

    return None


def translate_article(
    client: httpx.Client, api_key: str, article: sqlite3.Row,
    temperature: float = 0.0,
) -> dict | None:
    """Translate an entire article (title + body + OG description)."""
    # Translate title
    title_tvl = translate_text(client, api_key, article["title_en"], temperature)
    time.sleep(REQUEST_DELAY)

    if not title_tvl:
        print(f"  Failed to translate title, skipping article")
        return None

    # Translate OG description if present
    og_desc_tvl = None
    if article["og_description_en"]:
        og_desc_tvl = translate_text(client, api_key, article["og_description_en"], temperature)
        time.sleep(REQUEST_DELAY)

    # Translate body paragraphs (with sub-chunking for long ones)
    paragraphs = split_paragraphs(article["body_en"])
    translated_paragraphs = []
    failed_count = 0

    for para in paragraphs:
        if len(para) < 5:
            translated_paragraphs.append(para)
            continue

        # Sub-chunk long paragraphs at sentence boundaries
        chunks = sub_chunk_paragraph(para)
        translated_chunks = []
        chunk_failed = False

        for chunk in chunks:
            tvl = translate_text(client, api_key, chunk, temperature)
            time.sleep(REQUEST_DELAY)

            if tvl:
                translated_chunks.append(tvl)
            else:
                translated_chunks.append(chunk)
                chunk_failed = True

        translated_paragraphs.append(" ".join(translated_chunks))
        if chunk_failed:
            failed_count += 1

    body_tvl = "\n\n".join(translated_paragraphs)

    # Compute collapse metrics
    body_collapsed = is_collapsed(body_tvl)
    title_collapsed = is_collapsed(title_tvl)
    score = max(collapse_score(body_tvl), collapse_score(title_tvl))

    return {
        "title_tvl": title_tvl,
        "body_tvl": body_tvl,
        "og_description_tvl": og_desc_tvl,
        "paragraph_count": len(paragraphs),
        "failed_paragraphs": failed_count,
        "is_collapsed": body_collapsed or title_collapsed,
        "collapse_score": score,
        "temperature": temperature,
    }


def save_attempt(conn: sqlite3.Connection, article_id: str, translation: dict, attempt_num: int):
    """Record a translation attempt (for RL training data)."""
    conn.execute(
        """INSERT INTO translation_attempts
           (article_id, attempt_number, title_tvl, body_tvl, og_description_tvl,
            model_path, model_id, temperature, is_collapsed, collapse_score,
            paragraph_count, failed_paragraphs)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            article_id,
            attempt_num,
            translation["title_tvl"],
            translation["body_tvl"],
            translation["og_description_tvl"],
            TINKER_MODEL,
            TINKER_MODEL_ID,
            translation["temperature"],
            1 if translation["is_collapsed"] else 0,
            round(translation["collapse_score"], 4),
            translation["paragraph_count"],
            translation["failed_paragraphs"],
        ),
    )
    conn.commit()


def save_translation(conn: sqlite3.Connection, article_id: str, translation: dict, attempt_num: int):
    """Insert or replace translation in the database."""
    conn.execute(
        """INSERT OR REPLACE INTO translations
           (article_id, title_tvl, body_tvl, og_description_tvl,
            model_path, model_id, paragraph_count, failed_paragraphs,
            is_collapsed, collapse_score, attempt_number)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            article_id,
            translation["title_tvl"],
            translation["body_tvl"],
            translation["og_description_tvl"],
            TINKER_MODEL,
            TINKER_MODEL_ID,
            translation["paragraph_count"],
            translation["failed_paragraphs"],
            1 if translation["is_collapsed"] else 0,
            round(translation["collapse_score"], 4),
            attempt_num,
        ),
    )
    conn.commit()


def translate_with_retry(
    client: httpx.Client, api_key: str, conn: sqlite3.Connection,
    article: sqlite3.Row,
) -> bool:
    """Translate an article with up to 3 attempts using escalating temperature.

    All attempts are recorded. The best non-collapsed attempt (or last attempt
    if all collapse) is saved as the active translation.
    """
    best_translation = None
    best_attempt = 0

    for attempt_idx, temperature in enumerate(TEMPERATURE_SCHEDULE):
        attempt_num = attempt_idx + 1
        tqdm.write(f"  Attempt {attempt_num}/3 (temp={temperature})...")

        translation = translate_article(client, api_key, article, temperature)
        if translation is None:
            tqdm.write(f"  Attempt {attempt_num} failed completely")
            continue

        # Record every attempt for RL training
        save_attempt(conn, article["id"], translation, attempt_num)

        if translation["is_collapsed"]:
            score = translation["collapse_score"]
            tqdm.write(f"  Attempt {attempt_num} COLLAPSED (score={score:.2f})")
            # Keep as fallback if nothing better comes
            if best_translation is None or best_translation["is_collapsed"]:
                best_translation = translation
                best_attempt = attempt_num
        else:
            tqdm.write(f"  Attempt {attempt_num} OK (score={translation['collapse_score']:.2f})")
            best_translation = translation
            best_attempt = attempt_num
            break  # good translation, stop retrying

    if best_translation:
        save_translation(conn, article["id"], best_translation, best_attempt)
        if best_translation["is_collapsed"]:
            tqdm.write(f"  WARNING: All attempts collapsed, saving best (flagged)")
        return True
    return False


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


def get_collapsed_articles(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Get articles whose current translation is flagged as collapsed."""
    return list(conn.execute(
        """SELECT a.* FROM articles a
           JOIN translations t ON t.article_id = a.id
           WHERE t.is_collapsed = 1
           ORDER BY a.published_at DESC"""
    ).fetchall())


def main():
    parser = argparse.ArgumentParser(description="Translate football articles to Tuvaluan")
    parser.add_argument("--limit", type=int, help="Max articles to translate")
    parser.add_argument("--article", type=str, help="Translate a specific article ID")
    parser.add_argument("--retry-collapsed", action="store_true",
                        help="Retry articles with collapsed translations")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}. Run init_football_db.py first.")
        return

    api_key = get_api_key()
    conn = get_db()

    if args.retry_collapsed:
        articles = get_collapsed_articles(conn)
        if not articles:
            print("No collapsed translations found.")
            return
        print(f"Found {len(articles)} collapsed translations to retry")
    else:
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

        if translate_with_retry(client, api_key, conn, article):
            translated += 1
        else:
            failed += 1
            tqdm.write(f"  Failed to translate article")

    client.close()
    conn.close()

    print(f"\nDone! {translated} translated, {failed} failed")


if __name__ == "__main__":
    main()
