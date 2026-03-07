"""Clean scraped article bodies: remove promo blocks, ads, normalize whitespace.

Runs as a one-time migration on existing DB articles, and can be imported
as a module by scrapers for cleaning at ingest time.

Usage:
    uv run python scripts/clean_article_bodies.py              # clean all articles
    uv run python scripts/clean_article_bodies.py --dry-run    # preview changes
    uv run python scripts/clean_article_bodies.py --source sky # clean only Sky Sports
"""

import argparse
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football" / "football.db"


# ---------------------------------------------------------------------------
# Sky Sports cleaning
# ---------------------------------------------------------------------------

_SKY_GOT_SKY = re.compile(
    r"Got Sky\? Watch[^\n]*?[\U0001F4F1\U0001F4FA]",
    re.DOTALL,
)
_SKY_NOT_GOT = re.compile(
    r"(?:Not got Sky|No Sky)\? (?:Get Sky Sports|Get instant access|Stream your EFL team)[^\n]*?[\U0001F4FA]?",
    re.DOTALL,
)
_SKY_PUSH_NOTIF = re.compile(
    r"Choose the Sky Sports push notifications you want!?\s*\U0001F514?",
)
_SKY_DOWNLOAD_APP = re.compile(
    r"Download the Sky Sports app[^.]*",
)
_SKY_FREE_HIGHLIGHTS = re.compile(
    r"(?:Watch )?FREE [A-Za-z\s]+ (?:PL |Premier League )?highlights\u25B6\uFE0F?",
)
_SKY_NEWS_TRANSFERS = re.compile(
    r"[A-Z][A-Za-z\s]+ news & transfers\u26AA?\s*\|?\s*[A-Z][A-Za-z\s]+ fixtures & scores",
)
_SKY_FIXTURES_SCORES = re.compile(
    r"[A-Z][A-Za-z\s]{2,30} fixtures & scores(?:\s*\|[^|.]*)?",
)
_SKY_LIVE_TABLE = re.compile(
    r"Live Premier League table\s*\|[^.]*",
)
_SKY_WHATSAPP = re.compile(
    r"Get more EFL to your phone with WhatsApp",
)


def clean_sky_body(text: str) -> str:
    """Remove Sky Sports promo blocks, inline link artifacts, and emoji."""
    text = _SKY_GOT_SKY.sub("", text)
    text = _SKY_NOT_GOT.sub("", text)
    text = _SKY_PUSH_NOTIF.sub("", text)
    text = _SKY_DOWNLOAD_APP.sub("", text)
    text = _SKY_FREE_HIGHLIGHTS.sub("", text)
    text = _SKY_NEWS_TRANSFERS.sub("", text)
    text = _SKY_LIVE_TABLE.sub("", text)
    text = _SKY_WHATSAPP.sub("", text)
    text = _SKY_FIXTURES_SCORES.sub("", text)
    # Remove promo emoji
    text = re.sub(r"[\U0001F4F1\U0001F4FA\U0001F514\u25B6\uFE0F\u26AA]", "", text)
    return text


# ---------------------------------------------------------------------------
# Goal.com cleaning
# ---------------------------------------------------------------------------

_GOAL_READ_MORE = re.compile(
    r"^(?:READ )?MORE:[^\n]*$",
    re.MULTILINE,
)
_GOAL_NORDVPN = re.compile(
    r"^Stream .+? anywhere with NordVPN\s*Sign up now$",
    re.MULTILINE,
)
_GOAL_SUBSCRIBE = re.compile(
    r"^Watch .+? with a .+? subscription\s*Sign up (?:today|now)$",
    re.MULTILINE,
)
_GOAL_VPN_BOILERPLATE = re.compile(
    r"If you are out of the country and would like to watch .+?(?:Virtual Private Network \(VPN\)|using a VPN)\.?",
    re.DOTALL,
)
_GOAL_BEST_DEALS = re.compile(
    r"^Find the best deals$",
    re.MULTILINE,
)
_GOAL_AD = re.compile(
    r"^Ad \|[^\n]*$",
    re.MULTILINE,
)
_GOAL_HASHTAGS = re.compile(
    r"(?:#\w+){2,}",
)


def clean_goal_body(text: str) -> str:
    """Remove Goal.com promo links, NordVPN/subscribe blocks, and ad markers."""
    text = _GOAL_READ_MORE.sub("", text)
    text = _GOAL_NORDVPN.sub("", text)
    text = _GOAL_SUBSCRIBE.sub("", text)
    text = _GOAL_VPN_BOILERPLATE.sub("", text)
    text = _GOAL_BEST_DEALS.sub("", text)
    text = _GOAL_AD.sub("", text)
    text = _GOAL_HASHTAGS.sub("", text)
    return text


# ---------------------------------------------------------------------------
# FIFA.com cleaning
# ---------------------------------------------------------------------------

def clean_fifa_body(text: str) -> str:
    """Normalize FIFA.com article bodies (NBSP, BOM, narrow spaces)."""
    # Replace NBSP, narrow NBSP, and other Unicode spaces with regular space
    text = re.sub(r"[\u00a0\u202f\u2060]", " ", text)
    # Remove BOM / zero-width chars
    text = re.sub(r"[\ufeff\u200b\u200c\u200d]", "", text)
    return text


# ---------------------------------------------------------------------------
# Common normalization + paragraph splitting
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Normalize double spaces, trailing whitespace, and excessive blank lines."""
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^ +", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_SOURCE_CLEANERS = {
    "sky": clean_sky_body,
    "goal": clean_goal_body,
    "fifa": clean_fifa_body,
}


def clean_body(text: str, source_id: str) -> str:
    """Apply source-specific cleaning + common normalization."""
    cleaner = _SOURCE_CLEANERS.get(source_id)
    if cleaner:
        text = cleaner(text)
    text = _normalize_whitespace(text)
    return text


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z][a-z])")


def split_into_paragraphs(text: str, min_words_for_split: int = 100) -> str:
    """Split a single text blob into paragraphs at sentence boundaries.

    If the text already has double-newline paragraph breaks, returns as-is.
    For long blocks with no breaks, groups sentences into ~100-word paragraphs.
    """
    if "\n\n" in text:
        return text

    word_count = len(text.split())
    if word_count <= min_words_for_split:
        return text

    sentences = _SENTENCE_BOUNDARY.split(text)
    if len(sentences) <= 1:
        return text

    target_words = 100
    paragraphs = []
    current = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        current.append(sentence)
        current_words += s_words

        if current_words >= target_words:
            paragraphs.append(" ".join(current))
            current = []
            current_words = 0

    if current:
        if paragraphs and current_words < target_words // 3:
            paragraphs[-1] += " " + " ".join(current)
        else:
            paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# CLI: one-time migration
# ---------------------------------------------------------------------------

def migrate(source_filter: str | None = None, dry_run: bool = False):
    """Clean all article bodies in the database."""
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    query = "SELECT id, source_id, body_en, title_en FROM articles"
    params: tuple = ()
    if source_filter:
        query += " WHERE source_id = ?"
        params = (source_filter,)

    rows = conn.execute(query, params).fetchall()

    changed = 0
    unchanged = 0
    total_chars_removed = 0

    for article_id, source_id, body_en, title_en in rows:
        cleaned = clean_body(body_en, source_id)

        if source_id == "sky":
            cleaned = split_into_paragraphs(cleaned)

        # Also clean title whitespace
        clean_title = title_en.strip() if title_en else title_en

        if cleaned != body_en or clean_title != title_en:
            chars_diff = len(body_en) - len(cleaned)
            total_chars_removed += max(0, chars_diff)
            changed += 1

            if dry_run:
                print(f"  [{source_id}] {article_id} ({chars_diff:+d} chars)")
                if chars_diff > 0:
                    print(f"    Before: {body_en[:150]}...")
                    print(f"    After:  {cleaned[:150]}...")
            else:
                new_word_count = len(cleaned.split())
                conn.execute(
                    "UPDATE articles SET body_en = ?, title_en = ?, word_count = ? WHERE id = ?",
                    (cleaned, clean_title, new_word_count, article_id),
                )
        else:
            unchanged += 1

    if not dry_run:
        conn.commit()

    conn.close()

    print(f"\nResults:")
    print(f"  Total articles: {len(rows)}")
    print(f"  Changed: {changed}")
    print(f"  Unchanged: {unchanged}")
    print(f"  Total chars removed: {total_chars_removed:,}")
    if dry_run:
        print("  [DRY RUN — no changes written]")


def main():
    parser = argparse.ArgumentParser(
        description="Clean article bodies: remove promo blocks, normalize whitespace"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying DB")
    parser.add_argument("--source", choices=["sky", "goal", "fifa"], help="Only clean this source")
    args = parser.parse_args()

    migrate(source_filter=args.source, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
