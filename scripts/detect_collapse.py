"""Detect model collapse (degenerate repetition) in translated text.

A translation is considered "collapsed" if it exhibits excessive n-gram
repetition, indicating the model got stuck in a loop.

Usage as module:
    from detect_collapse import is_collapsed, collapse_score

Usage as CLI (retroactive scan):
    uv run python scripts/detect_collapse.py                    # scan + flag
    uv run python scripts/detect_collapse.py --dry-run          # preview only
    uv run python scripts/detect_collapse.py --threshold 0.3    # custom threshold
"""

import argparse
import re
import sqlite3
from collections import Counter
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football" / "football.db"

# Default threshold: if unique 4-gram ratio drops below this, it's collapsed
DEFAULT_NGRAM_SIZE = 4
DEFAULT_THRESHOLD = 0.3  # 30% unique 4-grams = highly repetitive
MIN_WORDS_FOR_DETECTION = 20  # skip very short texts


def ngram_repetition_score(text: str, n: int = DEFAULT_NGRAM_SIZE) -> float:
    """Compute the ratio of unique n-grams to total n-grams.

    Returns a value between 0.0 and 1.0:
    - 1.0 = all n-grams are unique (no repetition)
    - 0.0 = single n-gram repeated (maximum repetition)
    """
    words = re.findall(r'\w+', text.lower())
    if len(words) < n + 1:
        return 1.0  # too short to judge

    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 1.0

    unique = len(set(ngrams))
    return unique / len(ngrams)


def max_ngram_frequency(text: str, n: int = DEFAULT_NGRAM_SIZE) -> tuple[float, str]:
    """Find the most repeated n-gram and its frequency ratio.

    Returns (frequency_ratio, ngram_text):
    - frequency_ratio: fraction of all n-grams that are the most common one
    - ngram_text: the actual repeated phrase
    """
    words = re.findall(r'\w+', text.lower())
    if len(words) < n + 1:
        return 0.0, ""

    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0, ""

    counts = Counter(ngrams)
    most_common, count = counts.most_common(1)[0]
    return count / len(ngrams), " ".join(most_common)


def collapse_score(text: str) -> float:
    """Score how collapsed/repetitive a text is.

    Returns 0.0 (no collapse) to 1.0 (fully collapsed).
    Combines whole-text, tail, and per-paragraph signals.
    """
    words = re.findall(r'\w+', text.lower())
    if len(words) < MIN_WORDS_FOR_DETECTION:
        return 0.0

    # Primary signal: unique n-gram ratio (inverted so higher = worse)
    uniqueness = ngram_repetition_score(text)
    repetition = 1.0 - uniqueness

    # Secondary signal: max single n-gram dominance
    max_freq, _ = max_ngram_frequency(text)

    # Tail signal
    tail_words = words[-50:] if len(words) >= 50 else words
    tail_text = " ".join(tail_words)
    tail_uniqueness = ngram_repetition_score(tail_text)
    tail_rep = 1.0 - tail_uniqueness

    # Combine: take max of whole-text and tail signals
    whole_score = 0.7 * repetition + 0.3 * max_freq
    tail_score = tail_rep * 0.8  # tail collapse alone is strong signal

    return min(1.0, max(whole_score, tail_score))


def tail_collapse_detected(text: str, window_words: int = 80) -> bool:
    """Detect collapse that starts partway through the text (tail repetition).

    Checks the last `window_words` words for extreme repetition using both
    3-grams and 4-grams. Catches cases where the model starts coherent but
    degenerates at the end.
    """
    words = re.findall(r'\w+', text.lower())
    if len(words) < window_words:
        return False

    tail = words[-window_words:]

    # Check with 3-grams (catches shorter repeating phrases like "i te taimi")
    for n in [3, 4]:
        ngrams = [tuple(tail[i:i + n]) for i in range(len(tail) - n + 1)]
        if len(ngrams) < 5:
            continue

        unique = len(set(ngrams))
        ratio = unique / len(ngrams)
        if ratio < 0.25:  # tail has < 25% unique n-grams
            return True

        # Also check if any single n-gram dominates the tail
        counts = Counter(ngrams)
        _, top_count = counts.most_common(1)[0]
        if top_count / len(ngrams) > 0.10:  # one phrase is >10% of tail
            return True

    return False


def per_paragraph_collapse(text: str, n: int = DEFAULT_NGRAM_SIZE) -> bool:
    """Check each paragraph individually for collapse.

    Catches cases where one paragraph is fully collapsed even if the overall
    body metrics look fine because other paragraphs dilute the signal.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in paragraphs:
        words = re.findall(r'\w+', para.lower())
        if len(words) < 30:
            continue
        uniqueness = ngram_repetition_score(para, n)
        if uniqueness < 0.15:
            return True
        max_freq, _ = max_ngram_frequency(para, n)
        if max_freq > 0.5:
            return True
    return False


def is_collapsed(text: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Check if text shows model collapse (degenerate repetition).

    A text is collapsed if any of these triggers:
    1. The whole-text unique 4-gram ratio is below the threshold
    2. Any single 4-gram accounts for >40% of all 4-grams
    3. The tail (last 50 words) shows extreme repetition
    4. Any individual paragraph is heavily collapsed
    """
    words = re.findall(r'\w+', text.lower())
    if len(words) < MIN_WORDS_FOR_DETECTION:
        return False

    # Whole-text checks
    uniqueness = ngram_repetition_score(text)
    if uniqueness < threshold:
        return True

    max_freq, _ = max_ngram_frequency(text)
    if max_freq > 0.4:
        return True

    # Tail check — catches degeneration that starts mid-text
    if tail_collapse_detected(text):
        return True

    # Per-paragraph check — catches one bad paragraph among good ones
    if per_paragraph_collapse(text):
        return True

    return False


def scan_and_flag(dry_run: bool = False, threshold: float = DEFAULT_THRESHOLD):
    """Scan all translations and flag collapsed ones."""
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, article_id, title_tvl, body_tvl FROM translations"
    ).fetchall()

    print(f"Scanning {len(rows)} translations for model collapse...")

    collapsed_count = 0
    results = []

    for row in rows:
        body = row["body_tvl"] or ""
        title = row["title_tvl"] or ""

        # Check both body and title
        body_score = collapse_score(body)
        title_score = collapse_score(title)
        combined_score = max(body_score, title_score)

        body_collapsed = is_collapsed(body, threshold)
        title_collapsed = is_collapsed(title, threshold)
        any_collapsed = body_collapsed or title_collapsed

        if any_collapsed:
            collapsed_count += 1
            # Preview the repeated phrase
            _, repeated = max_ngram_frequency(body if body_collapsed else title)
            preview = (title or "")[:60]
            results.append({
                "id": row["id"],
                "article_id": row["article_id"],
                "score": combined_score,
                "repeated": repeated,
                "preview": preview,
            })
            print(f"  COLLAPSED [{combined_score:.2f}] {preview}...")
            print(f"    Repeats: \"{repeated}\"")

        if not dry_run:
            conn.execute(
                "UPDATE translations SET is_collapsed = ?, collapse_score = ? WHERE id = ?",
                (1 if any_collapsed else 0, round(combined_score, 4), row["id"]),
            )

    if not dry_run:
        conn.commit()
        print(f"\nFlagged {collapsed_count}/{len(rows)} translations as collapsed")
    else:
        print(f"\n[DRY RUN] Would flag {collapsed_count}/{len(rows)} translations as collapsed")

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Detect model collapse in translations")
    parser.add_argument("--dry-run", action="store_true", help="Preview without updating DB")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Unique n-gram ratio threshold (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    scan_and_flag(dry_run=args.dry_run, threshold=args.threshold)


if __name__ == "__main__":
    main()
