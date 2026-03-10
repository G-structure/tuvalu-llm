"""Scrape tuvalu.aa-ken.jp Tuvaluan learning app — words + expressions.

The site is a Svelte SPA that loads JSON data from ./data/ endpoints.
Two data types:
  - expressions.json: 23 categories of Tuvaluan/English/Japanese phrase sets
  - {subcategory}.json: 42 word lists with Tuvaluan/English pairs

Uses Docker curl-impersonate for fetching (browser-like TLS fingerprint).

Usage:
    uv run python scripts/scrape_tuvalu_app.py
    uv run python scripts/scrape_tuvalu_app.py --words-only
    uv run python scripts/scrape_tuvalu_app.py --expressions-only
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

# Add scripts dir to path for fetch module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import fetch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "tuvalu_app"
ALIGNED_DIR = DATA_DIR / "aligned"

BASE_URL = "https://tuvalu.aa-ken.jp/webapp/data"

# Categories and subcategories extracted from the app's bundle.js
WORD_CATEGORIES = [
    {"category": "Human", "subCategory": [
        "Family", "Relationships", "Body Parts", "Feelings",
        "Character & Appearance", "Health", "Occupation",
        "Sports & Amusements", "Life & Identity"]},
    {"category": "House", "subCategory": ["House", "Furniture & Necessities"]},
    {"category": "Food", "subCategory": ["Food"]},
    {"category": "Clothes", "subCategory": ["Clothes"]},
    {"category": "Number", "subCategory": ["Numbers"]},
    {"category": "Time & Dates", "subCategory": ["Times", "A Week & Months"]},
    {"category": "Colors", "subCategory": ["Colors"]},
    {"category": "Nature", "subCategory": ["Climates", "Land, Sea, Sky"]},
    {"category": "Education", "subCategory": ["Education"]},
    {"category": "Politics & Religion", "subCategory": ["Politics & Religion"]},
    {"category": "Countries & Places", "subCategory": [
        "Countries & Islands", "Town & Facilities"]},
    {"category": "Animals", "subCategory": ["Animals"]},
    {"category": "Plants", "subCategory": ["Plants"]},
    {"category": "Basic Verbs", "subCategory": [
        "A-E", "F", "faka-", "G-K", "L-N", "O-P", "S", "T", "U-V"]},
    {"category": "Adjectives & Adverbs", "subCategory": [
        "A-F", "G-L", "M", "N-S", "T-V"]},
    {"category": "Directions", "subCategory": ["Directions"]},
    {"category": "Useful Expressions", "subCategory": ["Useful Expressions"]},
    {"category": "Interrogatives", "subCategory": ["Interrogatives"]},
]


def subcategory_to_filename(name: str) -> str:
    """Convert subcategory name to JSON filename (matches app's Ve function)."""
    return name.replace("&", "and").lower().replace(" ", "_") + ".json"


def fetch_json(url: str) -> dict | list | None:
    """Fetch a URL and parse as JSON."""
    html = fetch(url)
    if html is None:
        return None
    try:
        return json.loads(html)
    except json.JSONDecodeError as e:
        print(f"  JSON decode error for {url}: {e}", file=sys.stderr)
        return None


def scrape_expressions() -> list[dict]:
    """Scrape expressions.json — trilingual phrase sets."""
    url = f"{BASE_URL}/expressions.json"
    print(f"Fetching expressions from {url}")

    # Cache raw JSON
    raw_path = RAW_DIR / "expressions.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and raw_path.stat().st_size > 0:
        data = json.loads(raw_path.read_text())
        print(f"  Loaded from cache: {len(data)} categories")
    else:
        data = fetch_json(url)
        if data is None:
            print("  FAILED to fetch expressions.json", file=sys.stderr)
            return []
        raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"  Fetched {len(data)} expression categories")

    records = []
    for cat in data:
        cat_name = cat.get("category_e", cat.get("name", ""))
        tvl_exprs = cat.get("expression_t", [])
        en_exprs = cat.get("expression_e", [])
        jp_exprs = cat.get("expression_j", [])

        for i, (tvl, en) in enumerate(zip(tvl_exprs, en_exprs)):
            tvl = tvl.strip()
            en = en.strip()
            if not tvl or not en:
                continue

            record = {
                "id": f"tuvalu_app_expr_{cat['name']}_{i}",
                "tvl": tvl,
                "en": en,
                "content_type": "expression",
                "domain": "dictionary",
                "alignment_method": "index",
                "alignment_confidence": 1.0,
                "source": "tuvalu.aa-ken.jp",
                "source_url": f"{BASE_URL}/expressions.json",
                "category": cat_name,
                "subcategory": cat["name"],
                "tvl_chars": len(tvl),
                "en_chars": len(en),
                "length_ratio": round(len(tvl) / len(en), 3) if len(en) > 0 else 0,
            }
            # Include Japanese if available
            if i < len(jp_exprs):
                record["ja"] = jp_exprs[i].strip()

            records.append(record)

    return records


def scrape_words() -> list[dict]:
    """Scrape all word category JSON files."""
    records = []
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Build flat list of (category, subcategory, filename)
    subcats = []
    for cat in WORD_CATEGORIES:
        for sub in cat["subCategory"]:
            fname = subcategory_to_filename(sub)
            subcats.append((cat["category"], sub, fname))

    print(f"Fetching {len(subcats)} word subcategories")

    for category, subcategory, fname in tqdm(subcats, desc="Scraping words"):
        raw_path = RAW_DIR / fname

        if raw_path.exists() and raw_path.stat().st_size > 0:
            data = json.loads(raw_path.read_text())
        else:
            url = f"{BASE_URL}/{fname}"
            data = fetch_json(url)
            if data is None:
                tqdm.write(f"  FAILED: {fname}")
                continue
            raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

        words = data.get("words", [])
        for word in words:
            tvl = word.get("tuvalu", "").strip()
            en = word.get("english", "").strip()
            if not tvl or not en:
                continue

            record = {
                "id": f"tuvalu_app_word_{fname[:-5]}_{word.get('id', 0)}",
                "tvl": tvl,
                "en": en,
                "content_type": "word",
                "domain": "dictionary",
                "alignment_method": "index",
                "alignment_confidence": 1.0,
                "source": "tuvalu.aa-ken.jp",
                "source_url": f"{BASE_URL}/{fname}",
                "category": category,
                "subcategory": subcategory,
                "tvl_chars": len(tvl),
                "en_chars": len(en),
                "length_ratio": round(len(tvl) / len(en), 3) if len(en) > 0 else 0,
            }
            records.append(record)

    return records


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Scrape tuvalu.aa-ken.jp learning app (words + expressions)")
    parser.add_argument("--words-only", action="store_true",
                        help="Only scrape word categories")
    parser.add_argument("--expressions-only", action="store_true",
                        help="Only scrape expressions")
    args = parser.parse_args()

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = ALIGNED_DIR / "tuvalu_app.jsonl"

    all_records = []

    if not args.words_only:
        expr_records = scrape_expressions()
        all_records.extend(expr_records)
        print(f"Expressions: {len(expr_records)} pairs")

    if not args.expressions_only:
        word_records = scrape_words()
        all_records.extend(word_records)
        print(f"Words: {len(word_records)} pairs")

    # Write output
    with open(output_file, "w") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(all_records)} total pairs written to {output_file}")


if __name__ == "__main__":
    main()
