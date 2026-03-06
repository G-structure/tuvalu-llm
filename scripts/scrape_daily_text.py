"""Scrape WOL daily text pages, date-aligned.

Uses Docker curl-impersonate for fetching (browser-like TLS fingerprint).

Usage:
    uv run python scripts/scrape_daily_text.py --year 2025
    uv run python scripts/scrape_daily_text.py --range 2025-03-01 2025-03-05
"""

import re
import sys
import json
import argparse
from datetime import date, timedelta
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

# Add scripts dir to path for fetch module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import fetch, fetch_and_save

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
ALIGNED_DIR = DATA_DIR / "aligned"

TVL_BUNDLE = {"lang": "tvl", "rcode": "r153", "lpcode": "lp-vl"}
EN_BUNDLE = {"lang": "en", "rcode": "r1", "lpcode": "lp-e"}

# Note: month and day are NOT zero-padded in WOL daily text URLs
WOL_DAILY_URL = "https://wol.jw.org/{lang}/wol/h/{rcode}/{lpcode}/{yyyy}/{m}/{d}"


def make_daily_url(bundle: dict, dt: date) -> str:
    """Build WOL daily text URL for a given date and language bundle."""
    return WOL_DAILY_URL.format(
        yyyy=dt.year,
        m=dt.month,
        d=dt.day,
        **bundle,
    )


def extract_daily_text(html: str, target_date: date) -> str | None:
    """Extract daily text (theme + commentary) for a specific date from HTML.

    The page contains multiple div.tabContent elements with data-date attributes.
    Each has p.themeScrp (theme scripture) and p.sb (commentary).
    """
    soup = BeautifulSoup(html, "html5lib")
    target_prefix = target_date.isoformat()  # e.g., "2025-03-05"

    for tab in soup.find_all("div", class_="tabContent"):
        data_date = tab.get("data-date", "")
        # data-date format: "2025-03-05T00:00:00.000Z"
        if not data_date.startswith(target_prefix):
            continue

        # Remove cross-refs and other link elements before text extraction
        for unwanted in tab.find_all("a", class_="b"):
            unwanted.decompose()
        for unwanted in tab.find_all("a", class_="fn"):
            unwanted.decompose()
        for sup in tab.find_all("sup"):
            sup.decompose()

        # Extract theme scripture
        theme_el = tab.find("p", class_="themeScrp")
        theme_text = ""
        if theme_el:
            theme_text = theme_el.get_text(strip=True)
            theme_text = re.sub(r"\s+", " ", theme_text).strip()

        # Extract commentary
        commentary_parts = []
        for sb_el in tab.find_all("p", class_="sb"):
            text = sb_el.get_text(strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                commentary_parts.append(text)
        commentary_text = " ".join(commentary_parts)

        if theme_text or commentary_text:
            parts = [p for p in [theme_text, commentary_text] if p]
            return "\n\n".join(parts)

    return None


def generate_date_range(start: date, end: date) -> list[date]:
    """Generate a list of dates from start to end (inclusive)."""
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(
        description="Scrape WOL daily text pages, date-aligned.")
    parser.add_argument("--year", type=int,
                        help="Scrape all daily texts for a given year")
    parser.add_argument("--range", nargs=2, metavar=("START", "END"),
                        help="Scrape daily texts for a date range (YYYY-MM-DD YYYY-MM-DD)")
    args = parser.parse_args()

    if args.year:
        start = date(args.year, 1, 1)
        end = date(args.year, 12, 31)
        dates = generate_date_range(start, end)
    elif args.range:
        start = date.fromisoformat(args.range[0])
        end = date.fromisoformat(args.range[1])
        dates = generate_date_range(start, end)
    else:
        print("Specify --year YYYY or --range YYYY-MM-DD YYYY-MM-DD")
        sys.exit(1)

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = ALIGNED_DIR / "daily_text.jsonl"

    # Load existing data to support resume
    existing_dates = set()
    existing_count = 0
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                row = json.loads(line)
                if row.get("date"):
                    existing_dates.add(row["date"])
                existing_count += 1
        print(f"Found {existing_count} existing daily text pairs "
              f"({len(existing_dates)} dates), will skip duplicates")

    # Filter out already-scraped dates
    dates_to_scrape = [d for d in dates if d.isoformat() not in existing_dates]
    if not dates_to_scrape:
        print("All dates already scraped, nothing to do.")
        return

    print(f"Scraping {len(dates_to_scrape)} dates "
          f"(skipping {len(dates) - len(dates_to_scrape)} already done)")

    # Track dates already extracted from multi-day pages
    extracted_from_page = {}  # date_iso -> (tvl_text, en_text) extracted from adjacent fetch

    total_new = 0
    failed = 0

    with open(output_file, "a") as out:
        for dt in tqdm(dates_to_scrape, desc="Scraping daily text"):
            dt_iso = dt.isoformat()

            # Skip if already written (handles adjacent-day duplicates)
            if dt_iso in existing_dates:
                continue

            # Check if we already extracted this date from a previous page fetch
            if dt_iso in extracted_from_page:
                tvl_text, en_text = extracted_from_page[dt_iso]
            else:
                # Fetch TVL page
                tvl_url = make_daily_url(TVL_BUNDLE, dt)
                raw_tvl_path = str(RAW_DIR / "wol_tvl" /
                                   f"daily_{dt.year}_{dt.month:02d}_{dt.day:02d}.html")
                tvl_html = fetch_and_save(tvl_url, raw_tvl_path)

                # Fetch EN page
                en_url = make_daily_url(EN_BUNDLE, dt)
                raw_en_path = str(RAW_DIR / "wol_en" /
                                  f"daily_{dt.year}_{dt.month:02d}_{dt.day:02d}.html")
                en_html = fetch_and_save(en_url, raw_en_path)

                if tvl_html is None or en_html is None:
                    tqdm.write(f"  SKIP {dt_iso} -- fetch failed "
                               f"(TVL: {'ok' if tvl_html else 'fail'}, "
                               f"EN: {'ok' if en_html else 'fail'})")
                    failed += 1
                    continue

                # Extract the target date and also try adjacent dates from this page
                tvl_text = extract_daily_text(tvl_html, dt)
                en_text = extract_daily_text(en_html, dt)

                # Optimization: extract adjacent dates from the same page
                # (each page returns ~3 consecutive days)
                for offset in [-1, 1, 2]:
                    adj_dt = dt + timedelta(days=offset)
                    adj_iso = adj_dt.isoformat()
                    if (adj_iso not in existing_dates and
                            adj_iso not in extracted_from_page):
                        adj_tvl = extract_daily_text(tvl_html, adj_dt)
                        adj_en = extract_daily_text(en_html, adj_dt)
                        if adj_tvl and adj_en:
                            extracted_from_page[adj_iso] = (adj_tvl, adj_en)

            if not tvl_text or not en_text:
                tqdm.write(f"  SKIP {dt_iso} -- extraction failed "
                           f"(TVL: {'ok' if tvl_text else 'empty'}, "
                           f"EN: {'ok' if en_text else 'empty'})")
                failed += 1
                continue

            tvl_chars = len(tvl_text)
            en_chars = len(en_text)

            tvl_url = make_daily_url(TVL_BUNDLE, dt)
            en_url = make_daily_url(EN_BUNDLE, dt)

            record = {
                "id": f"daily_{dt_iso}",
                "tvl": tvl_text,
                "en": en_text,
                "content_type": "daily_text",
                "domain": "daily_text",
                "alignment_method": "date",
                "alignment_confidence": 1.0,
                "doc_id": None,
                "source_url_tvl": tvl_url,
                "source_url_en": en_url,
                "book_num": None,
                "chapter": None,
                "verse": None,
                "date": dt_iso,
                "pub_code": None,
                "tvl_chars": tvl_chars,
                "en_chars": en_chars,
                "length_ratio": round(tvl_chars / en_chars, 3) if en_chars > 0 else 0,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_dates.add(dt_iso)
            total_new += 1

    print(f"\nDone! {total_new} new daily text pairs written ({failed} failed)")
    print(f"Total pairs in {output_file}: {existing_count + total_new}")


if __name__ == "__main__":
    main()
