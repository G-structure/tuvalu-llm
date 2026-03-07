"""Scrape Sky Sports football articles via news sitemap + JSON-LD extraction.

Fetches sitemap-news.xml, filters for football articles (URL contains /football/
or keywords=soccer), then extracts structured data from JSON-LD NewsArticle blocks
for metadata, and HTML body (div.sdc-article-body > p) for clean article text.
Stores in the shared football SQLite database.

Usage:
    uv run python scripts/scrape_football_sky.py                  # full scrape
    uv run python scripts/scrape_football_sky.py --limit 10       # first 10 articles
"""

import argparse
import html
import json
import re
import sqlite3
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm

from clean_article_bodies import clean_body, split_into_paragraphs

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football" / "football.db"
SOURCE_ID = "sky"

SITEMAP_URL = "https://www.skysports.com/sitemap/sitemap-news.xml"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}

RATE_DELAY = 1.0  # seconds between requests

# Section ID to category mapping (from observed URL patterns)
SECTION_MAP = {
    "11095": "premier-league",
    "11661": "premier-league",  # Tottenham
    "11667": "premier-league",  # Manchester United
    "11688": "championship",
    "11735": "championship",    # Millwall
    "11781": "scottish",
    "36621": "scottish",        # Scottish Premiership
    "12709": "transfers",
}


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def fetch_sitemap(client: httpx.Client) -> list[str]:
    """Fetch news sitemap XML and return football article URLs."""
    resp = client.get(SITEMAP_URL, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    ns = {
        "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
        "news": "http://www.google.com/schemas/sitemap-news/0.9",
    }

    urls = []
    for url_el in root.findall(".//sm:url", ns):
        loc = url_el.find("sm:loc", ns)
        if loc is None or not loc.text:
            continue

        loc_text = loc.text.strip()

        # Filter for football: URL contains /football/
        if "/football/" in loc_text:
            urls.append(loc_text)
            continue

        # Alternative filter: news:keywords = soccer
        keywords_el = url_el.find(".//news:keywords", ns)
        if keywords_el is not None and keywords_el.text:
            if "soccer" in keywords_el.text.lower():
                urls.append(loc_text)

    return urls


def extract_article_id(url: str) -> str | None:
    """Extract the numeric article ID from a Sky Sports URL.

    URL pattern: /football/news/{section_id}/{article_id}/{slug}
    """
    match = re.search(r"/news/\d+/(\d+)/", url)
    if match:
        return match.group(1)
    # Fallback: try last numeric segment
    parts = url.rstrip("/").split("/")
    for part in reversed(parts):
        if part.isdigit() and len(part) >= 5:
            return part
    return None


def extract_section_id(url: str) -> str | None:
    """Extract the section ID from a Sky Sports URL."""
    match = re.search(r"/news/(\d+)/\d+/", url)
    if match:
        return match.group(1)
    return None


def map_category(section_id: str | None, genre: str | None) -> str | None:
    """Map section ID or genre to a normalized category."""
    if section_id and section_id in SECTION_MAP:
        return SECTION_MAP[section_id]
    if genre and genre.lower() == "soccer":
        return "football"
    return None


def swap_image_resolution(url: str, target: str = "1600x900") -> str:
    """Swap the resolution segment in a Sky Sports CDN image URL.

    e.g. e0.365dm.com/26/03/2048x1152/skysports-xxx.jpg
      -> e0.365dm.com/26/03/1600x900/skysports-xxx.jpg
    """
    return re.sub(r"/\d+x\d+/", f"/{target}/", url, count=1)


def extract_json_ld(html: str) -> dict | None:
    """Extract the NewsArticle JSON-LD block from the page HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
        except json.JSONDecodeError:
            continue

        # Handle both single object and array of objects
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("@type") == "NewsArticle":
                    return item
        elif isinstance(data, dict):
            if data.get("@type") == "NewsArticle":
                return data

    return None


def extract_body_from_html(page_html: str) -> str:
    """Extract clean article body from Sky Sports HTML.

    Uses direct <p> children of div.sdc-article-body which contain only
    editorial text — promo blocks, CTAs, and related articles are in
    separate elements and are excluded.

    Falls back to JSON-LD articleBody if HTML parsing finds nothing.
    """
    soup = BeautifulSoup(page_html, "html.parser")
    body_div = soup.select_one("div.sdc-article-body")
    if body_div:
        paragraphs = []
        for p in body_div.find_all("p", recursive=False):
            text = p.get_text(strip=True)
            if text:
                paragraphs.append(text)
        if paragraphs:
            return "\n\n".join(paragraphs)

    # Fallback: JSON-LD articleBody
    ld = extract_json_ld(page_html)
    if ld:
        return ld.get("articleBody", "")
    return ""


def extract_og_description(page_html: str) -> str:
    """Extract og:description from meta tags (better than empty JSON-LD description)."""
    soup = BeautifulSoup(page_html, "html.parser")
    tag = soup.find("meta", property="og:description")
    if tag and tag.get("content"):
        return tag["content"]
    return ""


def extract_article(page_html: str, url: str) -> dict | None:
    """Extract article data from a Sky Sports article page.

    Uses HTML body (div.sdc-article-body > p) for clean text,
    JSON-LD for metadata (headline, author, image, dates).
    """
    ld = extract_json_ld(page_html)
    if ld is None:
        return None

    # Extract body from HTML (clean paragraphs), fallback to JSON-LD
    article_body = extract_body_from_html(page_html)
    if not article_body or len(article_body) < 50:
        return None

    # Clean promo artifacts and normalize whitespace
    article_body = clean_body(article_body, "sky")
    article_body = split_into_paragraphs(article_body)

    article_id = extract_article_id(url)
    if not article_id:
        return None

    section_id = extract_section_id(url)
    genre = ld.get("genre")

    # Author
    author_data = ld.get("author")
    author = None
    if isinstance(author_data, dict):
        author = author_data.get("name")
    elif isinstance(author_data, list) and author_data:
        author = author_data[0].get("name") if isinstance(author_data[0], dict) else None

    # Image from JSON-LD (2048x1152)
    image_data = ld.get("image", {})
    if isinstance(image_data, list) and image_data:
        image_data = image_data[0] if isinstance(image_data[0], dict) else {}
    elif not isinstance(image_data, dict):
        image_data = {}

    image_url = image_data.get("url", "")
    image_width = image_data.get("width")
    image_height = image_data.get("height")

    # Image alt from JSON-LD (may not exist) or headline as fallback
    image_alt = image_data.get("name", "") or image_data.get("description", "")

    # Word count from cleaned body
    word_count = len(article_body.split())

    # OG description (JSON-LD description is usually empty)
    og_description = extract_og_description(page_html)

    # Tags: use genre + section info
    tags = []
    if genre:
        tags.append(genre)
    alt_headline = ld.get("alternativeHeadline", "")
    if alt_headline and alt_headline != ld.get("headline", ""):
        tags.append(f"alt:{alt_headline}")

    return {
        "id": article_id,
        "url": url,
        "title_en": html.unescape(ld.get("headline", "")),
        "body_en": html.unescape(article_body),
        "author": author,
        "published_at": ld.get("datePublished", ""),
        "category": map_category(section_id, genre),
        "tags": json.dumps(tags),
        "image_url": image_url,
        "image_alt": image_alt,
        "image_width": image_width,
        "image_height": image_height,
        "og_description_en": html.unescape(og_description),
        "word_count": word_count,
    }


def scrape_article(client: httpx.Client, url: str) -> dict | None:
    """Fetch and extract a single Sky Sports article."""
    try:
        resp = client.get(url, timeout=30)
        if resp.status_code != 200:
            return None
    except httpx.HTTPError:
        return None

    return extract_article(resp.text, url)


def insert_article(conn: sqlite3.Connection, article: dict) -> bool:
    """Insert article into DB. Returns True if new, False if duplicate."""
    try:
        conn.execute(
            """INSERT INTO articles
               (id, source_id, url, title_en, body_en, author, published_at,
                category, tags, image_url, image_alt, image_width, image_height,
                og_description_en, word_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                article["id"],
                SOURCE_ID,
                article["url"],
                article["title_en"],
                article["body_en"],
                article["author"],
                article["published_at"],
                article["category"],
                article["tags"],
                article["image_url"],
                article["image_alt"],
                article["image_width"],
                article["image_height"],
                article["og_description_en"],
                article["word_count"],
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Scrape Sky Sports football articles")
    parser.add_argument("--limit", type=int, help="Max articles to scrape")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}. Run init_football_db.py first.")
        return

    conn = get_db()

    # Get existing article IDs to skip
    existing = set(
        row[0]
        for row in conn.execute(
            "SELECT id FROM articles WHERE source_id = ?", (SOURCE_ID,)
        )
    )
    print(f"Found {len(existing)} existing Sky Sports articles in DB")

    client = httpx.Client(headers=HEADERS, follow_redirects=True, http2=True)

    # Fetch sitemap and filter for football
    print("Fetching news sitemap...")
    football_urls = fetch_sitemap(client)
    print(f"  Found {len(football_urls)} football URLs in sitemap")

    if args.limit:
        football_urls = football_urls[: args.limit]

    new_count = 0
    skip_count = 0
    fail_count = 0
    errors = []

    for url in tqdm(football_urls, desc="Scraping Sky Sports"):
        # Extract ID from URL to check if already scraped
        url_id = extract_article_id(url)
        if url_id and url_id in existing:
            skip_count += 1
            continue

        article = scrape_article(client, url)
        time.sleep(RATE_DELAY)

        if article is None:
            fail_count += 1
            continue

        if article["id"] in existing:
            skip_count += 1
            continue

        if insert_article(conn, article):
            new_count += 1
            existing.add(article["id"])
            if new_count % 10 == 0:
                conn.commit()
        else:
            skip_count += 1

    conn.commit()

    # Update source metadata
    total = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE source_id = ?", (SOURCE_ID,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE sources SET last_fetched_at = datetime('now'), article_count = ? WHERE id = ?",
        (total, SOURCE_ID),
    )
    conn.execute(
        "INSERT INTO fetch_log (source_id, articles_found, articles_new, errors) VALUES (?, ?, ?, ?)",
        (SOURCE_ID, len(football_urls), new_count, json.dumps(errors) if errors else None),
    )
    conn.commit()
    conn.close()
    client.close()

    print(f"\nDone! {new_count} new articles, {skip_count} skipped, {fail_count} failed")
    print(f"Total Sky Sports articles in DB: {total}")


if __name__ == "__main__":
    main()
