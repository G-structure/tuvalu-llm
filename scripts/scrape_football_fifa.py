"""Scrape FIFA.com football articles via CXM API sitemap + two-step article fetch.

Uses the open FIFA CXM API (no auth needed for article endpoints):
1. Sitemap pages → list of article URLs
2. Page API → resolve URL slug to Contentful entry ID
3. Article Section API → fetch article content (Contentful Rich Text JSON)

Usage:
    uv run python scripts/scrape_football_fifa.py                  # first 3 sitemap pages
    uv run python scripts/scrape_football_fifa.py --pages 10       # first 10 pages
    uv run python scripts/scrape_football_fifa.py --limit 50       # max 50 articles
    uv run python scripts/scrape_football_fifa.py --pages 106      # all ~10,579 articles
"""

import argparse
import html
import json
import sqlite3
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
from tqdm import tqdm

from clean_article_bodies import clean_body

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football" / "football.db"
SOURCE_ID = "fifa"

API_BASE = "https://cxm-api.fifa.com/fifaplusweb/api"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

RATE_DELAY = 1.0  # seconds between requests

# Category mapping from URL path segments / tags to our normalized categories
CATEGORY_MAP = {
    "premier league": "premier-league",
    "epl": "premier-league",
    "english premier league": "premier-league",
    "champions league": "champions-league",
    "ucl": "champions-league",
    "uefa champions league": "champions-league",
    "world cup": "world-cup",
    "fifa world cup": "world-cup",
    "wc qualifying": "world-cup",
    "ofc": "world-cup",
    "transfer": "transfers",
    "transfers": "transfers",
    "transfer news": "transfers",
    "rumour": "transfers",
    "la liga": "la-liga",
    "bundesliga": "bundesliga",
    "serie a": "serie-a",
}

# URL path segment → category (for FIFA tournament URLs)
PATH_CATEGORY_MAP = {
    "worldcup": "world-cup",
    "fifaworldcup": "world-cup",
    "world-cup": "world-cup",
    "championsleague": "champions-league",
    "champions-league": "champions-league",
    "clubworldcup": "club-world-cup",
    "u20worldcup": "world-cup",
    "u17worldcup": "world-cup",
    "womensworldcup": "world-cup",
    "womens-world-cup": "world-cup",
}


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def fetch_sitemap_page(client: httpx.Client, page: int) -> list[str]:
    """Fetch a single FIFA sitemap page and return list of article URLs."""
    url = f"{API_BASE}/sitemaps/articles/{page}"
    resp = client.get(url, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//sm:url/sm:loc", ns) if loc.text]
    return urls


def extract_relative_url(full_url: str) -> str:
    """Extract relative URL path from full FIFA.com URL for the Page API.

    Example:
        https://www.fifa.com/en/tournaments/.../articles/harry-kane
        → en/tournaments/.../articles/harry-kane
    """
    prefix = "https://www.fifa.com/"
    if full_url.startswith(prefix):
        return full_url[len(prefix):]
    # Fallback: strip scheme + host
    from urllib.parse import urlparse

    parsed = urlparse(full_url)
    return parsed.path.lstrip("/")


def extract_richtext(node: dict) -> str:
    """Recursively extract plain text from a Contentful Rich Text JSON node.

    Walks the tree collecting text values. Skips embedded-entry-block nodes
    (images, videos, widgets). Joins paragraphs with double newlines.
    """
    if not isinstance(node, dict):
        return ""

    node_type = node.get("nodeType", "")

    # Skip embedded entries (images, videos, social posts, etc.)
    if node_type == "embedded-entry-block":
        return ""

    # Leaf text node
    if node_type == "text":
        return node.get("value", "")

    # Recurse into children
    children = node.get("content", [])
    if not children:
        return ""

    # For block-level nodes, join children with appropriate separators
    if node_type in ("document",):
        parts = []
        for child in children:
            text = extract_richtext(child)
            if text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)

    if node_type in ("paragraph", "heading-2", "heading-3", "heading-4"):
        return "".join(extract_richtext(child) for child in children)

    if node_type == "unordered-list":
        items = []
        for child in children:
            text = extract_richtext(child)
            if text.strip():
                items.append(text.strip())
        return "\n".join(items)

    if node_type == "list-item":
        return "".join(extract_richtext(child) for child in children)

    if node_type == "hyperlink":
        return "".join(extract_richtext(child) for child in children)

    # Default: recurse and concatenate
    return "".join(extract_richtext(child) for child in children)


def map_category_from_url(url: str) -> str | None:
    """Try to infer category from FIFA.com URL path segments."""
    path = url.lower()
    for segment, category in PATH_CATEGORY_MAP.items():
        if segment in path:
            return category
    return None


def map_category_from_tags(tags: list[dict]) -> str | None:
    """Map FIFA page tags to our normalized category."""
    for tag in tags:
        # Tags have various structures; try common fields
        name = ""
        if isinstance(tag, dict):
            name = tag.get("name", "") or tag.get("title", "") or ""
        name_lower = name.lower()
        if name_lower in CATEGORY_MAP:
            return CATEGORY_MAP[name_lower]
    return None


def fetch_page_info(client: httpx.Client, relative_url: str) -> dict | None:
    """Step 1: Call Page API to resolve URL slug to entry ID and metadata."""
    url = f"{API_BASE}/pages/{relative_url}"
    try:
        resp = client.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return None


def fetch_article_section(client: httpx.Client, entry_id: str) -> dict | None:
    """Step 2: Fetch article content from the Article Section API."""
    url = f"{API_BASE}/sections/article/{entry_id}"
    try:
        resp = client.get(url, params={"locale": "en"}, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return None


def extract_article(
    page_data: dict, article_data: dict, original_url: str
) -> dict | None:
    """Extract normalized article fields from Page API + Article Section API responses."""
    try:
        # Entry ID (pageId from Page API = entryId from Article API)
        entry_id = page_data.get("pageId", "") or article_data.get("entryId", "")
        if not entry_id:
            return None

        # Title
        title = article_data.get("articleTitle", "")
        if not title:
            title = page_data.get("meta", {}).get("title", "")
        if not title:
            return None

        # Body text from Contentful Rich Text
        richtext = article_data.get("richtext")
        if not richtext:
            return None
        body_text = extract_richtext(richtext)
        if not body_text or len(body_text) < 50:
            return None
        # Normalize NBSP, BOM, narrow spaces
        body_text = clean_body(body_text, "fifa")

        # Published date
        published_at = article_data.get("articlePublishedDate", "")
        if not published_at:
            return None

        # Hero image
        hero = article_data.get("heroImage") or {}
        image_url = hero.get("src", "")
        image_alt = hero.get("alt", "")
        image_width = hero.get("width")
        image_height = hero.get("height")

        # OG description — prefer Page API meta, fallback to articlePreviewText
        meta = page_data.get("meta", {})
        og_description = meta.get("description", "") or article_data.get(
            "articlePreviewText", ""
        )

        # Category — try URL path first, then tags
        category = map_category_from_url(original_url)
        if not category:
            tags = page_data.get("tags", [])
            category = map_category_from_tags(tags)

        # Tags as JSON array
        raw_tags = page_data.get("tags", [])
        tag_names = []
        for t in raw_tags:
            if isinstance(t, dict):
                name = t.get("name") or t.get("title") or t.get("id")
                if name:
                    tag_names.append(name)
        tags_json = json.dumps(tag_names) if tag_names else "[]"

        # Canonical URL
        canonical_url = original_url
        rel_url = page_data.get("relativeUrl", "")
        if rel_url:
            canonical_url = f"https://www.fifa.com/{rel_url}"

        return {
            "id": entry_id,
            "url": canonical_url,
            "title_en": html.unescape(title),
            "body_en": html.unescape(body_text),
            "author": None,  # FIFA API has no author field
            "published_at": published_at,
            "category": category,
            "tags": tags_json,
            "image_url": image_url,
            "image_alt": image_alt,
            "image_width": image_width,
            "image_height": image_height,
            "og_description_en": html.unescape(og_description),
            "word_count": len(body_text.split()),
        }
    except (KeyError, TypeError, IndexError):
        return None


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
    parser = argparse.ArgumentParser(description="Scrape FIFA.com football articles")
    parser.add_argument("--limit", type=int, help="Max articles to scrape")
    parser.add_argument(
        "--pages", type=int, default=3, help="Max sitemap pages to fetch (default: 3)"
    )
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
    print(f"Found {len(existing)} existing FIFA.com articles in DB")

    client = httpx.Client(headers=HEADERS, follow_redirects=True, http2=True)

    # Fetch sitemap pages
    all_urls = []
    for page_num in range(args.pages):
        print(f"Fetching sitemap page {page_num}...")
        try:
            urls = fetch_sitemap_page(client, page_num)
            print(f"  Found {len(urls)} URLs on page {page_num}")
            all_urls.extend(urls)
        except httpx.HTTPError as e:
            print(f"  Error fetching page {page_num}: {e}")
            break
        time.sleep(RATE_DELAY)

    print(f"Total URLs from sitemaps: {len(all_urls)}")

    if args.limit:
        all_urls = all_urls[: args.limit]

    new_count = 0
    skip_count = 0
    fail_count = 0
    errors = []

    for url in tqdm(all_urls, desc="Scraping FIFA.com"):
        # Quick skip check based on URL (not perfect but avoids unnecessary API calls)
        # We'll also check entry_id after Step 1
        relative_url = extract_relative_url(url)

        # Step 1: Page API — get entry ID and metadata
        page_data = fetch_page_info(client, relative_url)
        time.sleep(RATE_DELAY)

        if page_data is None:
            fail_count += 1
            continue

        entry_id = page_data.get("pageId", "")
        if not entry_id:
            fail_count += 1
            continue

        # Skip if already in DB
        if entry_id in existing:
            skip_count += 1
            continue

        # Step 2: Article Section API — get article content
        article_data = fetch_article_section(client, entry_id)
        time.sleep(RATE_DELAY)

        if article_data is None:
            fail_count += 1
            continue

        article = extract_article(page_data, article_data, url)
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
        (SOURCE_ID, len(all_urls), new_count, json.dumps(errors) if errors else None),
    )
    conn.commit()
    conn.close()
    client.close()

    print(f"\nDone! {new_count} new articles, {skip_count} skipped, {fail_count} failed")
    print(f"Total FIFA.com articles in DB: {total}")


if __name__ == "__main__":
    main()
