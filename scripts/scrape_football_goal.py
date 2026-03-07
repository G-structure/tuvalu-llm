"""Scrape Goal.com football articles via sitemap + __NEXT_DATA__ extraction.

Fetches editorial-news.xml sitemap, then each article page, extracts structured
data from the __NEXT_DATA__ JSON blob. Stores in the football SQLite database.

Usage:
    uv run python scripts/scrape_football_goal.py                  # full scrape
    uv run python scripts/scrape_football_goal.py --limit 10       # first 10 articles
    uv run python scripts/scrape_football_goal.py --lists           # include list/slide articles
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

from clean_article_bodies import clean_body

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football" / "football.db"
SOURCE_ID = "goal"

SITEMAPS = {
    "news": "https://www.goal.com/en-us/sitemap/editorial-news.xml",
    "lists": "https://www.goal.com/en-us/sitemap/editorial-slides.xml",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}

RATE_DELAY = 1.0  # seconds between requests

# Category mapping from Goal.com tags to our normalized categories
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


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def fetch_sitemap(client: httpx.Client, url: str) -> list[str]:
    """Fetch sitemap XML and return list of article URLs."""
    resp = client.get(url, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//sm:url/sm:loc", ns) if loc.text]
    return urls


def extract_next_data(html: str) -> dict | None:
    """Extract __NEXT_DATA__ JSON from a Goal.com article page."""
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return None
    try:
        return json.loads(script.string)
    except json.JSONDecodeError:
        return None


def clean_body_html(body_html: str) -> str:
    """Strip ads, widgets, scripts, and promo content from article body HTML."""
    soup = BeautifulSoup(body_html, "html.parser")

    # Remove embedded scripts (betting, ads, video players, match widgets)
    for script in soup.find_all("script"):
        script.decompose()

    # Remove Instagram embeds, iframes, and blockquotes (social embeds)
    for tag in soup.find_all(["iframe", "blockquote"]):
        tag.decompose()

    # Get text, preserving paragraph breaks
    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if not text:
            continue
        # Skip READ MORE: link paragraphs
        if text.startswith("READ MORE:") or text.startswith("MORE:"):
            continue
        paragraphs.append(text)

    if paragraphs:
        body = "\n\n".join(paragraphs)
    else:
        body = soup.get_text(separator="\n\n", strip=True)

    # Apply source-specific cleaning (NordVPN, subscribe CTAs, hashtags, etc.)
    return clean_body(body, "goal")


def map_category(tags: list[dict]) -> str | None:
    """Map Goal.com tags to our normalized category."""
    for tag in tags:
        name = tag.get("name", "").lower()
        if name in CATEGORY_MAP:
            return CATEGORY_MAP[name]
    # Check page type tags
    for tag in tags:
        page_type = tag.get("link", {}).get("pageType", "")
        slug = tag.get("link", {}).get("slug", "").lower()
        if page_type == "COMPETITION":
            if slug in CATEGORY_MAP:
                return CATEGORY_MAP[slug]
    return None


def extract_news_article(data: dict) -> dict | None:
    """Extract article data from __NEXT_DATA__ for /news/ pages."""
    try:
        content = data["props"]["pageProps"]["content"]
        article = content["article"]
        page = data["props"]["pageProps"].get("page", {})
        meta = page.get("meta", {})
        og = meta.get("openGraph", {})

        body_html = article.get("body", {}).get("body", "")
        body_text = clean_body_html(body_html)
        if not body_text or len(body_text) < 50:
            return None

        tags = article.get("tagList", {}).get("tags", [])
        poster = article.get("poster", {})
        image = poster.get("image", {})

        return {
            "id": article.get("id", ""),
            "url": meta.get("seo", {}).get("canonicalUrl", ""),
            "title_en": html.unescape(article.get("headline", "")),
            "body_en": html.unescape(body_text),
            "author": (article.get("author") or [{}])[0].get("name"),
            "published_at": article.get("publishTime", ""),
            "category": map_category(tags),
            "tags": json.dumps([t.get("name") for t in tags]),
            "image_url": image.get("src", ""),
            "image_alt": image.get("alt", ""),
            "image_width": image.get("width"),
            "image_height": image.get("height"),
            "og_description_en": html.unescape(og.get("description", "")),
            "word_count": len(body_text.split()),
        }
    except (KeyError, TypeError, IndexError):
        return None


def extract_list_article(data: dict) -> dict | None:
    """Extract article data from __NEXT_DATA__ for /lists/ pages."""
    try:
        content = data["props"]["pageProps"]["content"]
        slide_list = content["slideList"]
        article = slide_list["article"]
        page = data["props"]["pageProps"].get("page", {})
        meta = page.get("meta", {})
        og = meta.get("openGraph", {})

        # Combine intro body + all slide bodies
        parts = []
        intro_html = article.get("body", {}).get("body", "")
        if intro_html:
            parts.append(clean_body_html(intro_html))

        for slide in slide_list.get("slides", []):
            headline = slide.get("headline", "")
            if headline:
                parts.append(f"**{headline}**")
            slide_html = slide.get("body", {}).get("body", "")
            if slide_html:
                parts.append(clean_body_html(slide_html))

        body_text = "\n\n".join(parts)
        if not body_text or len(body_text) < 50:
            return None

        tags = article.get("tagList", {}).get("tags", [])
        poster = article.get("poster", {})
        image = poster.get("image", {})

        return {
            "id": article.get("id", ""),
            "url": meta.get("seo", {}).get("canonicalUrl", ""),
            "title_en": html.unescape(article.get("headline", "")),
            "body_en": html.unescape(body_text),
            "author": (article.get("author") or [{}])[0].get("name"),
            "published_at": article.get("publishTime", ""),
            "category": map_category(tags),
            "tags": json.dumps([t.get("name") for t in tags]),
            "image_url": image.get("src", ""),
            "image_alt": image.get("alt", ""),
            "image_width": image.get("width"),
            "image_height": image.get("height"),
            "og_description_en": html.unescape(og.get("description", "")),
            "word_count": len(body_text.split()),
        }
    except (KeyError, TypeError, IndexError):
        return None


def scrape_article(client: httpx.Client, url: str, is_list: bool = False) -> dict | None:
    """Fetch and extract a single Goal.com article."""
    try:
        resp = client.get(url, timeout=30)
        if resp.status_code != 200:
            return None
    except httpx.HTTPError:
        return None

    data = extract_next_data(resp.text)
    if data is None:
        return None

    if is_list:
        return extract_list_article(data)
    return extract_news_article(data)


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
    parser = argparse.ArgumentParser(description="Scrape Goal.com football articles")
    parser.add_argument("--limit", type=int, help="Max articles to scrape")
    parser.add_argument("--lists", action="store_true", help="Also scrape list/slide articles")
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
    print(f"Found {len(existing)} existing Goal.com articles in DB")

    client = httpx.Client(headers=HEADERS, follow_redirects=True, http2=True)

    # Fetch sitemaps
    sitemap_types = ["news"]
    if args.lists:
        sitemap_types.append("lists")

    all_urls = []
    for st in sitemap_types:
        print(f"Fetching {st} sitemap...")
        urls = fetch_sitemap(client, SITEMAPS[st])
        print(f"  Found {len(urls)} URLs in {st} sitemap")
        all_urls.extend([(url, st == "lists") for url in urls])

    if args.limit:
        all_urls = all_urls[: args.limit]

    new_count = 0
    skip_count = 0
    fail_count = 0
    errors = []

    for url, is_list in tqdm(all_urls, desc="Scraping Goal.com"):
        # Extract ID from URL to check if already scraped
        # URL pattern: /en-us/news/{slug}/{blt_id} or /en-us/lists/{slug}/{blt_id}
        url_id = url.rstrip("/").split("/")[-1]
        if url_id in existing:
            skip_count += 1
            continue

        article = scrape_article(client, url, is_list=is_list)
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
        (SOURCE_ID, len(all_urls), new_count, json.dumps(errors) if errors else None),
    )
    conn.commit()
    conn.close()
    client.close()

    print(f"\nDone! {new_count} new articles, {skip_count} skipped, {fail_count} failed")
    print(f"Total Goal.com articles in DB: {total}")


if __name__ == "__main__":
    main()
