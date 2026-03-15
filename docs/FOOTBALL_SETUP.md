# Talafutipolo — Local Setup & Run Guide

Football news in Tuvaluan and English. Scrapes articles from Goal.com, FIFA.com,
and Sky Sports, translates them to Tuvaluan via the Tinker API, stores everything
in SQLite, and serves a bilingual news site via SolidStart.

For the hackathon, this app matters because it is the live feedback flywheel:
users read Tuvaluan content, react to paragraphs, submit correction notes and
mode preferences, and those interactions can be exported into normalized JSONL
for future post-training.

## Fastest Demo

```bash
uv sync
uv run python scripts/init_football_db.py
uv run python scripts/pipeline_football.py --scrape-limit 10 --translate-limit 10
cd site && npm install && npm run dev
```

Then:

1. Open a translated article.
2. Use the thumbs feedback and `Coach the Translator` form.
3. Visit `/fatele` to show community totals.
4. Run `uv run python scripts/export_football_interactions.py`.

That is the shortest path to demonstrating the live product plus the data export loop.

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | >= 3.14 | `brew install python` or [python.org](https://www.python.org/) |
| uv | any | `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js | >= 18 | `brew install node` or [nodejs.org](https://nodejs.org/) |
| npm | >= 9 | comes with Node.js |

You also need a `TINKER_API_KEY` in `.env` for translations. Without it, scraping
works but articles will only have English text.

## Project structure

```
scripts/
  init_football_db.py             # creates SQLite/D1 schema
  scrape_football_goal.py         # Goal.com scraper
  scrape_football_fifa.py         # FIFA.com scraper
  scrape_football_sky.py          # Sky Sports scraper
  translate_football.py           # Tuvaluan translation via Tinker API
  pipeline_football.py            # scrape + translate pipeline
  export_football_interactions.py # JSONL export for feedback/poll data
tv/
  apps/
    football/                     # shared football app/repository/export logic
site/
  src/
    routes/                       # SolidStart routes
    components/                   # football UI components
    lib/                          # site-side DB/types helpers
data/
  football/
    football.db                   # local SQLite database
docs/
  FOOTBALL_SETUP.md               # local setup guide
  football_site_plan.md           # architecture + extraction notes
```

## Data Flywheel Outputs

The football stack currently produces four useful classes of signal:

- paragraph-level positive or negative feedback
- article-level helpfulness and preferred reading mode
- free-text correction suggestions
- implicit reveal/share-style engagement events

The export step writes those into normalized JSONL artifacts plus a manifest so
they can be consumed later by ranking, filtering, or post-training pipelines.

## Quick start

### 1. Install Python dependencies

```bash
uv sync
```

This installs `httpx`, `beautifulsoup4`, `lxml`, `tqdm`, and other deps from
`pyproject.toml`.

### 2. Initialize the database

```bash
uv run python scripts/init_football_db.py
```

Creates `data/football/football.db` with the schema and seeds the `sources` table
(goal, fifa, sky).

### 3. Scrape articles

Run any or all of these. Each scraper is independent and idempotent — rerunning
skips already-scraped articles.

```bash
# Goal.com — ~205 articles in news sitemap, ~5,000 in lists sitemap
uv run python scripts/scrape_football_goal.py               # all news articles
uv run python scripts/scrape_football_goal.py --limit 50    # first 50 only
uv run python scripts/scrape_football_goal.py --lists       # include list/slide articles

# FIFA.com — ~10,600 articles across 106 sitemap pages
uv run python scripts/scrape_football_fifa.py               # 3 sitemap pages (~300 articles)
uv run python scripts/scrape_football_fifa.py --pages 10    # 10 pages (~1,000 articles)
uv run python scripts/scrape_football_fifa.py --limit 20    # cap at 20 articles

# Sky Sports — ~25 football articles per sitemap refresh
uv run python scripts/scrape_football_sky.py                # all football articles
uv run python scripts/scrape_football_sky.py --limit 10     # first 10 only
```

Each scraper:
- Fetches the sitemap / API index to discover article URLs
- Fetches each article page and extracts structured data
- Stores title, body text, author, date, category, hero image, OG metadata
- Rate-limits to 1 req/sec (polite crawling)
- Logs fetch stats to the `fetch_log` table

### 4. Translate articles to Tuvaluan

Requires `TINKER_API_KEY` in `.env`.

```bash
# Translate all untranslated articles
uv run python scripts/translate_football.py

# Translate a specific number
uv run python scripts/translate_football.py --limit 20

# Translate a specific article
uv run python scripts/translate_football.py --article ARTICLE_ID
```

The translator:
- Picks untranslated articles (no entry in `translations` table)
- Translates title, OG description, and body paragraph-by-paragraph
- Uses the Stage A adapter via Tinker `/completions` endpoint
- Retries with exponential backoff on 429/500/timeout
- Stores results in the `translations` table with paragraph alignment preserved
- Takes ~2-3 minutes per article (sequential paragraph translation)

### 5. One-command pipeline

Or run everything at once:

```bash
# Scrape all sources + translate new articles
uv run python scripts/pipeline_football.py

# Limit scraping per source
uv run python scripts/pipeline_football.py --scrape-limit 10

# Just translate (skip scraping)
uv run python scripts/pipeline_football.py --translate-only

# Just scrape (skip translation)
uv run python scripts/pipeline_football.py --scrape-only
```

### 5b. Export interaction data for future RL / preference tuning

```bash
# Export from the default local football DB
uv run python scripts/export_football_interactions.py

# Write to a custom directory
uv run python scripts/export_football_interactions.py \
  --output-dir data/football/exports/demo_run

# Skip implicit reveal/share signals and keep only explicit feedback
uv run python scripts/export_football_interactions.py --skip-implicit
```

The export writes:

- `explicit_feedback.jsonl` — paragraph votes and richer article-level feedback
- `corrections.jsonl` — free-text Tuvaluan correction suggestions with article context
- `implicit_signals.jsonl` — reveal/share-style engagement events
- `football_polls.jsonl` — poll or prediction votes when those tables exist
- `manifest.json` — file paths and row counts

By default the artifacts land in `data/football/exports/interactions/`.
If Cloudflare D1 env vars are set, the exporter uses the same env-based backend
selection as the football scripts; otherwise it reads the local SQLite DB.

### 6. Install site dependencies

```bash
cd site
npm install
```

### 7. Start the dev server

```bash
cd site
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

The site reads directly from `data/football/football.db` (one directory up from
`site/`). No separate API server needed. Changes to the database (new scrapes,
new translations) appear on the next page load.

For the current Cloudflare-backed app flow, use `scripts/sync_to_d1.py` after
scraping/translating so the site and the interaction export operate on the same
data store.

## Site features

### Bilingual experience

- **TVL-first**: Article cards show Tuvaluan title first, English subtitle below
  in muted italic. The site is Tuvaluan-first by design.
- **Language toggle**: On article pages, cycle through three modes:
  - **TV** (default) — Tuvaluan text only, tap "Fakakite English" on any
    paragraph to reveal the English underneath
  - **EN** — English text only
  - **TV+EN** — Stacked bilingual view (TVL paragraph, then EN indented below)
- **Graceful fallback**: Articles without translations show English only with no
  broken UI. As translations are added, the bilingual UX appears automatically.

### Pages

- **Homepage** (`/`): Hero card for latest article + thumbnail list, category
  filter pills, and a `Te Fatele` call-to-action that points users to the
  coaching loop
- **Article page** (`/articles/[id]`): Hero image, bilingual title, paragraph-by-
  paragraph body with language toggle, paragraph thumbs feedback, a `Coach the
  Translator` form for explicit article-level feedback / preferred mode /
  correction text, `Fakasoa` (share) button, and source attribution
- **Category page** (`/category/[slug]`): Same layout filtered by category
  (premier-league, world-cup, transfers, champions-league, etc.)
- **Fatele** (`/fatele`): Community dashboard with monthly signals, island
  participation, mode-preference counts, and correction totals

### Mobile-first design

- 360px primary viewport, 48px touch targets, system fonts
- Lazy-loaded images with width/height hints to prevent layout shift
- No hamburger menus — categories are horizontal scroll pills

### Social sharing

- OpenGraph + Twitter Card meta tags on every page with `og:locale=tvl`
- "Fakasoa" share button uses Web Share API (native share sheet on mobile)
- When shared on WhatsApp/Facebook, the preview card shows Tuvaluan title + description

## Scraper details

### Goal.com

- **Method**: Fetch `editorial-news.xml` sitemap, then each article page.
  Extract `__NEXT_DATA__` JSON blob from `<script id="__NEXT_DATA__">`.
- **Content path**: `props.pageProps.content.article.body.body` (HTML string)
- **Images**: `assets.goal.com` CDN, supports `?width=N` query param resizing
- **Lists articles** (`--lists`): Different structure — `content.slideList` with
  intro body + `slides[]` array. Each slide has its own headline, body, and image.
- **robots.txt**: Fully permissive (`Allow: /`)

### FIFA.com

- **Method**: Two-step API fetch. (1) Page API resolves URL slug to Contentful
  entry ID. (2) Article Section API returns content.
- **API base**: `https://cxm-api.fifa.com/fifaplusweb/api` (no auth needed)
- **Content format**: Contentful Rich Text JSON — tree of `paragraph`, `heading`,
  `text` nodes. The scraper walks the tree recursively to extract plain text.
- **Images**: `digitalhub.fifa.com` CDN with `?io=transform:fill,width:N` params
- **Sitemaps**: Paginated at `/sitemaps/articles/{0-105}`, 100 per page
- **robots.txt**: Articles allowed

### Sky Sports

- **Method**: Fetch `sitemap-news.xml`, filter URLs containing `/football/`,
  then extract JSON-LD `NewsArticle` from each page.
- **Content path**: `articleBody` field in JSON-LD — full article as clean plain
  text (no HTML parsing needed)
- **Images**: `e0.365dm.com` CDN. Swap the resolution segment in the URL to get
  different sizes (e.g., `768x432` to `1600x900` for OG images).
- **Football filter**: URL contains `/football/` or keywords contains `soccer`
- **robots.txt**: Blocks named AI bots (GPTBot, CCBot) but allows generic crawlers

### Translation (Tinker API)

- **Endpoint**: `POST /completions` at Tinker OpenAI-compatible API
- **Model**: Stage A translation adapter (`tinker://a6453cc0-...`)
- **Auth**: `X-Api-Key` header with `TINKER_API_KEY`
- **Strategy**: Paragraph-by-paragraph translation with the training prompt template
- **Rate**: Sequential, 0.5s delay between requests, exponential backoff on errors
- **Quality**: Rough but functional — the base model has limited Tuvaluan training
  data. Football terms are often preserved as English loanwords.

## Database schema

```
articles          — scraped English articles (id, source_id, url, title_en, body_en,
                    author, published_at, category, tags, image_url, image_alt,
                    image_width, image_height, og_description_en, word_count)

translations      — Tuvaluan translations (article_id, title_tvl, body_tvl,
                    og_description_tvl, model_path, paragraph_count, failed_paragraphs)

sources           — source metadata (id, name, last_fetched_at, article_count)

fetch_log         — scrape run history (source_id, articles_found, articles_new, errors)
```

Query articles with translations joined:

```sql
SELECT a.*, t.title_tvl, t.body_tvl
FROM articles a
LEFT JOIN translations t ON t.article_id = a.id
ORDER BY a.published_at DESC
LIMIT 20;
```

## Common tasks

### Check database stats

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('data/football/football.db')
for row in conn.execute('SELECT source_id, COUNT(*), SUM(word_count) FROM articles GROUP BY source_id'):
    print(f'  {row[0]:6s}: {row[1]:4d} articles, {row[2]:,d} words')
total = conn.execute('SELECT COUNT(*) FROM articles').fetchone()[0]
translated = conn.execute('SELECT COUNT(*) FROM translations').fetchone()[0]
print(f'  Total: {total} articles, {translated} translated')
"
```

### Reset the database

```bash
rm data/football/football.db
uv run python scripts/init_football_db.py
```

### Build for production

```bash
cd site
npm run build
npm run start
```

### Quick demo (scrape + translate a small batch)

```bash
uv run python scripts/init_football_db.py
uv run python scripts/pipeline_football.py --scrape-limit 10 --translate-limit 10
uv run python scripts/sync_to_d1.py
cd site && npm install && npm run dev
```

Then:

1. Open any translated article.
2. Submit the `Coach the Translator` form.
3. Open `/fatele` to confirm the signal shows up in the community totals.
4. Run `uv run python scripts/export_football_interactions.py` to generate the
   normalized JSONL export.

### Full scrape (all sources)

```bash
uv run python scripts/init_football_db.py
uv run python scripts/scrape_football_goal.py --lists
uv run python scripts/scrape_football_fifa.py --pages 106
uv run python scripts/scrape_football_sky.py
uv run python scripts/translate_football.py
```

This will take a while (~16,000+ articles, translation at ~2-3 min each).
For a quick demo, use `--limit 50` on each scraper and translator.

## What's next (not yet built)

- **Cron automation**: Periodic scraping + translation via API routes or workers.
- **Richer gamified RL**: today the app captures paragraph votes, article-level
  coaching submissions, preferred reading mode, and correction text. Future work
  can add deeper A/B ranking, reviewer workflows, and direct post-training loops.
- **Island selector**: First-visit island picker for community contribution tracking.
- **Service Worker**: Cache last 20 articles for offline reading on outer islands.
- **Deployment**: Cloudflare Pages with Turso (edge SQLite) for production.
