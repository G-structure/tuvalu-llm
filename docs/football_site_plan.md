# Tuvaluan Football News — SolidJS site plan

A SolidJS website that fetches English football articles from Goal.com, FIFA.com,
and Sky Sports, translates them to Tuvaluan via the Stage A adapter, and serves
them as a bilingual news mirror.

## Audience

Tuvaluans (~11,000 speakers) and the Tuvaluan diaspora. Pacific islanders are
among the most passionate football fans in the world — the Premier League is the
most-watched league across the Pacific. A Tuvaluan football news site in their
own language would be the first of its kind.

### Content priorities (by Tuvaluan interest)

1. **Premier League** — the league Pacific islanders follow (Liverpool, Man United,
   Arsenal are the biggest Pacific fanbases)
2. **FIFA World Cup qualifiers** — especially OFC qualifying where Tuvalu competes
3. **Transfer news** — PL transfers dominate Pacific football conversation
4. **Champions League / big tournaments** — as neutral spectators
5. **OFC / Pacific football** — regional tournaments, Oceania Champions League

### Source mapping

| Priority | Content type | Best source | Why |
|----------|-------------|-------------|-----|
| 1 | Premier League news | Goal.com, Sky Sports | Daily PL coverage, match reports |
| 2 | World Cup / OFC | FIFA.com | Only source with OFC + small nation coverage |
| 3 | Transfers | Goal.com, Sky Sports | Dedicated transfer sections |
| 4 | Champions League | Goal.com, FIFA.com | Broad European coverage |
| 5 | Pacific football | FIFA.com | OFC tournament articles |

## Architecture

```
                    +-------------------+
                    |   Cron / Worker   |
                    |  (fetch + xlate)  |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
         Goal.com       FIFA.com      Sky Sports
         (sitemap)      (CXM API)     (sitemap)
              |              |              |
              v              v              v
         +---------------------------------+
         |        SQLite / Turso DB        |
         |  articles, translations, meta   |
         +---------------------------------+
                         |
                    +----+----+
                    | SolidJS |
                    |  Start  |
                    +---------+
                    SSR + API routes
                         |
                    Cloudflare / Vercel
```

### Stack choices

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | SolidJS (SolidStart) | Fast, small bundle, SSR built-in, good for content sites |
| Database | SQLite via Turso (libSQL) | Edge-compatible, zero-ops, free tier generous |
| ORM | Drizzle | Type-safe, lightweight, SQLite-native |
| Styling | UnoCSS or Tailwind | Utility-first, small footprint |
| Deployment | Cloudflare Pages | Free, edge SSR, cron triggers via Workers |
| Translation API | Tinker OpenAI-compatible endpoint | Already working, tested |
| Fetching | Standard fetch / node built-ins | No curl-impersonate needed for any source |

## Database schema

```sql
CREATE TABLE sources (
  id TEXT PRIMARY KEY,           -- 'goal', 'fifa', 'sky'
  name TEXT NOT NULL,
  last_fetched_at TEXT,
  article_count INTEGER DEFAULT 0
);

CREATE TABLE articles (
  id TEXT PRIMARY KEY,            -- source-specific ID (blt_id, contentful_id, article_id)
  source_id TEXT NOT NULL REFERENCES sources(id),
  url TEXT NOT NULL UNIQUE,
  title_en TEXT NOT NULL,
  body_en TEXT NOT NULL,          -- full article body (plain text or light HTML)
  author TEXT,
  published_at TEXT NOT NULL,
  category TEXT,                  -- 'premier-league', 'world-cup', 'transfers', etc.
  tags TEXT,                      -- JSON array of tags
  image_url TEXT,
  image_width INTEGER,             -- OG image dimensions (from source)
  image_height INTEGER,
  og_description_en TEXT,          -- short description for OG cards
  fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
  word_count INTEGER
);

CREATE TABLE translations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id TEXT NOT NULL REFERENCES articles(id),
  title_tvl TEXT,
  og_description_tvl TEXT,        -- translated OG description for share cards
  body_tvl TEXT,                  -- translated body (paragraph-aligned)
  model_path TEXT NOT NULL,       -- tinker:// path used for translation
  model_id TEXT,                  -- tinker model UUID for tracking
  translated_at TEXT NOT NULL DEFAULT (datetime('now')),
  paragraph_count INTEGER,
  failed_paragraphs INTEGER DEFAULT 0,
  is_collapsed BOOLEAN DEFAULT 0, -- model collapse detected
  collapse_score REAL,            -- 0.0 (clean) to 1.0 (fully collapsed)
  attempt_number INTEGER DEFAULT 1 -- which retry attempt was used
);

CREATE TABLE fetch_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL,
  fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
  articles_found INTEGER,
  articles_new INTEGER,
  errors TEXT
);

-- Indexes for the frontend
CREATE INDEX idx_articles_published ON articles(published_at DESC);
CREATE INDEX idx_articles_source ON articles(source_id, published_at DESC);
CREATE INDEX idx_articles_category ON articles(category, published_at DESC);
CREATE INDEX idx_translations_article ON translations(article_id);
```

## Scraper modules — verified extraction recipes (2026-03-06)

All three sources serve content to standard HTTP clients — no curl-impersonate
required (unlike JW.org). Standard `requests`/`httpx` or our Docker curl wrapper all
work. No Cloudflare, no CAPTCHA, no rate limiting detected on any source.

---

### 1. Goal.com

#### 1.1 Discovery — sitemaps

| Sitemap | URL | Content | Volume |
|---------|-----|---------|--------|
| News articles | `https://www.goal.com/en-us/sitemap/editorial-news.xml` | `/en-us/news/{slug}/{blt_id}` | ~400+ |
| List/slide articles | `https://www.goal.com/en-us/sitemap/editorial-slides.xml` | `/en-us/lists/{slug}/{blt_id}` | ~5,000+ |

38 regional sitemaps exist (change `en-us` to other locales). Article IDs start with
`blt` (Contentstack Bynder IDs).

#### 1.2 Framework

Next.js SSR. Full article data embedded in `<script id="__NEXT_DATA__" type="application/json">`.
No client-side rendering needed — all data available on first fetch.

#### 1.3 Extraction paths — news articles (`/news/`)

| Field | JSON path in `__NEXT_DATA__` |
|-------|------------------------------|
| Route pattern | `page` = `/news/[slug]/[id]` |
| Article ID | `props.pageProps.content.article.id` |
| Headline | `props.pageProps.content.article.headline` |
| Teaser/summary | `props.pageProps.content.article.teaser` |
| Body HTML | `props.pageProps.content.article.body.body` |
| Published date | `props.pageProps.content.article.publishTime` (ISO 8601) |
| Author name | `props.pageProps.content.article.author[0].name` |
| Author slug | `props.pageProps.content.article.author[0].link.slug` |
| Hero image URL | `props.pageProps.content.article.poster.image.src` |
| Hero image alt | `props.pageProps.content.article.poster.image.alt` |
| Hero image width | `props.pageProps.content.article.poster.image.width` |
| Hero image height | `props.pageProps.content.article.poster.image.height` |
| Hero image credit | `props.pageProps.content.article.poster.credit` |
| Hero image source | `props.pageProps.content.article.poster.source` |
| Tags | `props.pageProps.content.article.tagList.tags[]` — each has `.name`, `.link.pageType` (TEAM/COMPETITION/CATEGORY), `.link.slug` |
| OG image URL | `props.pageProps.page.meta.openGraph.image.url` |
| OG title | `props.pageProps.page.meta.openGraph.title` |
| OG description | `props.pageProps.page.meta.openGraph.description` |
| SEO canonical | `props.pageProps.page.meta.seo.canonicalUrl` |
| Targeting | `props.pageProps.page.data.targeting` — `parentCategoryName`, `childCategoryName`, `primaryTagName`, `secondaryTagName`, `pageSubType` ("News"), `firstPublishTime` |

Skip-worthy keys under `content`: `mostRead`, `editorsPicks`, `recirculationElements`,
`bettingActivation`, `arenaBlogSchema`.

#### 1.4 Extraction paths — list/slide articles (`/lists/`)

Lists use `content.slideList` instead of `content.article`.

| Field | JSON path in `__NEXT_DATA__` |
|-------|------------------------------|
| Route pattern | `page` = `/lists/[slug]/[id]` |
| Article metadata | `props.pageProps.content.slideList.article` — same sub-keys as news (`id`, `headline`, `teaser`, `publishTime`, `poster`, `author`, `tagList`, `body`) |
| Intro body HTML | `props.pageProps.content.slideList.article.body.body` |
| Slides array | `props.pageProps.content.slideList.slides[]` |
| Slide headline | `slides[i].headline` |
| Slide body HTML | `slides[i].body.body` |
| Slide image URL | `slides[i].media.image.src` |
| Slide image alt | `slides[i].media.image.alt` |
| Slide image dims | `slides[i].media.image.width`, `.height` |
| Slide image credit | `slides[i].media.credit` |

Full text for lists = intro body + all `slides[].body.body` concatenated. Each slide
also has its own image.

| Aspect | `/news/` | `/lists/` |
|--------|----------|-----------|
| Next.js route | `/news/[slug]/[id]` | `/lists/[slug]/[id]` |
| Content key | `content.article` | `content.slideList` |
| Article metadata | Direct under `content.article` | Under `content.slideList.article` |
| Body structure | Single `body.body` HTML string | Intro `body.body` + `slides[].body.body` array |
| Extra images | Only inline `<picture>` in body | Each slide has its own `media.image` |
| `pageSubType` | `"News"` | `"Slide List"` |
| Sitemap | `editorial-news.xml` | `editorial-slides.xml` |

#### 1.5 Image CDN

**Domain:** `assets.goal.com`

URL pattern:
```
https://assets.goal.com/images/v3/{getty-ID}/crop/{base64-crop-params}/{filename}.jpg
```

Query-string resizing:
- `?width=1400&upscale=true` — JSON-LD schema variant
- `?auto=webp&format=pjpg&width=640&quality=60` — Next.js srcSet variant
- `?format=webp` or `?format=jpg` — body HTML variant
- Observed width breakpoints in srcSet: 640, 750, 828, 1080, 1200, 1920, 2048, 3840

Some hero images are PNGs (GOAL-branded composite graphics):
```
https://assets.goal.com/images/v3/{blt-id}/GOAL%20-%20Multiple%20Images%20...png
```

Team crest images use a different CDN: `cdn.sportfeeds.io` (match embeds only, not articles).

#### 1.6 Inline images in body HTML

```html
<picture>
  <source srcset="...?format=webp" type="image/webp">
  <source srcset="...?format=jpg" type="image/jpeg">
  <img src="...?format=jpg"
       alt="descriptive alt text"
       data-source="Getty Images"
       data-copyright="AFP"
       data-portal-copyright="AFP"
       perform:prop="uuid:getty-1251845036;width:1467;height:2200">
</picture>
```

Key `<img>` attributes: `alt`, `data-source`, `data-copyright`, `perform:prop`
(contains getty UUID + original dimensions).

#### 1.7 Body HTML cleanup

Strip these embedded scripts before text extraction:
- `<script type="application/json" data-placement="footballco-bet-betsense">` — betting widget
- `<script type="application/json" data-placement="fcplayer-semantic">` — video embed
- `<script type="application/json" data-placement="mobile-advert">` — ad placeholder
- `<script type="application/json" data-placement="match-schedule">` — match widget

#### 1.8 OG meta tags (for our site's sharing)

From the HTML `<head>` (all have `data-next-head=""` attribute):
```html
<meta property="og:type" content="article">
<meta property="og:title" content="...">
<meta property="og:description" content="...">
<meta property="og:image" content="https://assets.goal.com/images/v3/...">
<meta property="og:image:secure_url" content="https://assets.goal.com/images/v3/...">
<meta property="og:image:alt" content="...">
<meta property="og:image:height" content="1238">
<meta property="og:image:width" content="2200">
```

`og:image` URL = same as `poster.image.src` (no query params). JSON-LD schema adds
`?width=1400&upscale=true`.

#### 1.9 JSON-LD schema

Path: `props.pageProps.page.schema.jsonld[0].dangerouslySetInnerHTML.__html`
(HTML-escaped JSON string). Standard `NewsArticle` with `@type`, `headline`,
`description`, `datePublished`, `dateModified`, `author`, `image`.

#### 1.10 Scraper config

```
Sitemaps:  editorial-news.xml, editorial-slides.xml
Method:    Fetch sitemap XML -> extract URLs -> fetch each page ->
           parse __NEXT_DATA__ JSON -> extract article content
Schedule:  Every 30 minutes
Rate:      1 req/sec (polite)
Anti-bot:  None detected. No Cloudflare, no CAPTCHA, no rate limiting
robots.txt: User-agent: * / Allow: / -- fully permissive
```

---

### 2. FIFA.com

#### 2.1 Architecture overview

React SPA. `www.fifa.com` returns an empty shell — all content loaded via open API.
Two-step fetch process: URL slug -> Page API -> entry ID -> Article Section API.

**API base URL:** `https://cxm-api.fifa.com/fifaplusweb/api`

No authentication required for any endpoint. Standard curl works.

#### 2.2 Discovery — sitemaps

```
GET https://cxm-api.fifa.com/fifaplusweb/api/sitemaps/articles/{page}
```

- 100 articles per page, pages numbered 0-105
- **~10,579 articles total** (105 full pages + 79 on last page)
- Ordered by `lastmod` descending (newest first)
- Returns XML: `<url><loc>...</loc><lastmod>...</lastmod></url>`
- Other sitemaps: `pages`, `videos`, `news`, `movies`

#### 2.3 Two-step article fetch

**Step 1: Page API** — resolve URL slug to Contentful entry ID:

```
GET /pages/{relativeUrl}
```

Example:
```
GET /pages/en/tournaments/mens/worldcup/canadamexicousa2026/articles/harry-kane-stats-quotes-records
```

Returns:

| Field | Description |
|-------|-------------|
| `pageId` | Contentful entry ID (e.g., `2IN6FvV5BCm7US1rqS18qD`) |
| `template` | Always `articleTemplate` for articles |
| `meta.title` | OG title |
| `meta.description` | OG description |
| `meta.image` | OG image URL (digitalhub CDN with focuspoint) |
| `relativeUrl` | Canonical path |
| `relativeUrlsSEO` | Localized URLs for all languages (de, en, es, fr, it, ja, ko, pt) |
| `tags[]` | Semantic tags with source/category/id |
| `sections[]` | Array of section objects: `entryId`, `entryType`, `entryEndpoint` |

The first section where `entryType == "article"` contains the article content endpoint.
The `pageId` always matches the article `entryId`.

**Step 2: Article Section API** — fetch actual content:

```
GET /sections/article/{entryId}?locale=en
```

Example:
```
GET /sections/article/2IN6FvV5BCm7US1rqS18qD?locale=en
```

#### 2.4 Article response schema

| Field | Type | Description |
|-------|------|-------------|
| `entryType` | string | `"article"` |
| `entryId` | string | Contentful ID |
| `articleTitle` | string | `"26 Superstars: Harry Kane"` |
| `articlePublishedDate` | ISO 8601 | `"2026-03-06T00:00:00+00:00"` |
| `heroImage` | object | `{entryId, title, src, alt, caption, width, height}` |
| `heroVideoEntryId` | string | Present instead of heroImage for video articles |
| `richtext` | object | Contentful Rich Text Document (see below) |
| `articlePreviewText` | string | Short description/excerpt |
| `previewImage` | object | `{entryId, title, src, alt, caption, width, height}` |
| `publishedLabel` | string | `"Published"` (UI label) |
| `authorLabel` | string | `"Author"` (UI label — **no actual author data**) |
| `theme` | string | `"Dark"` |
| `customTheme` | object | Color palette |

**No dedicated author field** exists. The `authorLabel` is a UI localization string only.

#### 2.5 Body content — Contentful Rich Text JSON

The `richtext` field is a standard Contentful Rich Text Document:

```json
{
  "nodeType": "document",
  "data": {},
  "content": [
    {
      "nodeType": "paragraph",
      "data": {},
      "content": [
        { "nodeType": "text", "value": "...", "marks": [], "data": {} },
        { "nodeType": "hyperlink", "data": { "uri": "https://..." }, "content": [...] }
      ]
    },
    { "nodeType": "heading-2", "content": [...] },
    { "nodeType": "hr" },
    { "nodeType": "unordered-list", "content": [...] },
    {
      "nodeType": "embedded-entry-block",
      "data": {
        "target": { /* inline Contentful entry with full data */ }
      }
    }
  ]
}
```

Node types observed:
- `paragraph`, `heading-2`, `heading-3`, `heading-4`
- `hr` (horizontal rule)
- `unordered-list`, `hyperlink` (inline links)
- `embedded-entry-block` with content types: `image`, `video`, `socialMediaPost`,
  `promotionalExternalLink`, `externalIntegrationEmbed`, `fdcpTournamentRelatedSection`

Text marks: `bold`, `italic`

**Text extraction recipe:** Walk the tree recursively, collect all `nodeType: "text"` →
`value` strings. Skip `embedded-entry-block` nodes (images, videos, widgets).

#### 2.6 Image CDN

**Domain:** `digitalhub.fifa.com`

Two URL patterns:

1. **Transform URLs** (resizable):
```
https://digitalhub.fifa.com/transform/{uuid}/{name}
  ?io=transform:fill,width:800,height:450
  &quality=75
```

Resize params:
- `io=transform:fill,width:{W},height:{H}` — fill mode (crop to fit)
- `io=transform:fit,width:{W}` — fit mode (preserve aspect ratio)
- `io=transform:fill,aspectratio:16x9,width:800` — aspect ratio mode
- `quality=75` (or `best`/`auto`/`eco`/`low`)
- `focuspoint=0.54,0.02` — crop focus point

2. **Direct URLs** (fixed variants):
```
https://digitalhub.fifa.com/m/{hash}/{variant}-{name}.{ext}
```
Variants: `original`, `webimage`, `mini`, `thul` (thumbnail)

Bynder image entries (in rich text) include `src`, `transformBaseUrl`, `original`,
and `thumbnails` with multiple pre-rendered sizes.

#### 2.7 Search API

```
GET https://cxm-api.fifa.com/fifacxmsearch/api/results
  ?locale=en
  &searchString=query
  &size=10
  &type=search
  &clientType=fifaplus
  &context=default
```

**Required header:** `X-Functions-Key: 2kD9zRYRT7xN6kSGs6EoHcvSyKOyK0B4YaKTf1Ygeaw8PM6bgfR6SQ==`

(Hardcoded in SPA shell at `window["fp.env"].SEARCH_KEY`)

| `clientType` | Returns content from |
|-------------|---------------------|
| `fifaplus` | www.fifa.com (articles + videos) |
| `fifacom` | inside.fifa.com (news + media releases) |

Search indexed: **4,772 articles** + 11,505 videos for `fifaplus`.

Result `_source` fields: `id`, `title`, `description`, `url`, `image` (`src`,
`focalPosX/Y`, `width`, `height`, `alt`), `recordType`, `contentDate`,
`semanticTags`, `locale`.

#### 2.8 URL slug → entry ID mapping

1. Take sitemap URL: `https://www.fifa.com/en/tournaments/.../articles/harry-kane-stats-quotes-records`
2. Extract path: `/en/tournaments/.../articles/harry-kane-stats-quotes-records`
3. Call Page API: `GET /pages/en/tournaments/.../articles/harry-kane-stats-quotes-records`
4. From response, get `sections[0].entryId` (where `entryType == "article"`)
5. Fetch article: `GET /sections/article/{entryId}?locale=en`

The `pageId` = `entryId` in all cases tested — can skip Step 2 if you have the pageId.

#### 2.9 OG-relevant fields

From **Page API**: `meta.title`, `meta.description`, `meta.image`

From **Article Section API**: `articleTitle`, `articlePreviewText`, `heroImage.src`,
`previewImage.src`, `articlePublishedDate`

#### 2.10 Scraper config

```
Sitemaps:  /sitemaps/articles/{0-105} (XML, 100/page)
Method:    Fetch sitemap -> extract URLs -> Page API -> Article Section API
           -> parse Contentful Rich Text JSON
Schedule:  Every 30 minutes
Rate:      2 req/sec
Anti-bot:  Kasada + Akamai on www.fifa.com, but API has none
robots.txt: Only blocks /*?archive?filters= -- articles fully allowed
Locales:   en, ar, zh, fr, de, hi, id, it, ja, ko, pt, es
```

---

### 3. Sky Sports

#### 3.1 Discovery — sitemaps

| Sitemap | URL | Content |
|---------|-----|---------|
| News (recent) | `https://www.skysports.com/sitemap/sitemap-news.xml` | ~61 articles across all sports, refreshed every 2-3 days |
| Sitemap index | `https://www.skysports.com/sitemap-index.xml` | Links to monthly image sitemaps back to April 2019 |

**Football filtering** — two reliable methods (both identify same set):
- URL path contains `/football/` (e.g., `/football/news/11095/...`)
- `<news:keywords>` value is `soccer` (not "football")

~25 of ~61 articles in each news sitemap refresh are football (41%).

#### 3.2 URL structure

```
https://www.skysports.com/football/news/{section_id}/{article_id}/{slug}
```

Section IDs observed:

| Section ID | Topic |
|-----------|-------|
| `11095` | General / Premier League |
| `11661` | Tottenham |
| `11667` | Manchester United |
| `11688` | Championship |
| `11735` | Millwall |
| `11781` | Scottish football |
| `12709` | Transfer talk |
| `36621` | Scottish Premiership |

#### 3.3 Extraction — JSON-LD (primary method)

Each article page has 3-4 `<script type="application/ld+json">` blocks. The
`NewsArticle` block contains:

| Field | Type | Description |
|-------|------|-------------|
| `@type` | string | `"NewsArticle"` |
| `headline` | string | Full headline |
| `alternativeHeadline` | string | Shorter/social headline |
| `description` | string | Usually empty (use OG description instead) |
| `articleBody` | string | **Full article as plain text** (e.g., 8,277 chars / 1,462 words) |
| `wordCount` | string | Integer as string (e.g., `"1462"`) |
| `genre` | string | `"soccer"` for football |
| `inLanguage` | string | `"en-GB"` |
| `datePublished` | ISO 8601 | `"2026-03-05T13:00:00+0000"` |
| `dateModified` | ISO 8601 | |
| `dateCreated` | ISO 8601 | |
| `dateline` | string | e.g., `"London,UK"` (not always present) |
| `author` | object | `{@type: "Person", name, sameAs (X/Twitter URL), url}` |
| `image` | object | `{@type: "ImageObject", url (2048x1152), width: 2048, height: 1152}` |
| `publisher` | object | Sky Sports Organization with logo |
| `mainEntityOfPage` | string | Relative URL path |

Additional JSON-LD blocks: `Organization` (Sky Sports), `WebSite`, `VideoObject`
(one per embedded video, with `thumbnailUrl` at 150x150, 768x432, 800x600, plus
`duration` and `embedUrl`).

**`articleBody` is the easiest extraction path** — full article as clean plain text,
no HTML parsing needed.

#### 3.4 Extraction — HTML body (for structured paragraphs)

```html
<div class="sdc-article-body sdc-article-body--lead"
     data-component-name="ui-article-body"
     data-testid="article-body"
     data-highlight-intro="true">
  <p>"I see my record of 41 goals in 29 games... Wow!"</p>
  <p>Even Robert Lewandowski finds it hard to comprehend...</p>
  <!-- inline images, videos, factboxes interspersed -->
</div>
```

Selector: `div[data-testid="article-body"] > p`

Paragraphs are plain `<p>` tags (no `data-pid`). Inline emphasis uses `<em>`,
strong uses `<strong>`. Inline images, videos, factboxes, Outbrain ads interspersed
between paragraphs.

#### 3.5 OG meta tags

```html
<meta property="og:site_name" content="Sky Sports">
<meta property="og:locale" content="en_GB">
<meta property="og:title" content="[full headline]">
<meta property="og:description" content="[actual description -- better than empty JSON-LD description]">
<meta property="og:url" content="https://www.skysports.com/football/news/...">
<meta property="og:type" content="article">
<meta property="og:image" content="https://e0.365dm.com/26/03/1600x900/skysports-..._NNNNNNN.jpg?YYYYMMDDHHMMSS">
```

**`og:image` uses 1600x900** (not the 2048x1152 from JSON-LD). Excellent for OG sharing
(exceeds the 1200x630 minimum). No Twitter card meta tags found.

#### 3.6 Image CDN

**Domain:** `e0.365dm.com` (articles/editorial), `e2.365dm.com` (team badges)

URL pattern:
```
https://e0.365dm.com/{YY}/{MM}/{WIDTHxHEIGHT}/{filename}_{id}.jpg?{timestamp}
```

**Resolution variants** (swap the `WIDTHxHEIGHT` segment):

| Resolution | Aspect | Use case | Status |
|-----------|--------|----------|--------|
| `150x150` | 1:1 | Thumbnail | OK |
| `320x180` | 16:9 | Small mobile | OK |
| `384x216` | 16:9 | srcset 380w | OK |
| `640x380` | ~16:9 | Mid mobile | OK |
| `768x432` | 16:9 | Default src | OK |
| `800x600` | 4:3 | Legacy | OK |
| `1280x720` | 16:9 | HD | OK |
| `1600x900` | 16:9 | OG image | OK |
| `1920x1080` | 16:9 | Full HD | OK |
| `2048x1152` | 16:9 | Max (JSON-LD) | OK |

Not available: `1024x576`, `1200x675` (404).

For our OG images: use `1600x900` variant (swap resolution in URL).

#### 3.7 Hero image HTML structure

```html
<div class="sdc-article-widget sdc-article-image" data-testid="article-image">
  <figure class="sdc-article-image__figure">
    <div class="sdc-article-image__wrapper" data-aspect-ratio="16/9">
      <img class="sdc-article-image__item"
           loading="lazy"
           intrinsicsize="768x432"
           src="https://e0.365dm.com/26/03/768x432/skysports-...jpg?..."
           srcset="...384x216/...jpg 380w,
                   ...768x432/...jpg 760w,
                   ...1600x900/...jpg 1024w,
                   ...2048x1152/...jpg 2048w"
           sizes="(min-width: 1024px) 1024px, 100vw"
           alt="[descriptive alt text]"
           data-testid="article-image-image">
    </div>
    <figcaption class="ui-media-caption" data-testid="article-image-caption">
      <span class="u-hide-visually">Image:</span>
      <span class="ui-media-caption__caption-text" data-testid="article-image-caption-text">
        [caption text]
      </span>
    </figcaption>
  </figure>
</div>
```

Selector for hero + inline images: `img[data-testid="article-image-image"]`

All inline article images use the identical HTML structure.

#### 3.8 Scraper config

```
Sitemap:   https://www.skysports.com/sitemap/sitemap-news.xml
Method:    Fetch sitemap -> filter by /football/ or keywords=soccer ->
           fetch each page -> extract JSON-LD articleBody
Schedule:  Every 30 minutes
Rate:      1 req/sec (polite)
Anti-bot:  None detected
robots.txt: Blocks GPTBot, CCBot, AhrefsBot, Yandexbot
           Generic User-agent * is allowed on /football/news/
```

---

### Summary — extraction cheat sheet

| Source | Volume | Text extraction | Hero image | OG image |
|--------|--------|----------------|------------|----------|
| Goal.com | ~5,500 | `__NEXT_DATA__` → `content.article.body.body` (HTML) | `content.article.poster.image.src` on `assets.goal.com` | Same as poster URL |
| FIFA.com | ~10,600 | 2-step API → `richtext` (Contentful Rich Text JSON) | `heroImage.src` on `digitalhub.fifa.com` | `meta.image` from Page API |
| Sky Sports | ~60/refresh | JSON-LD `articleBody` (plain text) | `img[data-testid="article-image-image"]` on `e0.365dm.com` | `og:image` at 1600x900 |

All images are CDN-hosted, support multiple resolutions, and are suitable for
OpenGraph sharing (1200x630+ available from all three).

## Translation pipeline

### Endpoint config

```
Base URL:   https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
Model:      tinker://a6453cc0-d0d8-5168-996a-c9b9ee3b8582:train:0/sampler_weights/final
Endpoint:   POST /completions
Auth:       Authorization: Bearer $TINKER_API_KEY
Stop:       ["\n\nUser:"]
Max tokens: 512
Temperature: 0.0
```

### Translation strategy

1. Split article body into paragraphs (by `\n\n` or `<p>` boundaries)
2. Translate each paragraph independently (stays within 2048 token training window)
3. Use the training prompt template:

```
System: You are a careful translator between Tuvaluan and English. Translate
faithfully. Preserve names, numbers, punctuation, line breaks, and structure
when possible. Output only the translation.

User: Convert this English text to natural Tuvaluan while keeping the original
structure when possible.

{paragraph}

[the model completes the assistant turn with the translation]
```

4. Reassemble translated paragraphs into full article
5. Store in `translations` table linked to the original article
6. On failure (timeout, empty response), mark paragraph as failed and retry later

### Rate management

Tinker's OpenAI-compatible endpoint is beta with variable throughput. Be conservative:

- Translate sequentially (1 paragraph at a time)
- 500ms delay between requests
- Retry with exponential backoff on 429/500
- Queue new articles, translate in background — don't block scraping
- Target: ~100 articles/hour (assuming ~10 paragraphs each, 1 req/sec)

### Title translation

Translate the title as a single short paragraph. Store separately in
`translations.title_tvl` for display in listings and headings.

## SolidStart app structure

```
src/
  routes/
    index.tsx                    # homepage — latest translated articles
    articles/
      [id].tsx                   # single article view (EN + TVL side by side)
    category/
      [slug].tsx                 # category listing (premier-league, transfers, etc.)
    api/
      cron/
        fetch.ts                 # API route triggered by cron — runs scrapers
        translate.ts             # API route triggered by cron — runs translation queue
      articles/
        index.ts                 # GET /api/articles — paginated article list
        [id].ts                  # GET /api/articles/:id — single article + translation
  components/
    ArticleCard.tsx              # card for article listings
    ArticleBody.tsx              # tap-to-reveal EN/TVL paragraph view
    LanguageToggle.tsx           # switch between TVL-only, EN-only, stacked
    Header.tsx                   # site header with navigation
    CategoryNav.tsx              # horizontal scroll category pills
    LoadMore.tsx                 # infinite scroll / load more button
    OGMeta.tsx                   # OpenGraph + Twitter Card meta tags per page
    ShareButton.tsx              # native Web Share API (WhatsApp/Facebook)
    IslandSelector.tsx           # first-visit island picker (9 islands + diaspora)
  lib/
    db.ts                        # Drizzle client + schema
    scrapers/
      goal.ts                    # Goal.com scraper module
      fifa.ts                    # FIFA.com CXM API scraper
      sky.ts                     # Sky Sports scraper
      common.ts                  # shared: sitemap parser, text cleaner, rate limiter
    translator.ts                # Tinker translation client
    categories.ts                # category mapping (tags → normalized categories)
  db/
    schema.ts                    # Drizzle schema definitions
    migrations/                  # SQL migration files

app.config.ts                    # SolidStart config
drizzle.config.ts                # Drizzle config (Turso connection)
```

## Page designs — mobile-first

All designs are for a 360px-wide viewport (typical Android phone in Tuvalu).
Desktop is a progressive enhancement, not the primary target. 97%+ of Tuvaluan
internet access is mobile. Outer island users may be on 3G with high latency.

### Design constraints

- **Touch targets**: minimum 48x48px (thumb-friendly)
- **Font size**: 16px minimum body text (no pinch-to-zoom needed)
- **Images**: lazy-loaded, compressed, with `loading="lazy"` and width/height hints
  to prevent layout shift on slow connections
- **No hover states**: everything is tap/swipe
- **Offline support**: Service Worker caches last 20 articles for offline reading
- **Data budget**: target <100KB initial page load (excluding images).
  Tuvaluan mobile data is expensive and often metered
- **No hamburger menus**: all navigation visible. Categories are a horizontal
  scroll strip — familiar from every social media app

### Homepage (`/`)

```
+--------------------------------+
| TALAFUTIPOLO          [TV|EN]  |
+--------------------------------+
| PL | Transfers | WC | All  →  |  ← horizontal scroll
+--------------------------------+
|                                |
| +----------------------------+ |
| | ┌────────────────────────┐ | |
| | │      [hero image]      │ | |
| | └────────────────────────┘ | |
| | Kane ne lakau ki mua mo    | |
| | Bayern ne malosi Dortmund  | |
| |                            | |
| | Kane scores hat-trick as   | |  ← smaller, muted EN subtitle
| | Bayern crush Dortmund      | |
| |                            | |
| | Goal.com · 2h              | |
| +----------------------------+ |
|                                |
| +----------------------------+ |
| | ┌──────────┐               | |
| | │  [thumb] │ Liverpool ne  | |
| | │          │ faka...       | |
| | └──────────┘               | |
| | Liverpool extend lead...   | |
| | Sky Sports · 3h            | |
| +----------------------------+ |
|                                |
| +----------------------------+ |
| | ┌──────────┐               | |
| | │  [thumb] │ OFC ne faka.. | |
| | │          │               | |
| | └──────────┘               | |
| | OFC qualifiers draw...     | |
| | FIFA.com · 5h              | |
| +----------------------------+ |
|                                |
| [  Faitau atu  ▼  ]           |  ← "Read more" in Tuvaluan
|                                |
+--------------------------------+
| Te Fatele: 1,247 improved      |  ← sticky footer teaser
+--------------------------------+
```

- **TVL title first, EN subtitle second** (muted color). This is a Tuvaluan site.
- Lead article: full-width hero image + stacked titles
- Remaining articles: thumbnail left, text right (like WhatsApp/Facebook feed — familiar pattern)
- Category bar: horizontal scroll strip, pill-shaped buttons, no wrapping
- Infinite scroll with "Faitau atu" (read more) button as fallback
- Sticky footer teaser showing community contribution count — tapping opens the fatele page
- Source + relative time on every card ("2h" not "2 hours ago" — saves space)

### Article page (`/articles/[id]`)

Default view is **TVL-only** (not side-by-side — that's a desktop luxury).
Tap a paragraph to reveal the English underneath. This is the key mobile
interaction: read in Tuvaluan, tap to peek at English when confused.

```
+--------------------------------+
| ←  TALAFUTIPOLO       [TV|EN]  |
+--------------------------------+
|                                |
| Kane ne lakau ki mua mo te    |
| tolu o koleni i te taimi ne   |
| malosi ei a Bayern Munich     |
| i Borussia Dortmund           |
|                                |
| Goal.com · 6 Mati 2026        |
|                                |
| ┌────────────────────────────┐ |
| │        [hero image]        │ |
| └────────────────────────────┘ |
|                                |
| Ko Harry Kane ne lakau ki     |
| mua mo te tolu o koleni i     |
| te taimi ne malosi ei a       |
| Bayern Munich 5-1 i Borussia  |
| Dortmund i te po nei.    [?]  |  ← flag button (right edge)
|                                |
|   ┌──────────────────────┐    |  ← revealed on tap
|   │ Harry Kane scored a  │    |
|   │ stunning hat-trick   │    |
|   │ as Bayern Munich     │    |
|   │ demolished Borussia  │    |
|   │ Dortmund 5-1 tonight │    |
|   └──────────────────────┘    |
|                                |
| Te kapeteni o Egelani ne      |
| taa te kolo muamua i te       |
| minute 12...             [?]  |
|                                |
| ... (more paragraphs)         |
|                                |
+--------------------------------+
| Source: goal.com/en-us/...     |
+--------------------------------+
```

- **TVL-first by default.** The reader is here to read Tuvaluan.
- **Tap-to-reveal English**: tap any TVL paragraph → EN slides out below it
  in a slightly indented, lighter-colored block. Tap again to collapse.
  This is natural, low-friction, and teaches the reader that EN is always
  available as a safety net.
- **Flag button** `[?]` on right edge of each paragraph — single tap to
  report "this sounds wrong." Small, unobtrusive, but always reachable
  with the right thumb.
- **Tap-to-reveal generates implicit signal**: if a reader taps to see EN,
  that paragraph likely has a quality issue. Free RL data.
- Back arrow top-left returns to feed (not browser back — avoids losing scroll position)

### Language modes (3-way toggle)

The `[TV|EN]` toggle in the header cycles through three modes:

| Mode | What shows | When to use |
|------|-----------|-------------|
| **TV** (default) | Tuvaluan only, tap-to-reveal EN | Normal reading |
| **EN** | English only | Reader wants the original |
| **TV+EN** | Alternating paragraphs (TVL then EN) | Learning/comparison mode |

On a phone, side-by-side columns don't work. TV+EN mode stacks paragraphs:

```
| Ko Harry Kane ne lakau ki     |  ← TVL (normal weight, dark)
| mua mo te tolu o koleni...    |
|                                |
|   Harry Kane scored a          |  ← EN (lighter weight, indented, muted)
|   stunning hat-trick as...     |
|                                |
| Te kapeteni o Egelani ne      |  ← TVL
| taa te kolo muamua...         |
|                                |
|   The England captain          |  ← EN
|   opened the scoring...        |
```

Visual distinction between TVL and EN: TVL is full-weight dark text, EN is
lighter weight + slightly indented + muted color. No boxes or borders needed.

### A/B preference screen (interstitial)

Appears between articles (never mid-article). Full-screen takeover:

```
+--------------------------------+
|                                |
|  Taki e ako!                   |
|  (Taki is learning!)           |
|                                |
|  Fea e sili atu?               |
|  (Which is better?)            |
|                                |
| +----------------------------+ |
| | Ne fakailoa ne te          | |
| | kapeteni o Egelani te      | |
| | savali muamua i te minute  | |
| | 12 o te taime...           | |
| |                     [ A ]  | |
| +----------------------------+ |
|                                |
| +----------------------------+ |
| | Ko te kapeteni o Egelani   | |
| | ne taa te kolo muamua i    | |
| | te minute 12...            | |
| |                     [ B ]  | |
| +----------------------------+ |
|                                |
| [  Fano  →  ]                  |  ← "Skip" in Tuvaluan
+--------------------------------+
```

- Two translation variants, large readable text
- Big tap targets for A and B
- Always skippable — "Fano" (skip/go) at the bottom
- Limit to 1 per session. Never nag.

### Name guardian prompt (inline)

Appears inline when a name mismatch is detected:

```
+--------------------------------+
|                                |
| ...ne malosi ei a Bayern       |
| Munich i Borussia Dortmund.    |
|                                |
| ┌──────────────────────────┐  |
| │ Taki ne tusi "Hariʹkein" │  |
| │ Taki wrote "Hariʹkein"   │  |
| │                          │  |
| │ E tonu?                  │  |
| │ (Is it correct?)         │  |
| │                          │  |
| │ [Io ✓]    [→ Harry Kane] │  |
| └──────────────────────────┘  |
|                                |
| Te kapeteni o Egelani ne...    |
```

- Inline card, not a modal or popup
- Two big buttons: "Io" (yes, it's fine) or the correct name from the source
- The correct name is pre-filled from the EN source text (NER comparison)
- Dismisses with a single tap either way

### Island selector (first visit only)

On first visit, before any content loads:

```
+--------------------------------+
|                                |
|  Talofa!                       |
|                                |
|  Ko koe mai fea?               |
|  (Where are you from?)         |
|                                |
|  +-----------+ +-----------+  |
|  | Funafuti  | | Vaitupu   |  |
|  +-----------+ +-----------+  |
|  | Nanumea   | | Nui       |  |
|  +-----------+ +-----------+  |
|  | Nukufetau | | Niutao    |  |
|  +-----------+ +-----------+  |
|  | Nanumaga  | | Nukulaelae|  |
|  +-----------+ +-----------+  |
|  | Niulakita |               |
|  +-----------+               |
|                                |
|  [  I fafo (Diaspora)  ]      |
|                                |
|  [  Fano → (Skip)  ]          |
|                                |
+--------------------------------+
```

- Simple grid of 9 islands + diaspora option + skip
- Stored in localStorage (no account, no cookies to accept)
- Only shown once. Can be changed in a minimal settings page
- This single choice powers the island contribution meter

### Te Fatele page (`/fatele`)

Accessible from the sticky footer teaser. Shows community progress:

```
+--------------------------------+
| ←  Te Fatele o te Gagana       |
+--------------------------------+
|                                |
|  Tatou ne fakalelei 1,247      |
|  fakataloga i te masina nei    |
|  (We improved 1,247            |
|   translations this month)     |
|                                |
|  Funafuti   ████████░░  312    |
|  Vaitupu    ██████░░░░  254    |
|  Nanumea    █████░░░░░  198    |
|  Nui        ████░░░░░░  147    |
|  Nukufetau  ███░░░░░░░  121    |
|  Niutao     ██░░░░░░░░   89   |
|  Nanumaga   ██░░░░░░░░   64   |
|  Nukulaelae █░░░░░░░░░   42   |
|  Niulakita  █░░░░░░░░░   20   |
|  I fafo     ████░░░░░░  143   |
|                                |
|  Tufuga o te Gagana:           |
|  (Language experts)            |
|  · [Name] — 47 corrections    |
|  · [Name] — 31 corrections    |
|  · [Name] — 28 corrections    |
|                                |
+--------------------------------+
```

### OpenGraph and social sharing

This is critical. Tuvaluans share links via **Facebook** (~5,100 users, ~50%
of the population) and **WhatsApp** (primary messaging). When someone shares
an article link in a WhatsApp group or on Facebook, the preview card is the
first impression. It must look good, load instantly, and be in Tuvaluan.

Both Facebook and WhatsApp use OpenGraph meta tags to generate link previews.
Twitter/X uses its own `twitter:card` tags but falls back to OG. All preview
cards are generated server-side in the `<head>` — no JS needed.

#### Meta tags per article page

```html
<!-- Primary OG tags -->
<meta property="og:type" content="article" />
<meta property="og:url" content="https://talafutipolo.tv/articles/{id}" />
<meta property="og:title" content="{title_tvl}" />
<meta property="og:description" content="{og_description_tvl}" />
<meta property="og:image" content="{image_url}" />
<meta property="og:image:width" content="{image_width}" />
<meta property="og:image:height" content="{image_height}" />
<meta property="og:locale" content="tvl" />
<meta property="og:locale:alternate" content="en" />
<meta property="og:site_name" content="Talafutipolo Tuvalu" />

<!-- Article-specific OG tags -->
<meta property="article:published_time" content="{published_at}" />
<meta property="article:section" content="{category}" />
<meta property="article:tag" content="{tag}" />  <!-- one per tag -->

<!-- Twitter Card (fallback for X/Twitter) -->
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="{title_tvl}" />
<meta name="twitter:description" content="{og_description_tvl}" />
<meta name="twitter:image" content="{image_url}" />

<!-- Standard HTML meta -->
<title>{title_tvl} — Talafutipolo</title>
<meta name="description" content="{og_description_tvl}" />
<link rel="canonical" href="https://talafutipolo.tv/articles/{id}" />
```

Key points:
- **Title and description are in Tuvaluan.** When a link is shared in a
  WhatsApp group, the preview shows Tuvaluan text. This is the whole point.
- `og:locale` is `tvl` (ISO 639-3 for Tuvaluan). Facebook may not render a
  locale label for it, but it's semantically correct.
- `og:image` uses the original source image URL directly (hotlinked from
  Goal.com/FIFA.com/Sky Sports). This avoids storing/proxying images and
  keeps our data budget tiny. If a source starts blocking hotlinks, fall
  back to a generic Talafutipolo branded card.
- `og:image:width` and `og:image:height` are stored in the DB at scrape
  time. Facebook uses these to decide whether to show a large or small
  preview — without them, Facebook fetches the image to measure it, adding
  latency to the first share.

#### Meta tags for homepage

```html
<meta property="og:type" content="website" />
<meta property="og:url" content="https://talafutipolo.tv/" />
<meta property="og:title" content="Talafutipolo Tuvalu — Tala Futipolo i te Gagana Tuvalu" />
<meta property="og:description" content="Tala futipolo mai te lalolagi i te gagana Tuvalu.
Football news from around the world in the Tuvaluan language." />
<meta property="og:image" content="https://talafutipolo.tv/og-default.png" />
<meta property="og:image:width" content="1200" />
<meta property="og:image:height" content="630" />
```

The default OG image (`og-default.png`) should be a 1200x630 branded card
with the site name and a football motif. This is the one image we host
ourselves. Keep it <50KB.

#### OGMeta component

```typescript
// src/components/OGMeta.tsx
import { Meta, Title } from "@solidjs/meta";

interface OGMetaProps {
  title: string;             // Tuvaluan title
  description: string;       // Tuvaluan description
  url: string;
  image?: string;
  imageWidth?: number;
  imageHeight?: number;
  publishedAt?: string;
  category?: string;
  tags?: string[];
  type?: "article" | "website";
}

export function OGMeta(props: OGMetaProps) {
  return (
    <>
      <Title>{props.title} — Talafutipolo</Title>
      <Meta name="description" content={props.description} />
      <Meta property="og:type" content={props.type ?? "article"} />
      <Meta property="og:url" content={props.url} />
      <Meta property="og:title" content={props.title} />
      <Meta property="og:description" content={props.description} />
      <Meta property="og:site_name" content="Talafutipolo Tuvalu" />
      <Meta property="og:locale" content="tvl" />
      <Meta property="og:image"
            content={props.image ?? "https://talafutipolo.tv/og-default.png"} />
      {props.imageWidth &&
        <Meta property="og:image:width" content={String(props.imageWidth)} />}
      {props.imageHeight &&
        <Meta property="og:image:height" content={String(props.imageHeight)} />}
      <Meta name="twitter:card" content="summary_large_image" />
      <Meta name="twitter:title" content={props.title} />
      <Meta name="twitter:description" content={props.description} />
      <Meta name="twitter:image"
            content={props.image ?? "https://talafutipolo.tv/og-default.png"} />
      {props.publishedAt &&
        <Meta property="article:published_time" content={props.publishedAt} />}
      {props.category &&
        <Meta property="article:section" content={props.category} />}
      {props.tags?.map(tag =>
        <Meta property="article:tag" content={tag} />
      )}
    </>
  );
}
```

#### Share button — Web Share API

The site should have a prominent share button on every article. Use the
Web Share API, which is supported on all mobile browsers and opens the
native share sheet (WhatsApp, Facebook, Messenger, SMS, etc.):

```typescript
// src/components/ShareButton.tsx
interface ShareButtonProps {
  title: string;    // Tuvaluan title
  url: string;
  text?: string;    // Tuvaluan description
}

export function ShareButton(props: ShareButtonProps) {
  const canShare = () => typeof navigator !== "undefined" && !!navigator.share;

  async function handleShare() {
    if (navigator.share) {
      await navigator.share({
        title: props.title,
        text: props.text,
        url: props.url,
      });
    } else {
      // Fallback: copy link to clipboard
      await navigator.clipboard.writeText(props.url);
      // Show brief "Copied!" toast
    }
  }

  return (
    <button
      onClick={handleShare}
      class="share-btn"
      aria-label="Fakasoa (Share)"
    >
      Fakasoa ↗
    </button>
  );
}
```

- **"Fakasoa"** = Tuvaluan for "share/distribute"
- Button is large (48px+ height), thumb-reachable, at the bottom of the article
- Web Share API opens the native OS share sheet — the user picks WhatsApp,
  Facebook, Messenger, SMS, or whatever they use. No need to build integrations
  for each platform.
- Fallback for desktop: copy URL to clipboard
- The share text is in Tuvaluan — when someone shares via WhatsApp, the
  message body is in Tuvaluan before the link

#### What the shared link looks like

When someone shares `https://talafutipolo.tv/articles/xyz` in a WhatsApp
group, WhatsApp fetches the OG tags and renders:

```
+------------------------------------------+
| ┌──────────────────────────────────────┐ |
| │          [hero image from            │ |
| │           Goal.com/FIFA]             │ |
| └──────────────────────────────────────┘ |
| talafutipolo.tv                          |
| Kane ne lakau ki mua mo te tolu o        |
| koleni i te taimi ne malosi ei a         |
| Bayern Munich i Borussia Dortmund        |
+------------------------------------------+
```

Tuvaluan title. Tuvaluan description. Source image. The link preview
becomes a mini advertisement for Tuvaluan-language football news.

#### Sharing as RL signal

When a user taps "Fakasoa", that's a strong positive signal — they thought
the article (and its translation) was worth sharing. Log this:

```sql
INSERT INTO implicit_signals (article_id, signal_type, session_id)
VALUES (?, 'share', ?);
```

Articles with high share counts likely have good translations. Articles
that get read but never shared may have quality issues. This is free
reward signal for the RL pipeline.

#### Facebook Sharing Debugger

After launch, verify OG tags render correctly by testing article URLs in:
- Facebook Sharing Debugger: `https://developers.facebook.com/tools/debug/`
- Twitter Card Validator: `https://cards-dev.twitter.com/validator`
- WhatsApp: just send a link to yourself in a chat

Facebook caches OG data aggressively. If we update an article's translation
(from community corrections), we can bust the cache by calling:
```
POST https://graph.facebook.com/?id={url}&scrape=true
```

### Performance budget

| Metric | Target | Why |
|--------|--------|-----|
| First Contentful Paint | <1.5s on 3G | Outer islands are 3G |
| Largest Contentful Paint | <2.5s on 3G | Hero image lazy-loads |
| Total page weight | <100KB (no images) | Metered mobile data |
| JS bundle | <30KB gzipped | SolidJS is already tiny (~7KB) |
| Image format | WebP with JPEG fallback | Smaller than PNG/JPEG |
| Image size | Max 400px wide | No phone screen is wider |
| Fonts | System fonts only | Zero font download cost |
| Service Worker | Cache last 20 articles | Offline reading on boats/outer islands |

### Technical implications

```typescript
// app.config.ts — SolidStart config
export default defineConfig({
  server: {
    preset: "cloudflare-pages",  // edge SSR
  },
  // No client-side routing for initial load speed.
  // Use MPA mode — each page is server-rendered HTML.
  // SolidJS hydrates only interactive elements (toggle, flag button).
});
```

- **MPA over SPA**: Server-render each page as full HTML. Client JS only
  hydrates interactive bits (language toggle, flag button, tap-to-reveal).
  This means the page is readable before any JS loads — critical on 3G.
- **System fonts**: `font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif`.
  Zero network cost. Every phone has these.
- **No client-side routing**: Page transitions are full navigations. On 3G,
  a server-rendered HTML page loads faster than fetching JSON + rendering
  client-side. SolidStart SSR handles this.
- **Preload next article**: When reading an article, `<link rel="prefetch">`
  the next article in the feed. Feels instant on tap.
- **Touch gestures**: Swipe left/right on article page to navigate between
  articles in the feed. No need to go back to the homepage.

## Cron schedule

| Job | Schedule | What it does |
|-----|----------|-------------|
| `fetch` | Every 30 min | Runs all 3 scrapers, inserts new articles into DB |
| `translate` | Every 15 min | Picks untranslated articles from queue, translates via Tinker |

On Cloudflare Workers, use [Cron Triggers](https://developers.cloudflare.com/workers/configuration/cron-triggers/).
On Vercel, use [Vercel Cron Jobs](https://vercel.com/docs/cron-jobs).

## Category mapping

Normalize source-specific tags into consistent categories:

```typescript
const CATEGORIES: Record<string, string[]> = {
  'premier-league': ['Premier League', 'EPL', 'English Premier League'],
  'champions-league': ['Champions League', 'UCL', 'UEFA Champions League'],
  'world-cup': ['World Cup', 'FIFA World Cup', 'WC Qualifying', 'OFC'],
  'transfers': ['Transfer', 'Transfers', 'Transfer News', 'Rumour'],
  'pacific': ['OFC', 'Oceania', 'Pacific Games'],
  'la-liga': ['La Liga', 'Spanish Football'],
  'bundesliga': ['Bundesliga', 'German Football'],
  'serie-a': ['Serie A', 'Italian Football'],
};
```

## Development phases

### Phase 1 — MVP (week 1-2) DONE

- [x] SolidStart project scaffolding with SQLite (local dev, Turso deferred)
- [x] Goal.com scraper (sitemap + `__NEXT_DATA__` extraction)
- [x] FIFA.com scraper (CXM API two-step fetch)
- [x] Sky Sports scraper (JSON-LD extraction)
- [x] Translation pipeline (Tinker `/completions` endpoint)
- [x] Homepage with article cards (TVL title + EN subtitle)
- [x] Article page with bilingual paragraphs + language toggle (TV/EN/TV+EN)
- [x] Category mapping and filter navigation (pills)
- [x] Mobile-responsive layout
- [x] Article hero images from sources
- [x] Source attribution and linking
- [ ] Deploy to Cloudflare Pages (deferred)

### Phase 2 — community RL features (2026-03-07) DONE

- [x] Fix HTML entities in scrapers (`html.unescape()`) + one-time DB migration
- [x] Real pagination on homepage and category pages (query-param based)
- [x] 404 page ("Seki kitea") with Tuvaluan text
- [x] Paragraph flag `[?]` button on each TVL paragraph (fires POST /api/feedback)
- [x] Implicit signal tracking: reveal, share, flag (fires POST /api/signal)
- [x] Island selector modal on first visit (9 islands + diaspora, localStorage)
- [x] Te Fatele community dashboard (`/fatele`) with island progress bars
- [x] Sticky footer teaser linking to `/fatele` with live signal count
- [x] API routes: POST /api/feedback, POST /api/signal, GET /api/stats
- [x] `feedback` and `implicit_signals` DB tables with indexes

### Phase 2.5 — model collapse detection (2026-03-07) DONE

- [x] `detect_collapse.py`: multi-signal n-gram collapse detector (whole-text, tail,
  sliding window, per-paragraph, repeated phrase)
- [x] `translate_football.py`: 3-attempt retry with temperature escalation (0.0→0.3→0.7)
- [x] All translation attempts saved to `translation_attempts` table for RL training
- [x] Model ID (`a6453cc0`) tracked on every translation
- [x] `db.ts`: CASE WHEN hides collapsed translations (site shows English-only)
- [x] `migrate_collapse_detection.py`: retroactive scan — 41/56 flagged, 0 false positives
- [x] Root cause analysis documented in `todo.md` #10
- [x] `--retry-collapsed` CLI flag to retry previously flagged translations

### Phase 3 — translation quality (next)

- [ ] Fix Sky Sports paragraph splitting (no `<p>` tags — use `<br>` / sentence splitting)
- [ ] Increase MAX_TOKENS from 512 to 1024+ (primary collapse driver)
- [ ] Sentence-level chunking fallback for paragraphs >100 words
- [ ] Domain-specific prompt: football glossary terms as loanwords
- [ ] A/B preference interstitial (needs variant generation infrastructure)
- [ ] Name guardian prompt (needs NER comparison)

### Phase 4 — polish

- [ ] Search functionality
- [ ] RSS feed output (in Tuvaluan)
- [ ] Service Worker / offline support (outer islands have intermittent connectivity)
- [ ] Tufuga o te Gagana (Language Experts) elevated contributor role

### Phase 5 — growth

- [ ] Deploy (Cloudflare Pages + Turso / Vercel)
- [ ] Automated cron jobs (fetch every 30min, translate every 15min)
- [ ] Add more sources (Guardian if licensed, ESPN if policy changes)
- [ ] Glossary system for football terms (loanwords vs translated terms)
- [ ] Retrain Stage A adapter with name-preservation augmentation

## Known issues (2026-03-07)

### Translation quality: model collapse (MITIGATED)

73% of translations (41/56) exhibit model collapse — degenerate repetitive loops.
Detection and hiding is implemented; collapsed articles show English-only on the site.

**Root causes (see `todo.md` #10 for full analysis):**
1. **Sky Sports single-paragraph bodies (72% collapse rate)** — no `<p>` tags in HTML,
   so entire 500-1800 word body sent as one chunk. Model exceeds MAX_TOKENS=512 and loops.
2. **Out-of-domain content** — football jargon, TV guides, pop culture are maximally far
   from JW.org training data. FIFA articles (formal/informational) collapse at only 15%.
3. **MAX_TOKENS=512 too low** — can't finish translating even moderate paragraphs.

**Implemented:** `detect_collapse.py` (6-signal detector), 3-attempt retry with
temperature escalation, all attempts recorded for RL training, site hides collapsed.
**Remaining:** Fix paragraph splitting, increase MAX_TOKENS, add sentence chunking.

### Feedback FK constraint

The `feedback` table has a `REFERENCES articles(id)` foreign key constraint.
The `implicit_signals` table does not (by design — signals may reference
articles that get deleted). The feedback API will return 500 if called with
a non-existent article_id. This is correct behavior for production but means
testing with fake IDs requires using a real article ID from the DB.

## Environment variables

```env
TURSO_DATABASE_URL=libsql://your-db.turso.io
TURSO_AUTH_TOKEN=your-turso-token
TINKER_API_KEY=your-tinker-key
TINKER_MODEL_PATH=tinker://a6453cc0-d0d8-5168-996a-c9b9ee3b8582:train:0/sampler_weights/final
FETCH_INTERVAL_MINUTES=30
TRANSLATE_INTERVAL_MINUTES=15
```

## Gamified RL data collection

### Think like a Tuvaluan

Tuvalu: ~11,000 people, ~7,100 internet users, ~5,100 on Facebook, 97%+ mobile-first.
The entire country is smaller than a mid-size high school. Everyone knows everyone.
Culture is collective, not individualistic — the family unit (kaaiga) and the island
community come before the individual. Respect (fakaalofa, faka'apa'apa) governs
social interaction. The fatele (communal song-dance) is the cultural metaphor: everyone
participates, everyone contributes, the whole is greater than its parts.

A gamification system that works here cannot look like Silicon Valley engagement
farming. No individual leaderboards, no "Top Contributor" badges, no streak anxiety.
Instead: communal progress, island pride, language preservation as an act of care.

### What RL data we actually need

| Data type | What it trains | User effort | Value |
|-----------|---------------|-------------|-------|
| **A/B preference** | DPO/RLHF reward model | Low (tap one of two) | High |
| **Paragraph flag** | Negative reward signal | Minimal (tap = "sounds wrong") | Medium |
| **Text correction** | SFT fine-tuning data | High (typing required) | Very high |
| **Name correction** | Fixes hallucination problem | Low (tap + confirm) | High |
| **"Sounds natural" confirm** | Positive reward signal | Minimal (keep scrolling = implicit positive) | Medium |

### Design: Te Fatele o te Gagana (The Language Fatele)

Frame the entire system as a communal fatele — the community is teaching
the machine to speak Tuvaluan properly. The machine has a name: **Taki**
(Tuvaluan: "to lead/guide"). Taki is learning. Taki needs help. The community
is Taki's teacher.

#### 1. Tap-to-flag while reading (zero friction)

As a reader scrolls through a translated article, any paragraph that sounds
unnatural gets a single tap. That's it. No typing, no forms, no explanation
needed.

```
+------------------------------------------+
|  Ko Harry Kane ne lakau ki mua mo te     |
|  tolu o koleni i te taimi ne malosi ei   |
|  a Bayern Munich i Borussia Dortmund...  |
|                                    [ ? ] |  ← tap if this sounds wrong
+------------------------------------------+
```

This generates paragraph-level negative signal. Unflagged paragraphs that
get read (scroll-past = implicit engagement) are weak positive signal.

**RL value**: Builds a reward model. Paragraphs flagged by multiple readers
become high-priority candidates for re-translation or correction.

#### 2. "Which sounds better?" — A/B preferences

Show two translations of the same paragraph. Reader taps the one that sounds
more natural. This is the highest-value RL signal (direct preference pairs
for DPO training).

Surface this as an interstitial — after reading 3-4 articles, show one
comparison. Keep it to one per session. Never interrupt reading flow.

```
+------------------------------------------+
|  Taki e ako: fea e sili atu?             |
|  (Taki is learning: which is better?)    |
|                                          |
|  A)  Ne fakailoa ne te kapeteni o        |
|      Egelani te savali muamua...         |
|                                          |
|  B)  Ko te kapeteni o Egelani ne         |
|      taa te kolo muamua...               |
|                                          |
|        [ A ]          [ B ]              |
+------------------------------------------+
```

**RL value**: Direct DPO preference pairs. Generate the B variant by
re-translating at temperature=0.7 or using a different checkpoint.

#### 3. Island contribution meter (communal, not competitive)

Instead of individual leaderboards, show a communal progress bar per island.
Tuvalu has 9 islands — each person self-identifies their island on first
visit. Contributions (flags, preferences, corrections) are pooled by island.

```
+------------------------------------------+
|  Te Fatele o te Gagana                   |
|  Community translations improved: 1,247  |
|                                          |
|  Funafuti    ████████████░░░░  312       |
|  Vaitupu     ██████████░░░░░░  254       |
|  Nanumea     ████████░░░░░░░░  198       |
|  Nui         ██████░░░░░░░░░░  147       |
|  Nukufetau   █████░░░░░░░░░░░  121       |
|  Niutao      ████░░░░░░░░░░░░   89       |
|  Nanumaga    ███░░░░░░░░░░░░░   64       |
|  Nukulaelae  ██░░░░░░░░░░░░░░   42       |
|  Niulakita   █░░░░░░░░░░░░░░░   20       |
+------------------------------------------+
```

This taps into island identity without being adversarial. It's not "Funafuti
is beating Vaitupu" — it's "look how much our community is contributing."
The framing matters: "1,247 translations improved" not "1,247 tasks completed."

**Cultural fit**: Tuvaluans strongly identify with their home island. Diaspora
communities (NZ, Fiji, Australia) would see their island represented and feel
motivated to contribute from abroad.

#### 4. Name guardian — fix the hallucination problem directly

When the translator produces a name that doesn't match the source, highlight
it and ask the reader to confirm or correct. This directly addresses the
JW-style name hallucination:

```
+------------------------------------------+
|  Original: Harry Kane scored...          |
|  Taki wrote: Ko Hariʹkein ne lakau...   |
|                                          |
|  Taki e le iloa te igoa nei.             |
|  (Taki doesn't know this name.)         |
|                                          |
|  E tonu "Hariʹkein"?                    |
|  (Is "Hariʹkein" correct?)              |
|                                          |
|  [ Io (Yes) ]  [ Ikai → Harry Kane ]    |
+------------------------------------------+
```

Auto-detect potential name issues by comparing proper nouns between source
and translation (NER on English side, fuzzy-match against Tuvaluan output).
When they diverge, surface this lightweight correction prompt.

**RL value**: Builds a name-preservation dataset. Even 200-300 corrections
would significantly improve the adapter's name handling.

#### 5. Tufuga o te Gagana (Language Experts)

Some community members are recognized language authorities — teachers,
elders, church leaders. Give them an elevated role: **Tufuga o te Gagana**
(Language Craftsperson). They can:

- Submit full paragraph corrections (not just flags)
- Their corrections are weighted 3-5x in training data
- Their corrections go live on the site immediately
- They see a "Corrected by [name]" attribution

This is not a gamification badge — it's a recognition of existing social
authority. In Tuvaluan culture, elders and experts already hold this status.
The site simply mirrors it.

**RL value**: High-quality SFT data from trusted speakers. A single tufuga
correcting 10 paragraphs/week produces ~500 gold-standard training examples
per year.

#### 6. Weekly match challenge

Before a big PL match, feature the preview article. After the match, feature
the match report. Readers who improve translations for that article see their
island credited:

```
+------------------------------------------+
|  Liverpool vs Arsenal — Match Report     |
|                                          |
|  Translation improved by readers from    |
|  Funafuti (3), Vaitupu (2), Nanumea (1) |
+------------------------------------------+
```

This ties contributions to something people already care about — the match.
The contribution is a social act around a shared event, not an isolated task.

#### 7. Implicit signals (no user action needed)

Track reading behavior passively (with consent):

- **Time on paragraph**: Long dwell time on a TVL paragraph may indicate
  re-reading (confusion signal)
- **Language toggle**: If a user switches from TVL to EN mid-article, the
  paragraph they switched at is likely a bad translation
- **Bounce after translation**: If a user reads EN version after seeing TVL,
  the TVL wasn't sufficient
- **Share behavior**: Articles that get shared (WhatsApp, Facebook) likely
  have acceptable translations

**RL value**: Weak but high-volume signal. Useful for reward model training
when aggregated across many readers.

### Data pipeline: reader actions → RL training

```
Reader actions on site
        │
        ▼
+-------------------+
|  feedback table   |  (flags, preferences, corrections, implicit signals)
|  in Turso DB      |
+-------------------+
        │
        ▼
+-------------------+
|  Export pipeline   |  weekly batch job
|  scripts/          |
|  export_rl_data.py |
+-------------------+
        │
        ├──→  preference_pairs.jsonl   (A/B choices → DPO)
        ├──→  corrections.jsonl        (tufuga edits → SFT)
        ├──→  name_fixes.jsonl         (name corrections → augmentation)
        └──→  flagged_paragraphs.jsonl (negative signal → reward model)
        │
        ▼
+-------------------+
|  Retrain adapter  |  Stage A v2 with community data mixed in
|  via Tinker       |
+-------------------+
        │
        ▼
    Improved Taki → better translations → fewer flags → cycle continues
```

### DB schema additions

```sql
CREATE TABLE feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id TEXT NOT NULL REFERENCES articles(id),
  paragraph_idx INTEGER,           -- which paragraph (0-indexed)
  feedback_type TEXT NOT NULL,      -- 'flag', 'preference', 'correction', 'name_fix'
  value TEXT,                       -- correction text, preferred variant ('A'/'B'), or NULL for flags
  island TEXT,                      -- self-reported island identity
  is_tufuga BOOLEAN DEFAULT FALSE, -- elevated contributor
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  session_id TEXT                   -- anonymous session (no login required)
);

CREATE TABLE ab_variants (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id TEXT NOT NULL REFERENCES articles(id),
  paragraph_idx INTEGER NOT NULL,
  variant_a TEXT NOT NULL,          -- translation at temp=0.0
  variant_b TEXT NOT NULL,          -- translation at temp=0.7 or different checkpoint
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE implicit_signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id TEXT NOT NULL REFERENCES articles(id),
  paragraph_idx INTEGER,
  signal_type TEXT NOT NULL,        -- 'dwell_time', 'lang_switch', 'share', 'bounce'
  value TEXT,                       -- e.g. dwell time in ms, target language
  session_id TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_feedback_article ON feedback(article_id, paragraph_idx);
CREATE INDEX idx_feedback_type ON feedback(feedback_type);
CREATE INDEX idx_feedback_island ON feedback(island);
```

### Volume estimates

With ~7,100 internet users in Tuvalu and ~5,100 on Facebook:

| Scenario | Monthly active readers | Flags/month | Preferences/month | Corrections/month |
|----------|----------------------|-------------|-------------------|-------------------|
| Pessimistic (1% of internet users) | 70 | 200 | 50 | 10 |
| Moderate (5%) | 350 | 1,000 | 250 | 50 |
| Optimistic (15% — viral in community) | 1,000 | 3,000 | 750 | 150 |

Even the pessimistic scenario produces usable RL data within a few months.
The name correction feature alone could fix the hallucination problem with
~200 examples.

### Key principles

1. **No login required.** Anonymous session IDs. Tuvaluans won't create
   accounts. Island identity is self-reported on first visit via a simple
   "Which island are you from?" selector (9 options + diaspora).

2. **Never interrupt reading.** Flags are a single tap. A/B preferences
   appear between articles, never mid-article. Corrections are opt-in.

3. **Community framing, always.** "Our community improved 47 translations
   this week" not "You completed 47 tasks." The unit of achievement is the
   community, not the individual.

4. **Respect the tufuga.** Language experts are recognized, not gamified.
   Their social authority already exists — the site mirrors it.

5. **Mobile-first.** 97%+ access via mobile. Every interaction must work
   with a thumb on a 5" screen. No hover states, no desktop assumptions.

6. **Offline-tolerant.** Outer islands have intermittent connectivity.
   Cache articles for offline reading. Queue feedback submissions for when
   connectivity returns.

## Cost estimate

| Component | Cost |
|-----------|------|
| Cloudflare Pages + Workers | Free tier (100k req/day) |
| Turso DB | Free tier (9GB storage, 500M rows read/month) |
| Tinker API | Free during beta (check current status) |
| Domain | ~$10/year |
| **Total** | **~$10/year** |

The entire stack runs on free tiers. The only real cost is a domain name.