# Tuvaluan-English Parallel Corpus: Scraping & Dataset Plan

## 0. Project goal

Build a HuggingFace-ready Tuvaluan↔English parallel corpus from JW.org and WOL content, producing artifacts suitable for:

1. **Machine translation** fine-tuning (tvl→en, en→tvl)
2. **Multilingual LLM fine-tuning** (instruction/chat format)
3. **Continued pretraining** on Tuvaluan text (monolingual)
4. **Evaluation benchmarks** (held-out verse-aligned test sets)

Tuvaluan (ISO 639-3: `tvl`) is a Polynesian language with ~11,000 speakers. It is mutually intelligible with Tokelauan. Existing NLP resources are extremely limited — JW content represents one of the largest available sources of parallel text.

### Prior art

- **JW300**: A parallel corpus covering 300+ languages from JW.org, widely used for low-resource MT. Our work extends this specifically for Tuvaluan with richer metadata and multiple output formats.
- **Christodoulopoulos Bible Corpus**: 100-language verse-aligned Bible corpus (XML). Does not include Tuvaluan.
- **FLORES+**: Meta's MT evaluation benchmark (200+ languages). May include Tuvaluan evaluation sentences.
- **jwsoup**: Python package for scraping JW Bible content (pip installable).

For a single practical map of dataset/datacollection docs, and exact pair/split contracts for the active pipeline, see:

- [docs/DATASET_COLLECTION_AND_ML_PIPELINE.md](docs/DATASET_COLLECTION_AND_ML_PIPELINE.md)

---

## 1. Fetching: Docker curl-impersonate + BeautifulSoup

### Why curl-impersonate is required

Both `jw.org` and `wol.jw.org` perform **TLS fingerprint detection**. Standard Python HTTP clients fail:

| Client | jw.org | wol.jw.org | Failure mode |
|---|---|---|---|
| `requests` | Timeout | Timeout | Read timeout after 30s |
| `httpx` (HTTP/2) | StreamReset | Intermittent (works 1st request, blocks after) | `RemoteProtocolError: StreamReset error_code:2` |
| `curl` (system) | Exit 92 | Exit 56 | HTTP/2 stream error / recv failure |
| **Docker curl-impersonate** | **200** | **200** | **Works reliably** |

The sites reset HTTP/2 streams when the TLS handshake doesn't match a known browser fingerprint. `curl-impersonate` mimics real browser TLS/HTTP handshakes, which passes the check.

### Docker setup

We use the pre-built Firefox Docker image (runs under Rosetta on Apple Silicon):

```bash
# Image already pulled:
docker pull lwthiker/curl-impersonate:0.6-ff

# Test:
docker run --rm lwthiker/curl-impersonate:0.6-ff \
  curl_ff117 -s -o /dev/null -w "HTTP %{http_code}\n" \
  "https://www.jw.org/tvl/sitemap.xml"
# → HTTP 200
```

The platform warning (`linux/amd64 does not match linux/arm64/v8`) is expected on Apple Silicon and does not affect functionality.

### Python integration

All scripts use `scripts/fetch.py`, which shells out to Docker curl-impersonate and returns HTML to BeautifulSoup:

```python
# scripts/fetch.py — shared fetcher
import subprocess, time
from bs4 import BeautifulSoup

DOCKER_IMAGE = "lwthiker/curl-impersonate:0.6-ff"
DOCKER_WRAPPER = "curl_ff117"
DELAY = 2  # seconds between requests

def fetch(url, timeout=30, retries=3):
    """Fetch via Docker curl-impersonate. Returns HTML string or None."""
    result = subprocess.run(
        ["docker", "run", "--rm", DOCKER_IMAGE, DOCKER_WRAPPER,
         "-s", "-L", "--max-time", str(timeout), url],
        capture_output=True, text=True, timeout=timeout + 10,
    )
    return result.stdout if result.returncode == 0 else None

def fetch_soup(url, parser="html5lib"):
    html = fetch(url)
    return BeautifulSoup(html, parser) if html else None
```

Rate limiting (2s delay) and exponential-backoff retries are built in. Raw HTML is saved to `data/raw/` for offline reprocessing.

### Performance

~4 seconds per chapter pair (2 fetches + parse) with Docker curl-impersonate, vs ~25 seconds with httpx when it intermittently worked. Full Bible scrape (1,189 chapters) estimated at ~80 minutes.

### Other endpoints tested

| Endpoint | Result |
|---|---|
| `b.jw-cdn.org/apis/pub-media/GETPUBMEDIALINKS` | Tuvaluan language codes `TVL`, `VL`, `TV` all return 400/404 — Tuvaluan is not in the CDN publication API |
| `www.jw.org/en/languages/` | StreamReset — blocked like other jw.org pages |
| Sitemap hreflang alternates | None present — 0/7,103 URLs have `<xhtml:link>` alternates |

---

## 2. Artifact structure

### 2.1 Directory layout

```
tv/
├── tv2en.md                       # URL pattern reference
├── README.md                      # This file
├── pyproject.toml                 # uv project config
├── curl-imp/                      # curl-impersonate setup
│   ├── curl-impersonate/          # cloned repo
│   ├── curl-impersonate-fetch-skill/  # Claude Code skill
│   ├── claude-code-curl-impersonate-guide.md
│   └── fetch-with-curl-impersonate.sh
├── data/
│   ├── raw/                       # Raw scraped HTML (not uploaded to HF)
│   │   ├── sitemap_tvl.xml        # Full sitemap (1.1MB, 7,103 URLs)
│   │   ├── sitemap_tvl.json       # Parsed + classified sitemap
│   │   ├── wol_tvl/               # WOL Tuvaluan pages (bible_{bookNo}_{ch}.html)
│   │   └── wol_en/                # WOL English pages
│   ├── aligned/                   # Verse/paragraph-aligned pairs (JSONL) — IMMUTABLE
│   │   ├── bible_verses.jsonl     # Verse-level aligned Bible text (30,838 pairs)
│   │   ├── articles.jsonl         # Article/paragraph-level aligned text (275,430 pairs)
│   │   ├── daily_text.jsonl       # Date-keyed daily text pairs (3,432 pairs)
│   ├── cleaned/                   # Quality-cleaned data (output of clean_pipeline.py)
│   │   ├── cleaned.jsonl          # Accepted pairs (178,371 pairs)
│   │   ├── rejected.jsonl         # Rejected pairs with reasons
│   │   ├── cleaning_report.json   # Statistics and metadata
│   │   └── rejection_samples.jsonl # 10 examples per rejection category
│   ├── hf_dataset/                # HuggingFace-ready Parquet files
│   │   ├── README.md              # HF dataset card
│   │   ├── bible/
│   │   │   ├── train.parquet
│   │   │   ├── validation.parquet
│   │   │   └── test.parquet
│   │   ├── articles/
│   │   │   └── train.parquet
│   │   ├── daily_text/
│   │   │   └── train.parquet
│   │   └── all/
│   │       ├── train.parquet
│   │       ├── validation.parquet
│   │       └── test.parquet
│   └── finetune/                  # LLM fine-tuning artifact outputs
│       └── stage_a_mt/            # Stage A MT format (current)
│           ├── train_full.jsonl
│           ├── train_balanced.jsonl
│           ├── validation.jsonl
│           └── test.jsonl
├── docs/
│   ├── SCRAPING_PLAYBOOK.md       # Step-by-step data collection guide
│   ├── DATASET_COLLECTION_AND_ML_PIPELINE.md  # Schema + lineage
│   ├── TRAINING_PIPELINE.md       # Stage A/B training commands
│   ├── TVL_EN_TINKER_PLAN.md      # Strategic training plan
│   └── SELECTIVE_TRANSLATION_SPEC.md  # Selective translation rules
├── scripts/
│   ├── fetch.py                   # Docker curl-impersonate fetcher (shared)
│   ├── scrape_sitemap.py          # Parse JW sitemap, classify URLs
│   ├── scrape_bible.py            # Scrape Bible chapters (verse-aligned)
│   ├── scrape_articles.py         # Scrape WOL articles by docId + library crawl
│   ├── scrape_daily_text.py       # Scrape date-based daily text
│   ├── clean_pipeline.py          # Data cleaning (dedup, filter, normalize)
│   ├── stats.py                   # Dataset statistics and quality report
│   ├── build_stage_a_mt_data.py    # Build Stage A MT JSONL from aligned pairs
│   ├── train_stage_a_translation.py# Train Stage A translation LoRA
│   ├── eval_stage_a_translation.py # Evaluate Stage A translation
│   ├── build_stage_b_sources.py    # Build normalized English source pools
│   ├── generate_stage_b_synthetic_tvl.py # Generate synthetic TVL capability data
│   ├── build_stage_b_mix.py       # Mix source/synthetic/anchor datasets for Stage B
│   ├── train_stage_b_agent.py     # Train Stage B bilingual LoRA
│   ├── eval_stage_b_agent.py      # Evaluate Stage B
│   └── bootstrap_tinker.sh        # Bootstrap script with staged run checklist
└── logs/
    ├── scrape.log
    └── alignment.log
```

### 2.2 Raw data storage

Bible chapters are saved as raw HTML files keyed by book number and chapter:
`data/raw/wol_tvl/bible_{bookNo}_{chapter}.html`

The parsed sitemap is saved as JSON with URL classification:
`data/raw/sitemap_tvl.json`

---

## 3. Sitemap analysis (Experiment 1 — completed)

The Tuvaluan sitemap contains **7,103 URLs** (significantly more than the 1,310 initially estimated from earlier sitemap retrieval attempts that timed out and returned partial results).

### Category breakdown

| Category | Count | Notes |
|---|---|---|
| magazine | 2,489 | Largest non-Bible category |
| publication_index | 1,605 | TOC/index pages |
| bible_chapter | 1,189 | All 66 books confirmed |
| book | 499 | Book chapters/sections |
| bible_index | 290 | Bible navigation pages |
| brochure | 257 | Brochure sections |
| song | 227 | Song pages |
| video | 146 | Video pages (may have transcripts) |
| program | 95 | Convention/assembly programs |
| bible_book_toc | 66 | One per book |
| faq | 49 | Bible Q&A pages |
| news | 36 | News articles |
| bible_supplement | 34 | Appendix/supplemental material |
| about_jw | 32 | About pages |
| study_youth | 27 | Youth study content |
| misc_publication | 22 | Miscellaneous articles |
| help | 17 | Help/support pages |
| study_children | 8 | Children's content |
| study_hub | 4 | Study landing pages |
| study_science | 4 | Science topic pages |
| other | 5 | Home, search, what's new, all topics |

### Key finding: no hreflang alternates

The sitemap contains **zero** hreflang alternate links. This means we cannot use sitemap metadata for cross-language page pairing. We must rely entirely on:
- WOL docId swap (strongest)
- WOL bookNo/chapter numeric alignment (Bible)
- WOL date alignment (daily text / meetings)
- Issue code matching (magazines)
- URL pattern heuristics (everything else)

---

## 4. WOL Bible HTML structure (Experiment 2 — confirmed)

The WOL Bible page structure differs from what prior scrapers (jwsoup, crawl-for-parallel-corpora) documented for jw.org. The actual structure on WOL:

### Verse markup

```html
<article class="article bible html5 pub-nwt jwac showRuby ml-VL ms-ROMAN dir-ltr"
         data-lang="VL" dir="ltr" id="article" lang="tvl">
  <div class="scalableui">
    <header><h1>Kenese</h1></header>

    <p class="sb" data-pid="2" id="p2">
      <span class="v" id="v1-1-1-1">
        <span class="cl" data-vlid="vl0"><strong>1</strong> </span>
        I te kamataga ne faite ne te Atua a te lagi
        <a class="fn" data-fnid="1" href="...">*</a>
        mo te lalolagi.
        <a class="b" data-bid="1-1" href="...">+</a>
      </span>
    </p>

    <p class="sb" data-pid="3" id="p3">
      <span class="v" id="v1-1-2-1">
        <span class="vl">2 </span>
        A te lalolagi e seki foliga faka‵lei kae lavaki,...
      </span>
    </p>
  </div>
</article>
```

### Key selectors (WOL-specific, different from jw.org)

| Element | Selector | Purpose |
|---|---|---|
| Verse span | `span.v` | Contains one verse's text (NOT `span.verse`) |
| Verse ID | `id="v{bookNo}-{ch}-{verse}-{part}"` | Parseable verse reference |
| Verse number | `span.vl` or `span.cl > strong` | Leading number display |
| Footnote | `a.fn` | Remove before text extraction (NOT `a.footnoteLink`) |
| Cross-reference | `a.b` | Remove before text extraction (NOT `a.xrefLink`) |
| Paragraph | `p[data-pid]` | Numbered paragraph container |
| Article root | `article#article` | Main content container |
| Language tag | `article[data-lang="VL"]` | Confirms Tuvaluan content |

### Verse ID format

`v{bookNo}-{chapter}-{verse}-{part}` where:
- bookNo: 1–66 (Genesis–Revelation)
- chapter: chapter number
- verse: verse number
- part: usually 1; >1 for verses split across paragraphs (merge these)

### Extraction algorithm

```python
for v_span in soup.find_all("span", class_="v"):
    vid = v_span.get("id", "")
    m = re.match(r"v(\d+)-(\d+)-(\d+)-(\d+)", vid)
    verse_no = int(m.group(3))

    # Remove footnotes and cross-references
    for unwanted in v_span.find_all("a", class_=["fn", "b"]):
        unwanted.decompose()
    for sup in v_span.find_all("sup"):
        sup.decompose()

    text = v_span.get_text(strip=True)
    text = re.sub(r"^\d+\s*", "", text)  # strip leading verse number
```

---

## 5. Bible pilot results (Experiment 2 — completed)

Scraped 3 chapters (Genesis 1, Psalm 19, John 3) in both Tuvaluan and English via WOL.

### Results

| Chapter | TVL verses | EN verses | Aligned pairs |
|---|---|---|---|
| Genesis 1 | 31 | 31 | 31 |
| Psalm 19 | 15 | 15 | 15 |
| John 3 | 36 | 36 | 36 |
| **Total** | **82** | **82** | **82** |

**100% alignment** — every verse matched 1:1.

### Sample pairs

| ID | Tuvaluan | English | Ratio |
|---|---|---|---|
| `bible_1_1_1` | I te kamataga ne faite ne te Atua a te lagimo te lalolagi. | In the beginning God created the heavens and the earth. | 1.05 |
| `bible_19_19_0` | Ki te takitaki o te kau fai pese. Ko te pese a Tavita. | To the director. A melody of David. | 1.54 |
| `bible_43_3_16` | (John 3:16 — full verse) | (John 3:16 — full verse) | 1.19 |

### Length ratio statistics

- **Average**: 1.21 (Tuvaluan text is ~21% longer than English)
- **Range**: 0.83 – 1.64
- **All within quality threshold** (0.3 – 3.0)

This ratio is consistent with Polynesian languages, which tend to use more function words and particles than English.

### Average text length

- TVL: 144 chars/verse
- EN: 121 chars/verse

---

## 6. Output formats (HuggingFace + fine-tuning)

### 6.1 HuggingFace parallel corpus (primary artifact)

**Format**: Parquet
**Feature type**: Flat columns (not nested `Translation` feature — flat is more widely compatible and allows richer metadata)

#### Schema

| Column | Type | Description |
|---|---|---|
| `id` | `string` | Unique row identifier: `{content_type}_{alignment_key}` |
| `tvl` | `string` | Tuvaluan text |
| `en` | `string` | English text |
| `content_type` | `string` | One of: `bible_verse`, `article_paragraph`, `daily_text`, `publication_section` |
| `domain` | `string` | One of: `bible`, `magazine`, `book`, `brochure`, `daily_text`, `faq`, `news`, `song`, `meeting_workbook` |
| `alignment_method` | `string` | How the pair was aligned: `verse_number`, `doc_id`, `date`, `paragraph_position`, `pub_code` |
| `alignment_confidence` | `float32` | 0.0–1.0 confidence score |
| `doc_id` | `string` | WOL document ID (when available; strongest cross-language key) |
| `source_url_tvl` | `string` | Source URL for Tuvaluan text |
| `source_url_en` | `string` | Source URL for English text |
| `book_num` | `int32` | Bible book number 1–66 (null for non-Bible) |
| `chapter` | `int32` | Bible chapter (null for non-Bible) |
| `verse` | `int32` | Bible verse (null for non-Bible) |
| `date` | `string` | ISO date YYYY-MM-DD (for daily text; null otherwise) |
| `pub_code` | `string` | WOL publication code (e.g., `lv`, `bt`, `T-31`; null if n/a) |
| `tvl_chars` | `int32` | Character count of Tuvaluan text |
| `en_chars` | `int32` | Character count of English text |
| `length_ratio` | `float32` | `tvl_chars / en_chars` (for quality filtering) |

#### HuggingFace dataset card YAML (README.md frontmatter)

```yaml
---
language:
  - tvl
  - en
license: cc-by-4.0
task_categories:
  - translation
task_ids:
  - translation-other-tvl-en
pretty_name: "Tuvaluan-English Parallel Corpus (JW)"
size_categories:
  - 100K<n<1M
tags:
  - parallel-corpus
  - low-resource
  - polynesian
  - bible
  - machine-translation
configs:
  - config_name: bible
    data_files:
      - split: train
        path: "bible/train.parquet"
      - split: validation
        path: "bible/validation.parquet"
      - split: test
        path: "bible/test.parquet"
  - config_name: articles
    data_files:
      - split: train
        path: "articles/train.parquet"
  - config_name: daily_text
    data_files:
      - split: train
        path: "daily_text/train.parquet"
  - config_name: all
    default: true
    data_files:
      - split: train
        path: "all/train.parquet"
      - split: validation
        path: "all/validation.parquet"
      - split: test
        path: "all/test.parquet"
---
```

#### Split strategy

For a low-resource language with limited data, splits must be chosen carefully:

- **Bible config**: 80/10/10 split by book (not random verse sampling, to avoid data leakage across adjacent verses)
  - Test: held-out books (e.g., Ruth, Philemon, Jude — short books for manageable eval)
  - Validation: another set of held-out books
  - Train: remaining books
- **All config**: same book-based split for Bible portion; articles/daily_text/publications go to train only (too small to split further)
- **Articles, daily_text configs**: train only (no split — not enough data)

### 6.2 OpenAI chat completion format (for fine-tuning)

```jsonl
{"messages": [{"role": "system", "content": "You are a translator between Tuvaluan and English. Translate the following text accurately."}, {"role": "user", "content": "Translate from Tuvaluan to English:\n\nKo lagi e fakailoa te manuia o te Atua"}, {"role": "assistant", "content": "The heavens declare the glory of God"}]}
{"messages": [{"role": "system", "content": "You are a translator between Tuvaluan and English. Translate the following text accurately."}, {"role": "user", "content": "Translate from English to Tuvaluan:\n\nThe heavens declare the glory of God"}, {"role": "assistant", "content": "Ko lagi e fakailoa te manuia o te Atua"}]}
```

Both directions (tvl→en and en→tvl) are generated from each pair. System prompt is kept consistent.

### 6.3 Alpaca-style instruction format

```jsonl
{"instruction": "Translate the following Tuvaluan text to English.", "input": "Ko lagi e fakailoa te manuia o te Atua", "output": "The heavens declare the glory of God", "metadata": {"content_type": "bible_verse", "book_num": 19, "chapter": 19, "verse": 1}}
```

### 6.4 Monolingual Tuvaluan (continued pretraining)

Dolma-compatible JSONL for language model pretraining on Tuvaluan:

```jsonl
{"id": "jw-tvl-bible-19-19", "text": "Ko lagi e fakailoa te manuia o te Atua; ko te ato e fakailoa te galuega a ona lima.", "source": "jw.org", "added": "2026-03-06", "metadata": {"lang": "tvl", "domain": "bible", "content_type": "bible_chapter", "url": "https://www.jw.org/tvl/tusi/tusi-tapu/nwt/tusi/salamo/19/"}}
```

For monolingual, text is at the chapter or article level (not verse level) to provide more context per document.

---

## 7. Scraping strategy

### 7.1 Scraping priority order

Content types are prioritized by alignment confidence and volume. Estimated pairs updated based on sitemap analysis (7,103 URLs):

| Priority | Content type | Alignment method | Est. pairs | Confidence |
|---|---|---|---|---|
| 1 | Bible chapters (1,189 ch) | verse number (bookNo + chapter + verse) | ~25,000 verses | Very high |
| 2 | WOL articles by docId | docId swap | ~500–2,000 articles | Very high |
| 3 | Daily text | date alignment | ~365/year × N years | Very high |
| 4 | Magazines (2,489 pages) | issue code + article position | ~1,000–2,000 articles | Medium-high |
| 5 | Books (499 pages) | pubCode + chapter / docId | ~200–500 sections | High |
| 6 | Brochures (257 pages) | title slug / docId | ~100–300 sections | Medium |
| 7 | FAQ/study (92 pages) | docId / slug match | ~50–100 pages | Medium |
| 8 | Songs (227 pages) | song number | ~200 songs | High (lyrics) |
| 9 | News (36 pages) | date + region match | ~30 articles | Medium |
| 10 | Help (17 pages) | path match | ~15 pages | Medium |

### 7.2 Bible scraping (Priority 1)

**Source**: WOL Bible pages (numeric bookNo URLs avoid slug mapping)
**Script**: `scripts/scrape_bible.py`

```bash
uv run python scripts/scrape_bible.py --pilot    # 3 chapters
uv run python scripts/scrape_bible.py --book 1    # Genesis only
uv run python scripts/scrape_bible.py --full      # all 66 books
```

**URL pattern**:
- TVL: `wol.jw.org/tvl/wol/b/r153/lp-vl/nwt/{bookNo}/{chapter}`
- EN: `wol.jw.org/en/wol/b/r1/lp-e/nwt/{bookNo}/{chapter}`

**Resumable**: tracks completed chapter IDs in the output JSONL; skips already-scraped chapters on restart.

### 7.3 WOL articles (Priority 2)

**Source**: WOL document pages by docId
**Script**: `scripts/scrape_articles.py`

```bash
uv run python scripts/scrape_articles.py --pub lv                         # single publication
uv run python scripts/scrape_articles.py --library-cat "faleleoleo-maluga" # library category
uv run python scripts/scrape_articles.py --library                         # all categories
uv run python scripts/scrape_articles.py --docids 1102008070 1102015820   # specific docIds
```

**DocId harvesting sources** (two methods):
1. Publication TOC pages: `wol.jw.org/tvl/wol/publication/r153/lp-vl/{pubCode}` — harvests docIds from a single publication's table of contents
2. Recursive library crawl: `wol.jw.org/tvl/wol/library/r153/lp-vl/tusi-katoa/{category}` — follows sub-page links up to 5 levels deep across 6 categories (Watchtower, Awake!, Meeting workbooks, Books, Ministry, Brochures)

**URL pattern**:
- TVL: `wol.jw.org/tvl/wol/d/r153/lp-vl/{docId}`
- EN: `wol.jw.org/en/wol/d/r1/lp-e/{docId}`

### 7.4 Daily text (Priority 3)

**Source**: WOL daily text pages
**Script**: `scripts/scrape_daily_text.py`

```bash
uv run python scripts/scrape_daily_text.py --year 2025
uv run python scripts/scrape_daily_text.py --range 2017-01-01 2025-12-31
```

**URL pattern** (month and day are NOT zero-padded):
- TVL: `wol.jw.org/tvl/wol/h/r153/lp-vl/{yyyy}/{m}/{d}`
- EN: `wol.jw.org/en/wol/h/r1/lp-e/{yyyy}/{m}/{d}`

**Date range**: Tuvaluan daily texts exist from 2017-01-01 onward. Pre-2017 pages return empty content.

### 7.5 Rate limiting and politeness

- **Delay**: 2 seconds between requests (built into `fetch.py`)
- **Respect robots.txt**: Avoid disallowed paths (`/choose-language`, query params like `contentLanguageFilter`)
- **Cache**: Raw HTML saved locally — never re-fetch a page already on disk
- **Resume**: Progress tracked by checking existing output IDs

---

## 8. Alignment strategy

### 8.1 Alignment levels

| Level | Use case | Method |
|---|---|---|
| Verse | Bible text | Book number + chapter + verse number |
| Paragraph | Articles, publications | Position within document (paragraph index) |
| Document | Short articles, FAQ | Whole document as one unit |
| Date | Daily text, meetings | Calendar date |

### 8.2 Paragraph alignment for articles

For articles aligned by docId, paragraph-level alignment uses position matching:

1. Extract ordered paragraphs from both language versions
2. If paragraph counts match: align 1:1
3. If counts differ: use length-ratio heuristics and fall back to document-level alignment
4. Flag mismatched documents for manual review

### 8.3 Quality filtering

Quality filtering happens in two layers:

**Layer 1: Scrape-time filters** (built into `scrape_articles.py`):
- Metadata paragraph detection (chapter headers, copyright, photo credits)
- Both sides < 20 chars → skip
- Extreme ratios (<0.15 or >7.0) → skip

**Layer 2: Cleaning pipeline** (`scripts/clean_pipeline.py`):

The cleaning pipeline reads immutable raw data from `data/aligned/` and writes cleaned data to `data/cleaned/`. It applies these filters in order:

| Filter | Balanced | Strict | Rationale |
|---|---|---|---|
| Duplicate ID | Remove | Remove | Overlapping library crawls produce tripled records |
| Duplicate content | Hash-based | Hash-based | Same text appearing under different IDs |
| Empty text | Drop | Drop | Broken scrape or missing translation |
| Metadata | Strip | Strip | Picture captions, photo credits, copyright, headers |
| Identical pair | Drop TVL==EN | Drop TVL==EN | Untranslated content (no training signal) |
| Min length | Both < 10 chars | Both < 20 chars | Too short to be useful |
| Max length | >8,192 chars | >4,096 chars | Practical limit for most models |
| Length ratio | 0.2–5.0 | 0.3–3.0 | Misalignment signal |
| Bible ratio | 0.4–2.5 | 0.5–2.0 | Tighter bounds for verse-aligned data |
| Truncated daily | Drop | Drop | May 2025 TVL has only theme, missing commentary |

Run with: `uv run python scripts/clean_pipeline.py [--profile balanced|strict|lenient] [--dry-run]`

### 8.4 Alignment confidence scoring

Each pair gets a confidence score based on:

- `1.0` — Verse-aligned Bible text, date-aligned daily text
- `0.9` — DocId-matched article with matching paragraph count
- `0.8` — DocId-matched article with paragraph count mismatch (position-aligned)
- `0.7` — PubCode-matched publication section
- `0.6` — Issue code + article position match
- `0.5` — Title/slug match
- `0.4` — Sitemap/hreflang match only

---

## 9. Quality considerations

### 9.1 Domain bias

JW content is predominantly religious text. This creates a well-known domain bias:

- **Register**: Formal, instructional, ecclesiastical
- **Vocabulary**: Religious terminology overrepresented; everyday/colloquial Tuvaluan underrepresented
- **Sentence structure**: Translation-influenced (translated from English originals; may not reflect natural Tuvaluan syntax)

**Mitigations documented in metadata**:
- The `domain` column allows downstream users to filter by content type
- The `content_type` column distinguishes Bible from magazine from FAQ etc.
- Users should be warned in the dataset card that this is religious-domain data

### 9.2 Translation directionality

Most JW content is originally written in English and translated into Tuvaluan. This means:
- The Tuvaluan text may exhibit "translationese" (English-influenced syntax)
- For training tvl→en models, this is fine (translationese source → natural target)
- For training en→tvl models, the "natural" Tuvaluan target may itself be unnatural

### 9.3 Deduplication

Religious texts contain significant repetition (cross-references, repeated phrases, liturgical formulas). Deduplication strategy:
- **Exact dedup**: Hash-based on concatenated `tvl+en` text
- **Near dedup**: MinHash/LSH for detecting paraphrased duplicates
- **Cross-split dedup**: Ensure test/validation pairs don't appear in training (critical for Bible verse dedup — adjacent verses may share content)

---

## 10. Experiments log

### Experiment 1: Sitemap enumeration — DONE

**Goal**: Parse `jw.org/tvl/sitemap.xml` and enumerate all Tuvaluan URLs.

**Method**: Fetched sitemap via Docker curl-impersonate (1.1MB XML), parsed all `<loc>` entries, classified by URL pattern using regex classifiers from `tv2en.md` section 10.

**Result**: **7,103 URLs** classified into 25 categories. No hreflang alternates present. See section 3 above for full breakdown.

**Output**: `data/raw/sitemap_tvl.xml`, `data/raw/sitemap_tvl.json`

---

### Experiment 2: Bible verse extraction (pilot) — DONE

**Goal**: Scrape 3 Bible chapters (Genesis 1, Psalm 19, John 3) in both Tuvaluan and English. Verify verse-level alignment works.

**Method**: Fetched WOL Bible pages via Docker curl-impersonate, parsed with BeautifulSoup (html5lib), extracted verses via `span.v` elements, aligned by verse number.

**Result**: **82/82 verses aligned** (100%). TVL/EN length ratio avg 1.21 (range 0.83–1.64). See section 5 above for details.

**Output**: `data/aligned/bible_verses.jsonl` (82 pairs)

---

### Experiment 3: Full Bible scrape — DONE

**Goal**: Scrape entire NWT Bible (all 66 books, 1,189 chapters) in both languages.

**Result**: **30,838 verse pairs** across all 66 books, 1,189 chapters. 0 failures. 2 transient DNS errors recovered via retry. 5 extreme ratios (up to 4.814) from Polynesian verse boundary differences — affects 0.016% of pairs.

**Output**: `data/aligned/bible_verses.jsonl`

---

### Experiment 4: Full article scrape — DONE

**Goal**: Harvest docIds from all Tuvaluan publications on WOL, scrape articles in both languages with paragraph alignment.

**Method**: Two-stage approach — (1) discover docIds by crawling WOL library pages recursively, then (2) fetch both language versions and align by `data-pid`.

**DocId sources**:
- 22 direct publication codes: `bh`, `bhs`, `bm`, `bt`, `fg`, `gf`, `hf`, `jl`, `jy`, `kr`, `lff`, `lffi`, `lmd`, `lv`, `lvs`, `my`, `sjj`, `snnw`, `th`, `wt`, `yc`, `ypq`
- 6 WOL library categories crawled recursively (Watchtower, Awake!, Meeting workbooks, Books, Ministry, Brochures)
- Total: **7,255 unique docIds** scraped

**Result**: **275,430 article paragraph pairs** (raw, before deduplication). Paragraph-level alignment by `data-pid` with document-level fallback for mismatched structures. Contains ~129k duplicates from overlapping library category crawls, removed during dataset build.

**Quality filtering applied at scrape time**:
- Removed chapter headers ("CHAPTER N" / "MATAUPU E N")
- Removed copyright/credits/metadata paragraphs
- Removed pairs with both sides < 20 chars
- Removed pairs with extreme ratios (<0.15 or >7.0)

**Output**: `data/aligned/articles.jsonl`

See [docs/SCRAPING_PLAYBOOK.md](docs/SCRAPING_PLAYBOOK.md) for the full step-by-step reproduction guide.

---

### Experiment 5: Daily text scrape — DONE

**Goal**: Scrape daily text in both languages for all available years.

**Date range discovery**: Tuvaluan daily texts exist from **2017-01-01** onward (verified by probing each year 2014–2021; pre-2017 pages return empty content).

**Result**: **3,432 daily text pairs** covering **3,287 unique dates** from 2017-01-01 through 2025-12-31. 0 failures, 0 gaps. Each page returns ~3 consecutive days; script optimizes by extracting adjacent dates from the same fetch.

**Output**: `data/aligned/daily_text.jsonl`

---

### Experiment 6: Data cleaning — DONE

**Goal**: Deduplicate, filter, and normalize the raw aligned data into a clean corpus ready for dataset assembly and training.

**Method**: Built `scripts/clean_pipeline.py` — a reusable pipeline that reads immutable data from `data/aligned/` and writes cleaned output to `data/cleaned/`. Pipeline stages: text normalization (NFC, strip invisible chars, fix HTML entities), ID deduplication, content hash deduplication, metadata filtering, identical-pair removal, length/ratio filtering.

**Result**: **178,371 accepted pairs** (57.6% acceptance rate) from 309,700 raw input. Removed 118,553 duplicate IDs (from overlapping library crawls), 11,075 duplicate content pairs, 1,049 metadata records, 406 identical TVL==EN pairs, and 246 bad ratios. Total: ~96.1M chars, ~25.3M tokens.

**Output**: `data/cleaned/cleaned.jsonl`, `data/cleaned/rejected.jsonl`, `data/cleaned/cleaning_report.json`

---

### Experiment 7: Dataset assembly — PENDING

**Goal**: Combine cleaned data into HuggingFace Parquet files with proper splits, metadata, and dataset card.

**Depends on**: Experiment 6 (completed). Cleaned data: 178,371 pairs in `data/cleaned/cleaned.jsonl`.

**Next steps**: Build train/validation/test splits, assemble HuggingFace Parquet files, write dataset card.

---

## 11. Tools and dependencies

### Python (managed with uv)

```toml
# pyproject.toml
[project]
dependencies = [
    "beautifulsoup4",
    "html5lib",
    "httpx[http2]",  # kept as fallback; primary fetching via Docker
    "lxml",
    "requests",      # kept as fallback
    "tqdm",
]
```

```bash
uv add pandas pyarrow datasets lingua-py  # add when needed for dataset assembly
```

### System dependencies

- **Docker Desktop** — required for curl-impersonate
- **Docker image**: `lwthiker/curl-impersonate:0.6-ff` (Firefox 117 fingerprint, linux/amd64 via Rosetta on Apple Silicon)
- **Python 3.14** via Homebrew

---

## 12. Dataset size (actual, as of March 2026)

### Raw aligned data (`data/aligned/` — immutable)

| Content type | Pairs | Chars (both langs) | Est. tokens |
|---|---|---|---|
| Bible verses | 30,838 | 8.9M | ~2.4M |
| Article paragraphs | 275,430 | 55.6M | ~14.6M |
| Daily text | 3,432 | 7.0M | ~1.8M |
| **Total (raw)** | **309,700** | **71.6M** | **~18.8M** |

### Quality issues in raw data

| Issue | Count | % |
|---|---|---|
| Duplicate IDs (overlapping library crawls) | 118,553 | 38.3% |
| Duplicate content (same text, different IDs) | 11,075 | 3.6% |
| Metadata (captions, credits, headers) | 1,049 | 0.3% |
| Identical pairs (TVL == EN, untranslated) | 406 | 0.1% |
| Extreme length ratios | 246 | 0.1% |
| **Total issues** | **131,329** | **42.4%** |

### Cleaned data (`data/cleaned/` — after `clean_pipeline.py`)

| Content type | Pairs | Chars (both langs) | Est. tokens |
|---|---|---|---|
| Bible verses | 30,827 | 8.9M | ~2.4M |
| Article paragraphs | 144,287 | 80.3M | ~21.1M |
| Daily text | 3,257 | 6.8M | ~1.8M |
| **Total (cleaned)** | **178,371** | **96.1M** | **~25.3M** |

Cleaned with balanced profile. Acceptance rate: 57.6%. See `data/cleaned/cleaning_report.json` for full statistics.

Run: `uv run python scripts/clean_pipeline.py [--profile balanced|strict|lenient]`

### Content coverage (complete)

All scrapeable TVL-EN parallel text from WOL has been captured:

| Source | Status | Coverage |
|---|---|---|
| Bible (NWT) | Complete | All 66 books, 1,189 chapters |
| Watchtower articles | Complete | Library crawl (2008–2025) |
| Awake! articles | Complete | Library crawl (1981–2025) |
| Meeting workbooks | Complete | Library crawl (2015–2025) |
| Books/publications | Complete | 22 pub codes |
| Brochures | Complete | All covered via library crawl |
| Ministry materials | Complete | 2010–2015 (all overlap with above) |
| Daily text | Complete | 2017-01-01 through 2025-12-31, 0 gaps |

### Not scraped (and why)

| Category | Count | Reason |
|---|---|---|
| Songs | 227 | Lyrics only; different text type |
| Videos | 146 | No text content |
| FAQ | 49 | Minimal, mostly covered via docId overlap |
| News | 36 | Very small, many English-only |
| Help pages | 17 | UI text, not useful for MT |

---

## 13. HuggingFace upload plan

1. Create dataset repo: `{username}/tuvaluan-english-parallel-jw`
2. Upload Parquet files organized by config
3. Include comprehensive dataset card with:
   - Language description and typological notes
   - Data collection methodology
   - Alignment strategy documentation
   - Known limitations (domain bias, translationese)
   - Suggested usage and citation
   - License information
4. Use `datasets` library `push_to_hub()` for upload
5. Verify dataset loads correctly: `datasets.load_dataset("{username}/tuvaluan-english-parallel-jw")`

---

## 14. Answered questions

- [x] **Does the JW.org Tuvaluan sitemap include hreflang alternates?** No — 0/7,103 URLs have alternates. Must use WOL-based alignment.
- [x] **What is the WOL Bible verse HTML structure?** `span.v` with `id="v{bookNo}-{ch}-{verse}-{part}"`, footnotes as `a.fn`, cross-refs as `a.b`. Different from jw.org's `span.verse` / `a.footnoteLink` / `a.xrefLink`.
- [x] **Does the CDN publication API support Tuvaluan?** No — language codes TVL, VL, TV all return 400/404.
- [x] **Can standard Python HTTP clients access JW.org/WOL?** No — TLS fingerprint detection blocks requests, httpx, and system curl. Docker curl-impersonate required.
- [x] **How many Tuvaluan pages exist?** 7,103 (not 1,310 as initially estimated from partial sitemap).
- [x] **What is the TVL/EN length ratio for Bible text?** Average 1.21 (TVL ~21% longer), range 0.83–1.64.

## 15. Tinker training cost estimates

All costs below are **USD** and use **only** the rates you provided.

Formulas, where:

* **P** = Prefill rate ($ / 1M input tokens)
* **S** = Sample rate ($ / 1M output tokens)
* **T** = Train rate ($ / 1M training tokens)

Formulas used for every model:

* **Stage 1 (10M-token training)**

  * 1x = `10 × T`
  * 2x = `20 × T`
  * 3x = `30 × T`
* **Stage 2 (synthetic data generation inference)**

  * Input = `200 × P`
  * Output = `200 × S`
  * Total = `200 × (P + S)`
* **Stage 3 (200M-token synthetic-data training)**

  * 1x = `200 × T`
  * 2x = `400 × T`
  * 3x = `600 × T`
* **Total pipeline**

  * **1x/1x** = `10T + 200(P+S) + 200T = 210T + 200(P+S)`
  * **2x/2x** = `20T + 200(P+S) + 400T = 420T + 200(P+S)`
  * **3x/3x** = `30T + 200(P+S) + 600T = 630T + 200(P+S)`

**Storage:** billed at **$0.10/GB/month**, but **cannot be calculated** from the information provided because no GB usage is given, so it is **excluded** from totals.

| Model | Prefill | Sample | Train | S1 1x | S1 2x | S1 3x | S2 In | S2 Out | S2 Total | S3 1x | S3 2x | S3 3x | Pipeline 1x | Pipeline 2x | Pipeline 3x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| meta-llama/Llama-3.2-1B | $0.03 | $0.09 | $0.09 | $0.90 | $1.80 | $2.70 | $6.00 | $18.00 | $24.00 | $18.00 | $36.00 | $54.00 | $42.90 | $61.80 | $80.70 |
| meta-llama/Llama-3.2-3B | $0.06 | $0.18 | $0.18 | $1.80 | $3.60 | $5.40 | $12.00 | $36.00 | $48.00 | $36.00 | $72.00 | $108.00 | $85.80 | $123.60 | $161.40 |
| Qwen/Qwen3-4B-Instruct-2507 | $0.07 | $0.22 | $0.22 | $2.20 | $4.40 | $6.60 | $14.00 | $44.00 | $58.00 | $44.00 | $88.00 | $132.00 | $104.20 | $150.40 | $196.60 |
| Qwen/Qwen3-30B-A3B | $0.12 | $0.30 | $0.36 | $3.60 | $7.20 | $10.80 | $24.00 | $60.00 | $84.00 | $72.00 | $144.00 | $216.00 | $159.60 | $235.20 | $310.80 |
| gpt-oss/GPT-OSS-20B | $0.12 | $0.30 | $0.36 | $3.60 | $7.20 | $10.80 | $24.00 | $60.00 | $84.00 | $72.00 | $144.00 | $216.00 | $159.60 | $235.20 | $310.80 |
| Qwen/Qwen3-8B | $0.13 | $0.40 | $0.40 | $4.00 | $8.00 | $12.00 | $26.00 | $80.00 | $106.00 | $80.00 | $160.00 | $240.00 | $190.00 | $274.00 | $358.00 |
| meta-llama/Llama-3.1-8B | $0.13 | $0.40 | $0.40 | $4.00 | $8.00 | $12.00 | $26.00 | $80.00 | $106.00 | $80.00 | $160.00 | $240.00 | $190.00 | $274.00 | $358.00 |
| gpt-oss/GPT-OSS-120B | $0.18 | $0.44 | $0.52 | $5.20 | $10.40 | $15.60 | $36.00 | $88.00 | $124.00 | $104.00 | $208.00 | $312.00 | $233.20 | $342.40 | $451.60 |
| Qwen/Qwen3-VL-30B-A3B-Instruct | $0.18 | $0.44 | $0.53 | $5.30 | $10.60 | $15.90 | $36.00 | $88.00 | $124.00 | $106.00 | $212.00 | $318.00 | $235.30 | $346.60 | $457.90 |
| Qwen/Qwen3-32B | $0.49 | $1.47 | $1.47 | $14.70 | $29.40 | $44.10 | $98.00 | $294.00 | $392.00 | $294.00 | $588.00 | $882.00 | $700.70 | $1,009.40 | $1,318.10 |
| Qwen/Qwen3-235B-Instruct-2507 | $0.68 | $1.70 | $2.04 | $20.40 | $40.80 | $61.20 | $136.00 | $340.00 | $476.00 | $408.00 | $816.00 | $1,224.00 | $904.40 | $1,332.80 | $1,761.20 |
| moonshotai/Kimi-K2-Thinking | $0.98 | $2.44 | $2.93 | $29.30 | $58.60 | $87.90 | $196.00 | $488.00 | $684.00 | $586.00 | $1,172.00 | $1,758.00 | $1,299.30 | $1,914.60 | $2,529.90 |
| Qwen/Qwen3-VL-235B-A22B-Instruct | $1.02 | $2.56 | $3.07 | $30.70 | $61.40 | $92.10 | $204.00 | $512.00 | $716.00 | $614.00 | $1,228.00 | $1,842.00 | $1,360.70 | $2,005.40 | $2,650.10 |
| deepseek-ai/DeepSeek-V3.1 | $1.13 | $2.81 | $3.38 | $33.80 | $67.60 | $101.40 | $226.00 | $562.00 | $788.00 | $676.00 | $1,352.00 | $2,028.00 | $1,497.80 | $2,207.60 | $2,917.40 |
| meta-llama/Llama-3.1-70B | $1.05 | $3.16 | $3.16 | $31.60 | $63.20 | $94.80 | $210.00 | $632.00 | $842.00 | $632.00 | $1,264.00 | $1,896.00 | $1,505.60 | $2,169.20 | $2,832.80 |
| moonshotai/Kimi-K2.5 | $1.47 | $3.66 | $4.40 | $44.00 | $88.00 | $132.00 | $294.00 | $732.00 | $1,026.00 | $880.00 | $1,760.00 | $2,640.00 | $1,950.00 | $2,874.00 | $3,798.00 |

Summary:

* **Cheapest model:** `meta-llama/Llama-3.2-1B` at **$42.90** for the **1x Stage 1 + inference + 1x Stage 3** pipeline
* **Most expensive model:** `moonshotai/Kimi-K2.5` at **$1,950.00** for the same pipeline
* **Cost range:** **$42.90 to $1,950.00**, a spread of **$1,907.10**

Notable ties:

* `Qwen/Qwen3-30B-A3B` and `gpt-oss/GPT-OSS-20B` have identical totals
* `Qwen/Qwen3-8B` and `meta-llama/Llama-3.1-8B` have identical totals

---

## 16. Current dataset token counts

> **Snapshot as of March 6, 2026 (post full scrape + cleaning).** Token estimates use the ~3.8 chars/token approximation for mixed TVL/EN text.

### Raw data (before cleaning)

| Source | Pairs | Total chars | Est. tokens |
|---|---|---|---|
| Articles | 275,430 | 55.6M | ~14.6M |
| Bible | 30,838 | 8.9M | ~2.4M |
| Daily text | 3,432 | 7.0M | ~1.8M |
| **Total (raw)** | **309,700** | **71.6M** | **~18.8M** |

### Cleaned data (after `clean_pipeline.py`, balanced profile)

| Source | Pairs | Total chars | Est. tokens |
|---|---|---|---|
| Articles | 144,287 | 80.3M | ~21.1M |
| Bible | 30,827 | 8.9M | ~2.4M |
| Daily text | 3,257 | 6.8M | ~1.8M |
| **Total (cleaned)** | **178,371** | **96.1M** | **~25.3M** |

### For fine-tuning (both directions, from cleaned data)

| Metric | Value |
|---|---|
| Training examples (x2 directions) | ~356k |
| Full sequence tokens (with prompts) | ~60M |
| Target tokens only | ~25.3M |

---

## 17. Open questions

- [x] How many WOL docIds actually have Tuvaluan translations? — 7,255 unique docIds scraped from 6 library categories + 22 publication codes, yielding 275,430 paragraph pairs (raw)
- [x] What date range of daily text content exists in Tuvaluan? — 2017-01-01 through 2025-12-31 (3,432 pairs, 3,287 dates, 0 gaps). Pre-2017 pages return empty content.
- [x] What is the exact `wtlocale` value for Tuvaluan `open` links? — `VL` (not `TVL`, which resolves to English)
- [x] Are there JW articles going back before 2017? — Yes, Tuvaluan publications on WOL go back to 1981 ("Mai Tou Malo" / kc). All discovered content has been scraped.
- [ ] Are there additional WOL Bible translation codes beyond `nwt` for Tuvaluan?
- [ ] Should songs/lyrics be included or excluded? (Different text type; may confuse MT models)
- [ ] Should we include a Tokelauan subset as a related-language augmentation?

## 18. Scraping Playbook

For a complete step-by-step guide to reproduce all data collection from scratch, see:

**[docs/SCRAPING_PLAYBOOK.md](docs/SCRAPING_PLAYBOOK.md)**

This includes all prerequisites, commands, troubleshooting, and a reproduction checklist.
