# TODO

> **Note:** Data gathering is still in progress. Issues below are based on the current snapshot (32,557 aligned pairs: 31,181 Bible + 1,052 articles + 324 daily text).

## 1. Verse boundary misalignment in Polynesian Bible translations

TVL verse 9 is 548 chars while EN verse 9 is only 115 chars — the TVL text for verse 9 includes content that EN splits across verses 9 and 10. And TVL verse 10 is just "10" (2 chars, essentially the verse number only). This is a translation paragraph structure difference — TVL wraps verses 9-10 into one span.v for verse 9, while the separate verse 10 span only has the number.

This is a known issue with Polynesian Bible translations — verse boundaries don't always align 1:1 with English at the span level. The current extraction merges multi-part verses but doesn't handle the case where one language combines verses.

This is a minor issue affecting ~5 out of 31,181 pairs (0.016%). The data is still correct — it's just that these verses have unusual length ratios. Not worth fixing now.

**Affected pairs (extreme TVL-long ratios in Bible):**
- `bible_49_1_9` — ratio 4.81 (544 vs 113 chars)
- `bible_47_5_7` — ratio 4.17 (175 vs 42 chars)
- `bible_44_13_9` — ratio 3.41 (290 vs 85 chars)
- `bible_4_5_12` — ratio 3.26 (395 vs 121 chars)
- `bible_1_8_6` — ratio 3.11 (233 vs 75 chars)

## 2. Em-dash-only verses (spurious verse markers)

16 Bible verse pairs where both TVL and EN text is just `——` (em-dash). These are verses that don't exist in the NWT translation (they appear in some manuscripts but are omitted in the NWT, which marks them with a dash placeholder).

All 16 are in the Gospels and Acts (books 40–45):
`bible_40_17_21`, `bible_40_18_11`, `bible_40_23_14`, `bible_41_7_16`, `bible_41_9_44`, `bible_41_9_46`, `bible_41_11_26`, `bible_41_15_28`, `bible_42_17_36`, `bible_42_23_17`, `bible_43_5_4`, `bible_44_8_37`, `bible_44_15_34`, `bible_44_24_7`, `bible_44_28_29`, `bible_45_16_24`

**Action:** Filter these out during data building — they contain no translatable content. The `build_tinker_mt_data.py` quality filter already catches them via `too_short` (< 10 chars), but `scrape_bible.py` could skip them at extraction time.

## 3. Article paragraph misalignment (position-based alignment drift)

Two articles have paragraphs where TVL and EN paragraph positions don't correspond to the same content:

**`article_1102008071`** (43 paragraphs, 3 misaligned):
- `_p40`: TVL question 15 paired with EN question 13-14 (ratio 0.30)
- `_p41`: TVL question 16-17 paired with EN question 15 (ratio 3.02)
- `_p31`: TVL footnote text paired with EN footnote stub (ratio 5.92)

**`article_1102008077`** (12 paragraphs, 3 misaligned):
- `_p8`: TVL is "© 2008, 2009" paired with EN disclaimer paragraph (ratio 0.09)
- `_p10`: TVL is "TE KAU ‵LOMI TUSI" paired with EN scripture credits (ratio 0.14)
- `_p14`: TVL disclaimer paragraph paired with EN "Watch Tower Bible..." org name (ratio 3.18)
- `_p4`: Both are "Photo Credits:" — identical, not translated (1 occurrence)

**Root cause:** Position-based paragraph alignment breaks when TVL and EN versions have different boilerplate/front-matter/back-matter paragraph structure. The actual body content is likely aligned correctly, but legal/credits paragraphs at the end differ.

**Action:** Consider stripping boilerplate paragraphs (copyright, photo credits, legal disclaimers) before alignment, or use a smarter alignment strategy for article tail paragraphs.

## 4. Duplicate text pairs across Bible (336 duplicates)

336 pairs have identical (tvl, en) text as another pair. Most are repeated liturgical formulas like:
- "Ne toe faipati atu a Ieova ki a Mose, ana muna:" / "Jehovah spoke further to Moses, saying:" (appears in Exodus, Leviticus, Numbers)
- "A Tavita." / "Of David." (Psalm superscriptions)

The `build_tinker_mt_data.py` already filters these (`duplicate_pair: 337` rejections), so this doesn't affect training data. But it slightly inflates raw pair counts.

**Action:** No action needed — already handled by the data builder. Worth noting for dataset statistics reporting.

## 5. Daily text gap: 2024-09-17 to 2024-12-31 (106 missing days)

The daily text data covers 2024-01-01 to 2025-03-05 but has a 106-day gap from Sep 17 to Dec 31, 2024. We have 324 out of an expected 430 days (75% coverage).

**Possible causes:**
- Scraping was interrupted or the date range wasn't fully specified
- TVL daily text pages may not exist for that period
- WOL may have changed page structure during that period

**Action:** Re-run `scrape_daily_text.py` targeting the missing date range to determine if the content exists. If the pages genuinely don't exist in TVL, document as a known coverage gap.

## Summary

Most issues above are already handled by `build_tinker_mt_data.py` quality filters (duplicate removal, min-length, length-ratio thresholds). The daily text gap (#5) is the most actionable — likely just needs the scraper re-run for those dates.

---

## 6. Chapter heading / superscription pairs are very short

28 Bible pairs have EN text < 10 chars (mostly Psalm superscriptions like "Of David.", "A melody.", "Job said:"). These are real translations but provide minimal training signal. Article chapter headings ("CHAPTER 7", "CHAPTER 8", etc.) are similarly short.

The data builder rejects pairs where either side is < 10 chars (`too_short: 30` rejections), which catches most of these. A few borderline ones (exactly 9 chars) slip through.

**Action:** No action needed — quality filter handles this. The threshold of 10 chars is reasonable.

## Known issues to address

7. ### Name hallucination

The Stage A adapter was trained on JW.org data where names are always
transliterated (Ieova↔Jehovah, Iesu↔Jesus). When it encounters unfamiliar
names, it hallucinates JW-style transliterations:

- "Tausa" → "Taʹhosh"
- "Nukufetau" → "Nuk·phatʹta"

Prompt engineering does not fix this — it's baked into the weights.

**Fix**: Augment Stage A training data with ~1,000 synthetic name-preservation
pairs (Tuvaluan place names, personal names in simple sentence templates) and
retrain. Football articles will then provide ongoing reinforcement since every
article contains dozens of proper nouns that must be copied verbatim.

8. ### Domain vocabulary gap

Football-specific terms (penalty, offside, midfielder, VAR) have no Tuvaluan
equivalents in the training data. The adapter may produce awkward paraphrases
or hallucinated translations. Two strategies:

1. **Loanword preservation**: Keep English football terms as loanwords in
   Tuvaluan output (common practice in Pacific languages for sports terms)
2. **Post-training glossary**: Build a football term glossary and use it as
   few-shot context in the translation prompt

9. ### Quality filtering

Not all translated output will be usable. Apply the same quality filters used
for the JW.org parallel corpus:

- Minimum character length per side
- Length ratio filtering (reject extreme ratios)
- Duplicate detection
- Metadata detection (headers, footers, navigation text)

10. ### Model collapse in football translation pipeline

**Status:** Detection + hiding implemented. Root cause identified. Fix pending.

**Stats:** 31/56 translations (55%) flagged as collapsed.

#### Root cause analysis

Three factors drive collapse, in order of impact:

**1. Single-paragraph body (primary cause — 72% collapse rate)**

All 25 Sky Sports articles have `paragraph_count=1` because Sky's HTML has no
`<p>` tags — the scraper's `split_paragraphs()` falls back to `\n\n`-splitting
and finds nothing, so the entire 500-1800 word body is sent as a single chunk.
With `MAX_TOKENS=512`, the model can't finish the translation and loops.

- paras=1 articles: 18/25 collapsed (72%)
- paras>1 articles: 13/31 collapsed (42%)
- Sky Sports: 18/25 collapsed (72%)
- FIFA.com: 2/13 collapsed (15%)
- Goal.com: 11/18 collapsed (61%)

**2. Out-of-domain content (secondary cause)**

The Tinker model was trained on JW.org religious/educational text. Football
jargon, TV guide listings, subscription pricing, and pop culture content
("Ted Lasso", "Fubo Review") are maximally out of domain. The model has no
anchoring vocabulary and resorts to repetitive paraphrasing.

Collapse rate by content type:
- Football match analysis: ~60% collapse
- TV/streaming guides: ~50% collapse (mixed — short ones survive)
- FIFA World Cup articles: ~15% collapse (closer to training domain: formal, informational)
- Player interviews: ~20% collapse (narrative structure similar to training data)

**3. Text length (correlated with #1)**

Longer articles collapse more, but this is confounded by #1 — long articles
from Sky are all single-paragraph. Among multi-paragraph articles, length
alone is not a strong predictor.

Collapse rate by EN word count:
- 0-200 words: 29%
- 200-400: 53%
- 400-600: 43%
- 600-800: 67%
- 800-1000: 62%
- 1000+: 69%

#### Detection system

Implemented in `scripts/detect_collapse.py`. Multi-signal approach:
1. Whole-text 4-gram uniqueness ratio (threshold < 0.3)
2. Max single 4-gram frequency (threshold > 0.4)
3. Tail collapse: last 80 words checked with 3-gram and 4-gram (catches mid-text degeneration)
4. Per-paragraph collapse (catches one bad paragraph diluted by good ones)
5. Dominant n-gram check (any phrase > 10% of tail)

The `is_collapsed` flag in `translations` table hides affected articles on the
site via `CASE WHEN t.is_collapsed = 1 THEN NULL` in the SQL query.

#### Fixes to implement

1. **Fix Sky Sports paragraph splitting** — Sky uses `<br>` tags or bare
   newlines instead of `<p>` tags. Update `split_paragraphs()` to handle this.
   This alone should cut collapse rate from 55% to ~25%.

2. **Increase MAX_TOKENS** — 512 is too low for even single paragraphs of
   200+ words. Increase to 1024 or 2048 (check Tinker API limits).

3. **Add sentence-level chunking fallback** — if a paragraph exceeds ~100
   words, split on sentence boundaries before translating. Reassemble after.

4. **Domain-specific prompt tuning** — add football glossary terms to the
   system prompt. Instruct the model to preserve loanwords for terms without
   Tuvaluan equivalents.

5. **Temperature + retry is already implemented** — 3 attempts with
   temperature escalation (0.0 → 0.3 → 0.7). All attempts saved to
   `translation_attempts` table for RL training.

11. ### Unstructured language assets (unstruct_lang_data) are not yet in the training graph

The following assets are present but not yet integrated:

- `unstruct_lang_data/Tatoeba-v2023-04-12-en&tvl.tsv` (14 aligned en↔tvl rows)
- `unstruct_lang_data/DICTIONARY_Tuv_Palagi.pdf` (dictionary entries, parseable by `pdftotext`)
- `unstruct_lang_data/Tuvalu_News_Sheets_66-99.pdf` (image-based scan)
- `unstruct_lang_data/Tuvalu_News_Sheets_Part 2.pdf` (image-based scan)
- `unstruct_lang_data/The_magical_garlands_of_Nukufetau.pdf` (image-based scan)

Action plan:

1. Add a new external ingestion script path (`data/external` + loader + manifest).
2. Normalize Tatoeba rows into Stage A-compatible translation pairs.
3. Parse `DICTIONARY_Tuv_Palagi.pdf` into lexical pairs + plant/animal/name term lists and route:
   - small, high-fidelity bilingual term rows to Stage A seed,
   - larger glossary list to Stage B preservation checks.
4. Build an OCR pipeline for scanned PDFs and extract only high-confidence text blocks.
5. Run a name/place/fauna/flora extraction pass to create anti-hallucination test set.
6. Add metadata fields (`source_file`, `source_section`, `source_row`, `quality_score`) to all injected rows.
7. Track each batch in `todo.md` and in stage manifests as a separate artifact source.
