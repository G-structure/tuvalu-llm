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
