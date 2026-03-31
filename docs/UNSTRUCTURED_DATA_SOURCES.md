# Unstructured Data Sources

Catalog of the raw assets stored under `unstruct_lang_data/`, grouped by source family and annotated with current pipeline status.

Related docs:

- [UNSTRUCTURED_DATA_PIPELINE.md](UNSTRUCTURED_DATA_PIPELINE.md)
- [UNSTRUCTURED_DATA_MINING_PLAYBOOK.md](UNSTRUCTURED_DATA_MINING_PLAYBOOK.md)
- [DATA_PIPELINE.md](DATA_PIPELINE.md)

Last checked: `2026-03-30`

## Snapshot

`unstruct_lang_data/` currently contains 182 files:

| Type | Count |
|---|---:|
| PDF | 74 |
| MP4 | 66 |
| JPG | 26 |
| CSV | 4 |
| TSV | 3 |
| ZIP | 2 |
| JSONL | 2 |
| WEBM | 1 |
| DOCX | 1 |
| TXT | 1 |
| JSON | 1 |

This folder is broader than the current text-training pipeline. Some assets are already converted into `data/external/stage_a_seed/*.jsonl`, some are only used for OCR term mining, and some are still raw-only.

## Status Legend

| Status | Meaning |
|---|---|
| `Ingested` | Converted into Stage A seed rows in `data/external/stage_a_seed/` |
| `Merged` | Present in the merged training render `data/finetune/stage_a_mt_v2/` |
| `Term-only` | Used only for OCR term extraction in `data/external/stage_b_seed/unstruct_ocr_terms.jsonl` |
| `Raw-only` | Present in `unstruct_lang_data/` but not wired into the current text pipeline |
| `Duplicate/reference` | Extra copy, archive, manifest, or metadata helper rather than a primary source |
| `Media-only` | Audio/video asset not currently transcribed into text training data |

## 1. Canonical root-level sources

These are the top-level files in `unstruct_lang_data/` that originally seeded the unstructured work.

| Source | Description | Current status |
|---|---|---|
| `DICTIONARY_Tuv_Palagi.pdf` | Tuvaluan-English dictionary PDF | `Ingested`, `Merged` |
| `Tatoeba-v2023-04-12-en&tvl.tsv` | Tatoeba sentence/phrase pairs | `Ingested`, `Merged` |
| `tuvalu_en_bilingual_corpus_full_listing.pdf` | Corpus listing reference PDF | `Duplicate/reference` |
| `The_magical_garlands_of_Nukufetau.pdf` | Scanned children's book at repo root | `Term-only` |
| `Tuvalu_News_Sheets_66-99.pdf` | Historic news archive scan | `Term-only` |
| `Tuvalu_News_Sheets_Part 2.pdf` | Historic news archive scan | `Term-only` |
| `REAL ONES ONLY-20260310T045923Z-3-001.zip` | Archive bundle | `Duplicate/reference` |

## 2. Word-pair and lexical sources

These are the highest-yield sources for short translation pairs.

### 2.1 Dictionary

- Primary file: `DICTIONARY_Tuv_Palagi.pdf`
- Duplicate copy: `REAL ONES ONLY/Linguistic Academic Guides/DICTIONARY Tuv_Palagi (2).PDF.pdf`
- Current outputs:
  - `data/external/stage_a_seed/unstruct_dictionary_tvl_en.jsonl`
  - `data/external/stage_a_seed/unstruct_dictionary_en_tvl.jsonl`
  - `data/external/stage_b_seed/unstruct_dictionary_terms.jsonl`
- Coverage:
  - Stage A seed rows: `9304` TVL->EN and `10780` EN->TVL
  - Merged Stage A v2 domain contribution: `46734` directional examples

### 2.2 Tatoeba

- Primary file: `Tatoeba-v2023-04-12-en&tvl.tsv`
- Duplicate copy: `REAL ONES ONLY/word pairings and data sets/Tatoeba-v2023-04-12-en_26tvl (1).tsv`
- Status: `Ingested`, `Merged`
- Current output:
  - `data/external/stage_a_seed/unstruct_tatoeba.jsonl` with `14` rows

### 2.3 Corpus v2 package

Folder:

- `REAL ONES ONLY/word pairings and data sets/tuvalu_en_bilingual_corpus_v2/`

Important files:

- `pairs/corpus_pairs_dedup.jsonl`
- `pairs/training_pairs.tsv`
- `pairs/corpus_pairs_with_audio.csv`
- `pairs/corpus_pairs_without_audio.csv`
- `metadata/corpus_full.jsonl`
- `metadata/corpus_full.csv`
- `tuvalu_en_bilingual_corpus_full_listing.pdf`

Status:

- The pair data itself is `Ingested`, `Merged` via `unstruct_corpus_v2.jsonl`
- The listing PDF, CSVs, README, and metadata files are mostly `Duplicate/reference`

Current output:

- `data/external/stage_a_seed/unstruct_corpus_v2.jsonl` with `3658` rows

## 3. Government, health, education, and civic PDFs

These are the bilingual and paired-policy documents under `REAL ONES ONLY/Documents/`.

### 3.1 Eng-TVL together

Folder:

- `REAL ONES ONLY/Documents/Eng-TVL together/`

Sources:

| Source | Notes | Status |
|---|---|---|
| `BILINGUAL Family Tax Benefit - Tuvaluan.pdf` | Bilingual government service document | `Ingested`, `Merged` |
| `Child Care Subsidy - Tuvaluan.pdf` | Bilingual government service document | `Ingested`, `Merged` |
| `mpp_te_gana_tuvalu_language_cards_bilingual.pdf` | Language-card style vocabulary source | `Ingested`, `Merged` |
| `tepapa_tuvalu_activity_book_bilingual.pdf` | Te Papa educational activity book | `Ingested`, `Merged` |
| `Medicare is Australia’s health care system - Tuvaluan.pdf` | Health/system guide | `Raw-only` |
| `Tuvalu STEPS report 2015.pdf` | Public health / survey report | `Raw-only` |

### 3.2 En-TVL separate

Folder:

- `REAL ONES ONLY/Documents/En-TVL seperate/`

Sources already ingested:

- `BCG Vaccine/` pair
- `Citizen budget 2025/` pair
- `Climate children/` pair
- `Covid alert Levels/` pair
- `covid level 4/` pair
- `Diabetes/` pair
- `Health reform/` pair
- `Measles/` pair
- `Menincoccal (inconsistent format/` pair
- `Mormon Prayer/` JPG pair via OCR
- `Pac education 2030/` pair
- `Resilient emergency sheet/` pair
- `Strategic Action Plan /` pair
- `TCCP 2012/` pair
- `Traveller Factsheet/` pair
- `biogass /` pair

These feed:

- `data/external/stage_a_seed/unstruct_paired_pdfs.jsonl`
- `data/external/stage_a_seed/unstruct_bilingual_pdfs.jsonl`
- `data/external/stage_a_seed/unstruct_language_cards.jsonl`
- `data/external/stage_a_seed/unstruct_mormon_prayer.jsonl`

Known raw-only items in this family:

| Source | Status |
|---|---|
| `Finance budget/finance_budget_2025_2026_en.pdf` | `Raw-only` |
| `Finance budget/finance_budget_2025_2026_tuvaluan.pdf` | `Raw-only` |
| `Pacific Education action Plan/Action-Plan-for-Pacific-Education-20202030.pdf` | `Raw-only` |

### 3.3 "Don't use yet" holding area

Folder:

- `REAL ONES ONLY/Documents/Don_t use yet/`

Files:

- `he2212_bcg_info_parents_tuvaluan.pdf`
- `he2233_bcg_aftercare_tuvaluan.pdf`
- `he2783_free_bowel_screening_tuvaluan.pdf`
- `he5031_measles_watch_for_symptoms_tuvaluan.pdf`
- `he5032_measles_protect_yourself_tuvaluan.pdf`
- `he5033_measles_could_you_have_it_tuvaluan.pdf`
- `he7521_bowel_test_kit_instructions_tuvaluan.pdf`

Status: `Raw-only`

These appear to be staging copies or untranslated-side files that have not been paired cleanly yet.

## 4. Children's books and oral material

Folder:

- `REAL ONES ONLY/Childrens books/`

Sources:

| Source | Notes | Status |
|---|---|---|
| `The gifts of Pai and Vau-spreads.pdf` | Bilingual children's book | `Ingested`, `Merged` |
| `Tuvalu Toku Atufenua Pele.pdf` | Bilingual essays/book | `Ingested`, `Merged` |
| `Matua Fakamoe of Nanumaga(1).pdf` | Additional children's book | `Raw-only` |
| `The magical garlands of Nukufetau(2).pdf` | Duplicate/alternate copy of root scan | `Raw-only` |
| `Am I Small/*.jpg` | 24 screenshot images of a children's book | `Raw-only` |

Oral and traditional narrative material:

- `REAL ONES ONLY/Documents/nanumea/Tefolaha tale 1 - Tepou, pp 292-307 from Heirs of Tefolaha.pdf`
- `REAL ONES ONLY/Documents/nanumea/Tefolaha tale 2 - Sosemea & Takitua, pp 308-316 from Heirs of Tefolaha.pdf`

Status: `Ingested`, `Merged`

These feed `unstruct_nanumea_tales.jsonl`.

## 5. Linguistic and academic references

Folder:

- `REAL ONES ONLY/Linguistic Academic Guides/`

Sources:

| Source | Notes | Status |
|---|---|---|
| `epdf.pub_tuvaluan-a-polynesian-language-of-the-central-pacific-descriptive-grammars.pdf` | Besnier grammar source | `Ingested`, `Merged` |
| `DICTIONARY Tuv_Palagi (2).PDF.pdf` | Duplicate dictionary copy | `Duplicate/reference` |

Outputs:

- `data/external/stage_a_seed/unstruct_grammar.jsonl` with `2333` rows

## 6. Nature and biodiversity sources

Folder:

- `REAL ONES ONLY/Nature/`

Sources:

| Source | Notes | Status |
|---|---|---|
| `Fauna/Thaman_2015_Fishes_Tuvalu_Tokelau.PDF.pdf` | Fish names / biodiversity table | `Ingested`, `Merged` |
| `Thaman 2016.pdf` | Flora listing | `Ingested`, `Merged` |
| `Flora/Copy of Thaman 2016.pdf` | Duplicate flora copy | `Duplicate/reference` |
| `Tuvalu R2R BioRAP Field Guide.pdf` | Broader biodiversity reference | `Raw-only` |
| `tv-nr-05-en.pdf` | Nature-related PDF, English-side reference | `Raw-only` |

Outputs:

- `data/external/stage_a_seed/unstruct_fishes.jsonl` with `998` rows
- `data/external/stage_a_seed/unstruct_flora.jsonl` with `436` rows

## 7. Historic archives and OCR-heavy scans

These are primarily scanned historical documents and newspapers.

Sources:

| Source | Notes | Status |
|---|---|---|
| `Tuvalu_News_Sheets_66-99.pdf` | Root-level archive scan | `Term-only` |
| `Tuvalu_News_Sheets_Part 2.pdf` | Root-level archive scan | `Term-only` |
| `REAL ONES ONLY/Documents/Historic archives 70s-2000_s/Tuvalu News Sheets 66-99 (1).pdf` | Archive copy | `Raw-only` |
| `REAL ONES ONLY/Documents/Historic archives 70s-2000_s/Tuvalu News Sheets Part 2 (1).pdf` | Archive copy | `Raw-only` |
| `REAL ONES ONLY/Documents/Historic archives 70s-2000_s/Tuvalu - News Sheets Part One (1).pdf` | Part one archive copy | `Raw-only` |
| `The_magical_garlands_of_Nukufetau.pdf` | Root-level scan | `Term-only` |

Current pipeline use:

- OCR artifacts exist in `data/external/ocr_scans/`
- These scans currently do **not** produce translation pairs
- They only contribute conservative OCR term candidates in `data/external/stage_b_seed/unstruct_ocr_terms.jsonl`

## 8. Audio and video collections

Folders:

- `REAL ONES ONLY/Audio/`
- `REAL ONES ONLY/Audio/Tuvalu songs/`
- `REAL ONES ONLY/Audio/voice/`

Examples:

- `WIKITONGUES_ Paulo speaking Tuvaluan.mp4`
- `Tuvalu language strong but still under threat.mp4`
- `Tuvalu Language Week 2025.mp4`
- `Ocean-Buoy-Awareness-Tuvaluan...mp4`
- `Coastal_Inundation_Awareness_-_Tuvaluan.webm...`
- 20+ song files under `Audio/Tuvalu songs/`

Status: `Media-only`

There is no checked-in speech-to-text or subtitle extraction path feeding these files into the training datasets.

## 9. Miscellaneous, duplicates, and support files

These exist in the tree but are not primary corpus sources.

| Source family | Notes | Status |
|---|---|---|
| `REAL ONES ONLY/misc copies/*` | Duplicate copies of other PDFs | `Duplicate/reference` |
| `REAL ONES ONLY/tuvalu.zip` | Archive bundle | `Duplicate/reference` |
| `REAL ONES ONLY/Tuvalu - Instructions for building an eval.docx` | Planning doc, not corpus text | `Duplicate/reference` |
| `tuvalu_en_bilingual_corpus_v2/README.txt` | Metadata helper | `Duplicate/reference` |
| `tuvalu_en_bilingual_corpus_v2/metadata/*.csv|jsonl` | Corpus metadata | `Duplicate/reference` |
| `tuvalu_en_bilingual_corpus_v2/pairs/*csv` | Derived/export helper tables | `Duplicate/reference` |

## 10. Current pipeline coverage summary

### Sources already converted into Stage A seed files

- Dictionary
- Tatoeba
- Corpus v2 pairs
- Paired government and health PDFs
- Bilingual PDFs
- Language cards
- Te Papa activity book
- Mormon prayer image pair
- Besnier grammar examples
- Fishes and flora
- Pai and Vau
- Toku Atufenua
- Nanumea tales

### Sources present in `unstruct_lang_data/` but still not converted into text training rows

- `The_magical_garlands_of_Nukufetau.pdf`
- `Tuvalu_News_Sheets_66-99.pdf`
- `Tuvalu_News_Sheets_Part 2.pdf`
- `Matua Fakamoe of Nanumaga(1).pdf`
- `Am I Small/*.jpg`
- `Tuvalu STEPS report 2015.pdf`
- `Medicare is Australia’s health care system - Tuvaluan.pdf`
- `finance_budget_2025_2026_en.pdf`
- `finance_budget_2025_2026_tuvaluan.pdf`
- `Action-Plan-for-Pacific-Education-20202030.pdf`
- `Tuvalu R2R BioRAP Field Guide.pdf`
- `tv-nr-05-en.pdf`
- All audio/video assets

## 11. Important training note

The presence of a source in `unstruct_lang_data/` does not mean it was used in a training run.

Current repo state:

- The raw unstructured extractions live in `data/external/stage_a_seed/`
- The standalone unstructured Stage A build lives in `data/finetune/stage_a_mt/unstructured_seed/`
- The merged render that includes unstructured data is `data/finetune/stage_a_mt_v2/`

If training used `data/finetune/stage_a_mt/train_balanced.jsonl`, that run did not use the unstructured corpus. If training used `data/finetune/stage_a_mt_v2/train_balanced.jsonl`, then the merged unstructured sources listed above were included.
