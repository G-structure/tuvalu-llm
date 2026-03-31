# Stage C Native-Document Grounding Spec

This document turns the current Stage C direction into an operational data pipeline.

Short version:

- Do not treat Stage C as one fixed prompt template.
- Treat it as a native-document grounding pipeline built around real Tuvaluan documents.
- Keep the assistant side real Tuvaluan whenever possible.
- Spend most generation effort on prompts, task variants, preference pairs, metadata, and evaluation artifacts around native TVL answers.

This spec is intentionally compatible with the repo's current staged training pipeline, where:

- Stage A = translation adapter
- Stage B = bilingual capability adapter

In discussion, "Stage C" here means the next data-improvement layer for the bilingual capability stage, centered on grounded native-TVL data.

Related docs:

- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)
- [TVL_EN_TINKER_PLAN.md](TVL_EN_TINKER_PLAN.md)
- [UNSTRUCTURED_DATA_SOURCES.md](UNSTRUCTURED_DATA_SOURCES.md)
- [PRIVATE_CHAT_DATA_PLAN.md](PRIVATE_CHAT_DATA_PLAN.md)

Last updated: `2026-03-30`

## 1. Goals

Stage C should improve:

1. Native Tuvaluan instruction following
2. Faithfulness to local source documents
3. Register control and style flexibility
4. Entity preservation for place names, institutions, dates, and quotes
5. Resistance to English leakage, translationese, and wrong-language drift

It should not primarily optimize for:

- raw synthetic volume
- translated-English-only instructions
- RL-heavy experimentation under short deadlines

## 2. Core Principle

The center of gravity is the source document, not the prompt template.

For each native TVL document or segment, the pipeline should derive many task variants while preserving a real or source-supported Tuvaluan answer. The synthetic work should mostly expand the instruction surface around grounded TVL content.

Preferred order of value:

1. Native TVL answer copied or lightly transformed from a real source
2. Native TVL answer tightly constrained by extracted facts from a real source
3. Synthetic TVL answer only when grounded answers are impossible and the example is clearly marked

## 3. Source Priorities

Stage C should draw first from documents that are already local, culturally grounded, and mostly native TVL.

Highest-priority source families:

- Recovered native TVL news scans and notices
- Government and civic documents with substantial Tuvaluan text
- Stories, oral-history material, and cultural writing
- Community-facing health and education documents
- Radio/news style prose
- Speech transcripts or subtitle-like material once cleaned

Likely repo-local candidates today:

- historic news scans after OCR recovery
- native or bilingual government PDFs
- Nanumea oral/traditional narrative material
- children's books and community literature
- any held-out native TVL notices or admin text not already overused in training

For source inventory, see [UNSTRUCTURED_DATA_SOURCES.md](UNSTRUCTURED_DATA_SOURCES.md).

## 4. Task Families

One source document should expand into multiple task families. Do not over-concentrate on `topic -> full article`.

Required task families:

- `native_request_article`
- `english_request_tvl_answer`
- `mixed_request_tvl_answer`
- `fact_sheet_to_article`
- `headline_generation`
- `lead_generation`
- `summary_short`
- `summary_medium`
- `qa_grounded`
- `entity_extraction`
- `quote_preservation`
- `radio_rewrite`
- `formal_rewrite`
- `plain_language_rewrite`
- `translation_to_english`
- `explain_in_english`

Optional later families:

- `bulletin_board_notice`
- `speech_script`
- `reading_comprehension`
- `stance_or_theme_identification`
- `error_correction`

## 5. Prompt Modes

Each grounded answer should usually be paired with multiple prompt styles.

Minimum prompt-mode set:

1. `native_tvl_user`
2. `english_user_tvl_answer`
3. `mixed_user_tvl_answer`
4. `fact_sheet_transform`

Recommended extras:

- `radio_host_style`
- `formal_official_style`
- `simple_reader_style`
- `headline_editor_style`

Example:

The same answer paragraph can be paired with:

- a native Tuvaluan request
- an English request asking for a Tuvaluan answer
- a mixed TVL/EN request
- a structured fact sheet asking for a prose article

## 6. Output Artifact Families

Stage C should produce four main artifact types.

Before those, it also needs one OCR-recovery artifact family for native scans that are not yet training-ready.

### 6.0 OCR recovery artifacts

Purpose:

- turn OCR-only native scans into article-level grounded text that can actually feed Stage C

Suggested path:

- `data/stage_c/ocr_recovered/*.jsonl`

Suggested record fields:

- `source_scan`
- `page_range`
- `layout_type`
- `tvl_text`
- `en_text` if present
- `article_id`
- `ocr_confidence`
- `recovery_method`
- `qa_status`

### 6.1 Document registry

Purpose:

- canonical list of grounded sources
- provenance, text quality, rights notes, domain tags, and split assignment

Suggested path:

- `data/stage_c/native_doc_registry.jsonl`

### 6.2 Grounded SFT examples

Purpose:

- source-anchored instruction-response examples for supervised tuning

Suggested path:

- `data/stage_c/grounded_sft/*.jsonl`

### 6.3 Preference pairs

Purpose:

- language-fidelity and style-alignment training
- chosen vs rejected answers for leakage, translationese, entity loss, and register errors

Suggested path:

- `data/stage_c/preferences/*.jsonl`

### 6.4 Held-out eval set

Purpose:

- contamination-resistant native-TVL evaluation

Suggested path:

- `data/stage_c/eval/*.jsonl`

## 7. JSONL Schemas

These are operational schemas, not abstract ideas. They are designed to be implementable in the repo.

### 7.1 Native document registry schema

```json
{
  "doc_id": "native_doc:news:funafuti:2026-0001",
  "source_path": "unstruct_lang_data/REAL ONES ONLY/Documents/...",
  "source_family": "government_pdf",
  "title": "Optional title if known",
  "language_profile": "tvl_primary",
  "domains": ["news", "civic"],
  "content_kind": "article",
  "text_quality": {
    "ocr_quality": "medium",
    "normalization_status": "normalized_v1",
    "orthography_reviewed": false
  },
  "grounding_level": "direct_text",
  "copyright_status": "internal_research_only",
  "ingest_status": "candidate",
  "segment_count": 14,
  "notes": "Native TVL article with some OCR noise.",
  "metadata": {
    "page_start": 3,
    "page_end": 5,
    "source_hash": "sha256:...",
    "created_at": "2026-03-30T00:00:00Z"
  }
}
```

### 7.2 Grounded SFT schema

```json
{
  "id": "grounded_sft:news:funafuti:2026-0001:headline:native_tvl_user:00",
  "task_family": "headline_generation",
  "prompt_mode": "native_tvl_user",
  "messages": [
    {"role": "system", "content": "You are a careful Tuvaluan writer. Stay faithful to the source."},
    {"role": "user", "content": "Tuku mai se ulutala puupuu mo tonu mo te tala tenei."},
    {"role": "assistant", "content": "..." }
  ],
  "answer_origin": "source_derived",
  "target_language": "tvl",
  "source_doc_id": "native_doc:news:funafuti:2026-0001",
  "source_segments": ["seg_04", "seg_05"],
  "support_type": "direct_support",
  "faithfulness_risk": "low",
  "domain": "news",
  "register": "journalistic",
  "metadata": {
    "generator": "gpt-5.4-mini-batch",
    "review_status": "auto_pass",
    "entity_constraints": ["Funafuti", "date", "institution_name"]
  }
}
```

### 7.3 Preference pair schema

```json
{
  "id": "pref:news:funafuti:2026-0001:radio_rewrite:00",
  "task_family": "radio_rewrite",
  "prompt_mode": "english_user_tvl_answer",
  "messages": [
    {"role": "system", "content": "Write in natural Tuvaluan and stay faithful to the source."},
    {"role": "user", "content": "Rewrite this for radio in Tuvaluan."}
  ],
  "chosen": "Natural Tuvaluan answer...",
  "rejected": "Leaky or translationese answer...",
  "preference_reason_tags": [
    "english_leakage",
    "translationese",
    "entity_drop"
  ],
  "source_doc_id": "native_doc:news:funafuti:2026-0001",
  "source_segments": ["seg_04", "seg_05", "seg_06"],
  "metadata": {
    "pair_source": "model_judged_then_human_verified",
    "chosen_model": "gpt-5.4",
    "rejected_model": "gpt-5.4-mini"
  }
}
```

### 7.4 Eval item schema

```json
{
  "id": "eval:native_doc:news:funafuti:2026-0001:summary_medium:00",
  "split": "held_out",
  "task_family": "summary_medium",
  "prompt": "Fakatoetoefaka se tala tenei i te Tuvaluan.",
  "reference_answer": "Reference TVL answer...",
  "source_doc_id": "native_doc:news:funafuti:2026-0001",
  "source_segments": ["seg_02", "seg_03", "seg_04"],
  "scoring_axes": [
    "adequacy",
    "in_language_fidelity",
    "entity_preservation",
    "style_fit",
    "source_support"
  ],
  "metadata": {
    "human_verified": true
  }
}
```

## 8. Generation Rules

### 8.1 What should stay real

Prefer real Tuvaluan for:

- assistant answers
- quotes
- names
- institution names
- dates and local references
- idiomatic phrasing when present in the source

### 8.2 What can be synthesized

Synthetic expansion is encouraged for:

- user prompts
- rubric text
- style labels
- task framing
- preference comparisons
- extracted facts and structured metadata

### 8.3 What must stay grounded

Do not generate unsupported facts. Each example should carry source segments and a support type.

Support levels:

- `direct_support`
- `light_transform`
- `fact_compilation`
- `weak_support`

Only the first three should enter default SFT pools.

## 9. Cleaning and Filtering

Cleaning should happen before large-scale synthesis.

Mandatory preprocessing:

1. OCR repair
2. orthography normalization
3. duplicate removal
4. language-ID sanity checks
5. entity normalization
6. boilerplate/header/footer stripping
7. domain balancing

Hard filters for grounded SFT:

- empty or near-empty answer
- unsupported answer spans
- obvious English leakage in TVL target
- Samoan or other Polynesian drift
- broken entities or dates
- duplicated prompt-answer pairs
- low-confidence OCR segments without review

## 10. Preference-Tuning Targets

Preference data should focus on failure modes actually seen in Stage B/Stage C behavior.

Priority rejection tags:

- `english_leakage`
- `translationese`
- `wrong_language`
- `entity_drop`
- `hallucinated_fact`
- `bad_register`
- `unnatural_headline`
- `quote_mangling`
- `overly_literal_translation`

Default order:

1. grounded SFT
2. preference data for fidelity/style
3. only later, if needed, any RL-style work

## 11. Eval Design

The eval set should be:

- native first
- held out by document, not by random example
- resistant to contamination
- decomposed into axis-level scoring

Required eval slices:

- news/article
- government notice
- cultural or narrative prose
- short factual notice
- OCR-noisy source after cleanup
- mixed-prompt requests

Scoring should not depend on a raw TVL-only model judge alone. Use rubric-based scoring with human review on a subset.

## 12. Active Error Mining Loop

Do not keep generating evenly across all task families.

Loop:

1. sample real native-TVL prompts against the current model
2. cluster errors by failure type
3. generate new grounded data only for weak slices
4. rebuild preference data only for the failure tags that matter
5. refresh held-out eval slices if the task mix changes

Suggested error clusters:

- native prompt misunderstanding
- English output leakage
- wrong named entity
- loss of local idiom
- inability to rewrite by register
- unsupported summary hallucination

## 13. Effort Allocation

Recommended Stage C effort split:

| Workstream | Share |
|---|---:|
| Reverse-instruction grounded SFT from native documents | 18% |
| Multi-task expansion from each document | 12% |
| Cleaning, OCR repair, dedup, normalization, balancing | 12% |
| Bilingual and mixed prompt mirrors around native outputs | 10% |
| Preference tuning data for language fidelity | 10% |
| Native held-out eval set | 10% |
| Terminology and named-entity preservation tasks | 8% |
| Active error mining and targeted regeneration | 7% |
| Backtranslation and round-trip filtering | 6% |
| Continued-pretraining pack preparation | 4% |
| Helper-model filtering / metadata extraction | 2% |
| RL-heavy or RFT-heavy work | 1% |

This intentionally prioritizes grounded SFT, cleanup, and preference data over speculative RL work.

## 14. OpenAI Credit Strategy

Use external API credits mainly for:

- prompt-side synthesis
- grounded task expansion
- adjudication on hard examples
- preference-pair drafting
- metadata extraction and filtering

Suggested operating mix:

- strongest model for rubric writing, hard adjudication, and gold preference decisions
- mini model for bulk task synthesis and grounded rewrites
- nano-class model for cheap tagging, language checks, and metadata extraction

Use batch mode for large offline jobs whenever possible. Do not spend most credits on generic translated-English SFT.

## 15. Suggested Repo Layout

```text
data/
  stage_c/
    native_doc_registry.jsonl
    grounded_sft/
      native_tvl_user.jsonl
      english_user_tvl_answer.jsonl
      mixed_user_tvl_answer.jsonl
      fact_sheet_transform.jsonl
    preferences/
      language_fidelity.jsonl
      register_control.jsonl
    eval/
      held_out_native.jsonl
      rubrics.jsonl
    manifests/
      build_manifest.json
```

## 16. Practical First Build

If only one short build can ship first, do this:

1. Pick 50 to 200 native TVL source documents
2. Clean and segment them
3. Build document registry entries
4. Generate 6 to 12 grounded task variants per document
5. Keep assistant answers source-derived and TVL-first
6. Build a small preference set focused on leakage and entity preservation
7. Hold out a document-level native eval slice

That is the smallest useful Stage C package.

## 17. Explicit Non-Goals

For the first Stage C pass, do not:

- translate large English capability corpora again as the main bet
- rely on one article template family
- treat all synthetic TVL as equal-quality
- push RL as the primary optimization path
- trust low-resource automatic judges without source-aware rubrics

## 18. Recommended Next Implementation Steps

1. Add a document-registry builder for native TVL sources.
2. Add segmentation and source-span tracking.
3. Add grounded task expansion for the required task families.
4. Add filters for English leakage, entity loss, and weak support.
5. Add preference-pair builders for language fidelity.
6. Add a held-out native eval builder.

Once those exist, Stage C becomes a reproducible data pipeline rather than an ad hoc prompting recipe.
