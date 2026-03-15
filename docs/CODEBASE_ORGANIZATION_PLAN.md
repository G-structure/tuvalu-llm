# Codebase Organization Plan

## Executive Summary

The repo already has one good organizational seam: `training/` is a real Python package with reusable code and tests. The main problem is that most non-training business logic still lives in `scripts/`, so the repo currently has:

- reusable library code in `training/`
- large operational pipelines in `scripts/`
- duplicated config flattening, IO helpers, data normalization, scraper logic, translation client logic, and eval flows

The recommendation is:

1. Keep `scripts/` as thin CLI entrypoints and orchestration only.
2. Move reusable non-training logic into a first-class package organized by domain.
3. Keep the current on-disk JSONL/JSON outputs stable while refactoring internals.
4. Consolidate three explicit ABIs:
   - scraping
   - bulk translation
   - evals
5. Treat football/news as an app-specific package, not part of the generic core.

This should be done incrementally, with legacy scripts preserved as wrappers until all automation moves over.

## What Is Duplicated Today

### 1. Language-specific JW scrapers are copied instead of parameterized

Examples:

- `scripts/scrape_sitemap.py`
- `scripts/scrape_sitemap_sm.py`
- `scripts/scrape_bible.py`
- `scripts/scrape_bible_sm.py`
- `scripts/scrape_daily_text.py`
- `scripts/scrape_daily_text_sm.py`
- `scripts/scrape_articles.py`
- `scripts/scrape_articles_sm.py`

Most of the logic is the same. The differences are mostly:

- locale bundle constants
- URL shapes
- regexes / category slugs
- a few traversal quirks

That is a configuration problem, not a new-script problem.

### 2. Stage A corpus prep is split across multiple implementations

There is overlapping logic across:

- `training/stage_a_mt/build_data.py`
- `scripts/render_training_data.py`
- `scripts/build_splits.py`
- `scripts/validate_splits.py`
- `scripts/clean_pipeline.py`

Examples of repeated behavior:

- `_stable_hash`
- text normalization
- split assignment
- template selection
- JSONL read/write helpers
- Stage A example rendering

`training/stage_a_mt/build_data.py` is already library-quality and should become the canonical owner of that logic. The scripts should not re-implement adjacent variants.

### 3. Config translation is repeated in wrappers

Examples:

- `scripts/build_stage_a_mt_data.py`
- `scripts/train_stage_a_translation.py`
- `scripts/eval_stage_a_translation.py`
- `scripts/build_stage_b_mix.py`
- `scripts/train_stage_b_agent.py`
- `scripts/eval_stage_b_agent.py`
- deprecated wrappers under `scripts/*tinker*`

These all do variations of:

- load config
- flatten / rename keys
- forward to `training.*.main(config)`

This should be one shared CLI/config layer.

### 4. Translation client logic exists in several places

Examples:

- `training/synthetic/generate.py`
- `scripts/translate_football.py`
- `scripts/benchmark_eval.py`
- `scripts/test_stage_b_live.py`

These currently duplicate:

- model client setup
- prompt building
- request batching / concurrency
- retry behavior
- parsing model outputs
- task-specific validation

### 5. Evals are separate systems instead of one runner

Examples:

- `training/stage_a_mt/eval.py`
- `training/stage_b_agent/eval.py`
- `scripts/benchmark_eval.py`

All three do the same broad workflow:

- load evaluation examples
- construct prompt/reference pairs
- call one or more model backends
- compute metrics
- write reports

The task types differ, but the execution model is shared.

### 6. Unstructured ingestion has overlapping generations of pipeline code

Examples:

- `scripts/ingest_new_unstruct.py`
- `scripts/build_unstructured_seed.py`
- `scripts/run_unstructured_datamining.py`
- `scripts/ocr_scanned_pdfs.py`
- `scripts/extract_bilingual_pdfs.py`
- `scripts/extract_grammar_pairs.py`
- `scripts/extract_species_names.py`

This area already has "old path" and "new path" behavior in parallel. It should become one package with extractors, OCR helpers, and a reproducible pipeline runner.

### 7. Football ETL duplicates storage behavior

Examples:

- `scripts/scrape_football_goal.py`
- `scripts/scrape_football_sky.py`
- `scripts/scrape_football_fifa.py`
- `scripts/translate_football.py`
- `scripts/init_football_db.py`
- `scripts/db_conn.py`
- `scripts/d1_client.py`
- `scripts/sync_to_d1.py`

The football app currently repeats:

- article insert SQL
- fetch log updates
- D1 vs SQLite branching
- translation persistence

This should be isolated behind a repository/storage layer, while preserving
explicit backend selection between local SQLite and Cloudflare D1.

### 8. Tests mostly cover the package code, not the script-heavy pipelines

`tests/` has good coverage for:

- `training.common`
- `training.stage_a_mt.build_data`
- `training.synthetic.selective_translate`
- `training.stage_b_agent.build_mix`
- CLI smoke for selected wrappers

But there is effectively no automated coverage for:

- JW scrapers
- tuvalu.aa-ken.jp scrapers
- football scrapers
- football translation pipeline
- unstructured ingestion pipelines
- benchmark eval path

That means the least organized parts are also the least protected.

## Design Principles

1. Scripts are wrappers, not homes for business logic.
2. Shared data contracts come before file moves.
3. Source/language differences are data, not copied modules.
4. Training code should only own training concerns.
5. Core libraries should be network-agnostic where possible.
6. App-specific systems stay separate from generic infrastructure.
7. Refactors should preserve current JSONL/JSON schemas unless there is a very good reason not to.
8. Backend abstraction should preserve runtime storage choice rather than hard-coding one persistence target.

## Recommended Target Layout

End state:

```text
tv/
  common/
    config.py
    io.py
    manifests.py
    types.py
    cli.py
  corpus/
    records.py
    clean.py
    splits.py
    render.py
    stage_a.py
    stage_b.py
    unstructured/
      ocr.py
      extractors.py
      pipeline.py
  scrape/
    http.py
    registry.py
    jw/
      specs.py
      sitemap.py
      bible.py
      daily_text.py
      articles.py
    tuvalu/
      app.py
      dictionary.py
  translate/
    core.py
    batching.py
    chunking.py
    validators.py
    selective.py
    jobs.py
    backends/
      tinker.py
      openrouter.py
  evals/
    core.py
    scorers.py
    reports.py
    tasks/
      translation.py
      preservation.py
      bilingual.py
      benchmark.py
  apps/
    football/
      models.py
      repository.py
      storage.py
      db.py
      translate.py
      sync.py
      scrape/
        goal.py
        sky.py
        fifa.py

training/
  common/
  stage_a_mt/
    train.py
    eval.py
    export.py
  stage_b_agent/
    train.py
    eval.py
  local_mlx/

scripts/
tests/
```

Important note: this is the target layout, not the first move. The first move should be extracting shared library code while keeping the current scripts as compatibility wrappers.

## What Should Stay in `training/`

`training/` should own:

- train loops
- model-specific runtime setup
- checkpoint/export behavior
- training-time evaluation entrypoints
- local MLX prep that is clearly training-run specific

`training/` should not own generic repo utilities forever. Today `training.common` is already being used outside training, which means it is misnamed. Those utilities should be promoted to `tv.common`, with compatibility imports left in `training.common` during migration.

Suggested steady state:

- keep:
  - `training/stage_a_mt/train.py`
  - `training/stage_a_mt/eval.py`
  - `training/stage_a_mt/export.py`
  - `training/stage_b_agent/train.py`
  - `training/stage_b_agent/eval.py`
  - `training/local_mlx/*`
- move out over time:
  - `training/stage_a_mt/build_data.py` -> `tv/corpus/stage_a.py`
  - `training/stage_b_agent/build_mix.py` -> `tv/corpus/stage_b.py`
  - `training/synthetic/selective_translate.py` -> `tv/translate/selective.py`
  - `training/synthetic/quality.py` -> `tv/translate/validators.py`
  - `training/synthetic/loaders.py` -> `tv/corpus/sources.py`
  - `training/synthetic/generate.py` -> `tv/translate/jobs.py`
  - `training/common/*` -> `tv/common/*`

## Canonical Data Contracts

Before reorganizing files, define explicit shared record types.

Minimum set:

### `AlignedPairRecord`

Canonical schema for scraped/aligned bilingual records.

Fields already exist de facto:

- `id`
- source language field (`tvl`, `sm`, etc.)
- target language field (`en`)
- `content_type`
- `domain`
- `alignment_method`
- `alignment_confidence`
- source metadata such as `doc_id`, `date`, `book_num`, `pub_code`

Recommendation:

- keep on-disk JSONL unchanged
- add one typed constructor / validator in `tv.corpus.records`
- normalize language fields via explicit metadata:
  - `source_lang`
  - `target_lang`

### `NormalizedExample`

This already exists informally via `training.common.schema.make_example`.

Use it as the canonical contract for:

- Stage A training examples
- Stage B examples
- selective translation inputs
- eval examples

### `TranslationRequest` / `TranslationResult`

Needed for the bulk translation ABI.

Should include:

- `id`
- `input_type` (`text`, `messages`, `article`)
- `source_lang`
- `target_lang`
- `payload`
- `mode` (`plain`, `selective`)
- `policy`
- `metadata`

Result should include:

- translated payload
- attempts
- validation signals
- backend info
- latency / token stats when available

### `EvalCase` / `EvalPrediction` / `EvalReport`

Needed for a reusable eval system.

Every eval should reduce to:

- prompt messages
- optional reference
- metadata
- scorer set

## ABI 1: Scraping

### Goal

Make it easy to scrape from JW in Tuvaluan, Samoan, English, or another locale without creating a new script per language.

### Recommended API

```python
@dataclass(frozen=True)
class LocaleSpec:
    code: str
    field_name: str
    rcode: str | None = None
    lpcode: str | None = None

@dataclass(frozen=True)
class ScrapeJob:
    source: str
    dataset: str
    src_locale: str
    tgt_locale: str = "en"
    output_path: Path
    resume: bool = True
    limit: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

class Scraper(Protocol):
    def discover(self, job: ScrapeJob) -> Iterable[Any]: ...
    def fetch(self, ref: Any, ctx: Any) -> Any: ...
    def extract(self, raw: Any, ctx: Any) -> list[AlignedPairRecord]: ...
```

### How this maps to current code

- `scripts/fetch.py` -> `tv.scrape.http`
- JW language bundles and regexes -> `tv.scrape.jw.specs`
- `scrape_bible*.py` -> one `tv.scrape.jw.bible` implementation parameterized by locale/version
- `scrape_daily_text*.py` -> one `tv.scrape.jw.daily_text`
- `scrape_articles*.py` -> one `tv.scrape.jw.articles`
- `scrape_sitemap*.py` -> one `tv.scrape.jw.sitemap`
- `scrape_tuvalu_app.py` and `scrape_tuvalu_dictionary.py` -> `tv.scrape.tuvalu.*`

### Key rule

Locale/site variation should live in `LocaleSpec` or source-specific config tables, not in copied modules.

### CLI shape after migration

Examples:

```bash
uv run scripts/scrape_jw_bible.py --src-locale tvl --tgt-locale en --version nwt --full
uv run scripts/scrape_jw_bible.py --src-locale sm --tgt-locale en --version bi12 --book 1
uv run scripts/scrape_jw_articles.py --src-locale tvl --scope library
uv run scripts/scrape_jw_articles.py --src-locale sm --scope library --category tusi
```

Legacy scripts can continue to exist as wrappers that call the generic implementation with hard-coded locale defaults.

## ABI 2: Bulk Translation

### Goal

Have one reusable interface for:

- selective Stage B synthetic generation
- football article translation
- ad hoc batch translation jobs
- benchmark/eval model calls

### Recommended API

```python
@dataclass
class TranslationPolicy:
    mode: Literal["plain", "selective"]
    chunking: str
    retry_schedule: list[dict[str, Any]]
    validate_collapse: bool = False
    validate_structure: bool = False

class TranslationBackend(Protocol):
    def translate_batch(
        self,
        requests: list[TranslationRequest],
    ) -> list[TranslationResult]: ...

class TranslationJob(Protocol):
    def load_requests(self) -> list[TranslationRequest]: ...
    def save_results(self, results: list[TranslationResult]) -> None: ...
```

### Where current code should land

- Tinker client setup from `training.common.tinker_runtime` -> backend adapter
- selective masking from `training/synthetic/selective_translate.py` -> `tv.translate.selective`
- preservation and collapse checks from `training/synthetic/quality.py` and `scripts/detect_collapse.py` -> `tv.translate.validators`
- paragraph splitting / chunking from `scripts/translate_football.py` -> `tv.translate.chunking`
- batch/fire-all concurrency from `training/synthetic/generate.py` -> `tv.translate.batching`

### Important separation

Separate:

- backend concerns
  - model path
  - sampling params
  - API retries
- job concerns
  - articles vs examples vs plain text
- validation concerns
  - placeholder leak
  - JSON preservation
  - code preservation
  - collapse detection

That keeps football translation and synthetic data generation on the same engine without forcing them into the same job code.

## ABI 3: Evals

### Goal

Make Stage A eval, Stage B eval, and benchmark eval all use the same runner shape.

### Recommended API

```python
@dataclass
class EvalCase:
    id: str
    prompt_messages: list[dict[str, str]]
    reference: str | None
    metadata: dict[str, Any]

class EvalTask(Protocol):
    name: str
    def load_cases(self) -> list[EvalCase]: ...
    def score(self, predictions: list[EvalPrediction]) -> dict[str, Any]: ...

class ModelAdapter(Protocol):
    def generate(self, cases: list[EvalCase], config: dict[str, Any]) -> list[EvalPrediction]: ...
```

### Shared flow

Every eval should follow:

1. load cases
2. generate predictions
3. score
4. write predictions
5. write report
6. write manifest

### Where current code should land

- metric helpers from `training/common/metrics.py` stay shared
- prompt/reference extraction from `training/stage_b_agent/eval.py` becomes generic eval helper
- benchmark model adapters from `scripts/benchmark_eval.py` become `ModelAdapter`s
- translation regression, preservation, bilingual comparison become `EvalTask`s

### Result

Adding a new eval should mean:

- add one task module
- register it
- no new custom runner script

## Folder-by-Folder Recommendations

### `scripts/`

Target state:

- wrappers only
- argument parsing only
- config loading only
- call one library entrypoint
- print summary / exit code

Rules:

- no `sys.path.insert(...)`
- no duplicated JSONL helpers
- no duplicated config flattening logic
- no direct DB SQL except in dedicated app repositories
- no direct model client setup except in library backends

### `training/`

Target state:

- model training runtime
- model eval runtime
- checkpointing/export
- MLX preparation

Non-training dataset build logic should migrate out over time.

### `tests/`

Recommended structure:

```text
tests/
  unit/
  integration/
  contracts/
  cli/
  fixtures/
    jw/
    tuvalu/
    football/
    evals/
```

Add:

- contract tests for `AlignedPairRecord`, `NormalizedExample`, `TranslationRequest`, `EvalCase`
- scraper fixture tests using saved HTML/XML/JSON
- integration tests for JW locale specs
- translation job tests with fake backends
- eval task tests with a fake model adapter

Current gap to fix first:

- scraping
- football ETL
- unstructured ingestion

### `apps/football/`

Target state:

- app-specific article models and translation flows
- a repository layer that owns queries and persistence semantics
- a storage backend interface with at least:
  - local SQLite
  - Cloudflare D1

Recommended shape:

```python
class FootballStorage(Protocol):
    def insert_article(self, article: ArticleRecord) -> bool: ...
    def update_source_stats(self, source_id: str, article_count: int) -> None: ...
    def log_fetch(self, event: FetchLogRecord) -> None: ...
    def save_translation(self, translation: TranslationRecord) -> None: ...
    def save_translation_attempt(self, attempt: TranslationAttemptRecord) -> None: ...
```

Then:

- `SqliteFootballStorage` implements that contract
- `D1FootballStorage` implements that contract
- `FootballRepository` contains the app-facing methods and delegates to the selected storage backend

Selection should remain explicit and cheap:

- env var, config value, or CLI flag selects backend
- local development can keep using SQLite
- CI / production can keep using D1
- scraper and translation jobs should not care which backend is active

## Concrete Migration Mapping

High-value moves:

- `scripts/fetch.py` -> `tv/scrape/http.py`
- `scripts/clean_pipeline.py` -> `tv/corpus/clean.py`
- `scripts/build_splits.py` + `scripts/validate_splits.py` -> `tv/corpus/splits.py`
- `scripts/render_training_data.py` + `training/stage_a_mt/build_data.py` -> `tv/corpus/stage_a.py`
- `scripts/build_stage_b_sources.py` -> `tv/corpus/sources.py`
- `scripts/build_crosslingual_data.py` -> `tv/corpus/stage_b_crosslingual.py`
- `training/stage_b_agent/build_mix.py` -> `tv/corpus/stage_b.py`
- `training/synthetic/generate.py` + `scripts/translate_football.py` -> `tv/translate/jobs.py`
- `scripts/benchmark_eval.py` + stage eval modules -> `tv/evals/*`
- football DB/storage modules -> `tv/apps/football/*`

## Recommended Migration Order

### Phase 1: Establish shared foundations

- create `tv.common`
- move generic config/io/manifest helpers there
- keep compatibility imports in `training.common`
- create one shared CLI helper for config loading and override merging

This is the lowest-risk first move.

### Phase 2: Unify scraping

- create `tv.scrape.http`
- create `tv.scrape.jw.specs`
- merge `*_sm.py` and non-`_sm.py` JW scripts behind locale specs
- keep existing scripts as wrappers

Success criterion:

- adding a new JW locale does not require copying a script

### Phase 3: Unify corpus/data prep

- make `AlignedPairRecord` explicit
- merge split/render/build logic into `tv.corpus`
- choose one unstructured pipeline path and retire the other

Success criterion:

- there is one owner for normalization, one owner for split logic, one owner for Stage A rendering

### Phase 4: Unify translation

- introduce `TranslationBackend`
- move selective translation, chunking, retry, validation into `tv.translate`
- rewrite synthetic generation and football translation to use the same backend + policy layer

Success criterion:

- backend setup exists in one place
- validation policies are reusable

### Phase 5: Unify evals

- introduce `EvalTask`, `ModelAdapter`, `EvalRunner`
- port Stage A eval, Stage B eval, benchmark eval

Success criterion:

- new evals are task modules, not new standalone systems

### Phase 6: Reduce wrapper/dead-script debt

- deprecate copied wrappers
- shorten scripts to thin entrypoints
- remove any remaining duplicated helpers

For football specifically, this phase should end with one backend selection path,
not backend removal.

## Guardrails

Do not do these during the first refactor wave:

- do not change the on-disk data schema and module structure in the same PR
- do not rewrite all scripts into one mega-pipeline
- do not move football-specific code into the generic translation package
- do not delete old CLI names until automation and docs are updated
- do not introduce a heavy framework just to model records

Prefer:

- plain dataclasses / TypedDicts
- explicit registry tables
- small adapters
- compatibility wrappers
- explicit backend selection for storage-backed apps

## Immediate Next Refactors I Would Prioritize

1. Promote `training.common` to a shared `tv.common` package.
2. Build a generic JW scraper core with locale specs.
3. Unify translation backends and policies.
4. Consolidate Stage A dataset prep under one package owner.
5. Create a shared eval runner.
6. Split football app code behind a repository/storage layer.
7. Backfill tests for scrapers and football pipelines.

## Definition of Done for the Reorg

The reorg is working when:

- new scraping support for a locale is a spec/config addition, not a copied script
- `scripts/` contains almost no business logic
- there is one bulk translation engine used by both synthetic generation and football translation
- there is one eval runner used by Stage A, Stage B, and benchmark tasks
- `training/` no longer acts as the repo-wide home for generic utilities
- core pipeline code is covered by unit/integration tests with offline fixtures
- football jobs can still run against either local SQLite or Cloudflare D1 by configuration
