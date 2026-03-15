# tv2en / TalaFutipolo

Low-resource-language AI infrastructure for Tuvaluan: corpus collection, decontaminated training data, Tinker-based model adaptation, evaluation, and a live football product that turns usage into future training signals.

Tuvaluan (ISO 639-3: `tvl`) has roughly 11,000 speakers and almost no modern NLP tooling. This repo is the full system we built to show that a specialized model can outperform frontier general models on a narrow low-resource-language workload when the surrounding data, eval, and product loop are built carefully.

## Why This Repo Matters

- We built the largest Tuvaluan-English corpus we know of: `342,505` raw aligned pairs and `377,122` rendered training examples.
- We fine-tuned a Tuvaluan translation model on Thinking Machines' Tinker stack using `Qwen3-30B-A3B-Base`, a `30B` total / `3B` active MoE model.
- We shipped a live Tuvaluan-first football news app at [talafutipolo.pages.dev](https://talafutipolo.pages.dev).
- The app collects explicit feedback, correction text, mode preferences, and implicit engagement signals, then exports them as normalized JSONL for future post-training.
- On the current shared Tuvaluan benchmark subset with `28` overlapping examples, the Stage B model scores `chrF++ 41.8` versus `GPT-5.4` at `36.1` overall and leads on `6/7` task slices by chrF++.

This is not just a model checkpoint. It is a full pipeline for training, evaluating, serving, and continuously improving a specialized model.

## What Is Real Today

| Layer | Status | What exists in the repo |
|---|---|---|
| Corpus pipeline | Live | Scrapers, cleaning, leak-resistant splits, renderers |
| Stage A translation model | Live | Training, eval, export, local MLX prep |
| Stage B bilingual stack | Partially built | Source normalization, synthetic translation, mixed-data builder, eval |
| Live product | Live | SolidStart football site, Cloudflare/SQLite storage, translation pipeline |
| Feedback flywheel | Live | Article feedback UI, community dashboard, normalized interaction export |
| Public artifacts | In progress / partially live | Hugging Face dataset and model-card links below |

The important claim boundary for judges: the live feedback-signal loop is real today; a fully closed RL/reward-training loop from those signals is the next step, not something this README pretends is already finished.

## Public Results

### Stage A translation

| Metric | All | EN→TVL | TVL→EN |
|---|---|---|---|
| chrF++ | `64.5` | `68.2` | `59.9` |
| BLEU | `46.7` | `49.9` | `42.1` |

### Cross-model comparison

Use this wording carefully:

> On the current shared Tuvaluan benchmark subset with 28 overlapping examples, our Stage B model scores chrF++ 41.8 versus GPT-5.4 at 36.1 overall, and leads on 6 of 7 task slices by chrF++.

That is an early but real benchmark signal, not a blanket claim about every Tuvaluan workload.

## Public Artifacts

### Hugging Face status

Artifacts are being published under the `FriezaForce` account:

- Profile: `https://huggingface.co/FriezaForce`
- Datasets: `https://huggingface.co/datasets/FriezaForce`

Current status:

- `tv2en-cleaned` (`182 MB`) is live with `178,371` cleaned translation pairs:
  `https://huggingface.co/datasets/FriezaForce/tv2en-cleaned`
- `tv2en-raw-aligned` (`690 MB`) is queued / uploading next with about `310K` raw parallel pairs:
  `https://huggingface.co/datasets/FriezaForce/tv2en-raw-aligned`
- Stage A model card is live for the final Qwen `30B` translation checkpoint at step `3546`, with `chrF++ 64.49` and `BLEU 46.74`:
  `https://huggingface.co/FriezaForce/tvl-en-llm-translation-stage-a`
- Stage B bilingual model card is queued / uploading next for the Llama `8B` checkpoint at step `18000`:
  `https://huggingface.co/FriezaForce/tvl-en-llm-translation-stage-b-llama8b`

What this yields for the hackathon submission:

- `2` public datasets: cleaned plus raw aligned
- `2` model cards: Stage A translation plus Stage B bilingual
- roughly `490K` total pairs across the published dataset surfaces
- documented training/eval context rather than just raw files

If an upload is still running when you show the project, keep that qualifier explicit rather than claiming every artifact is already fully live.

## Repository Map

```text
scripts/                      # Thin CLI entrypoints and orchestration
tv/
  common/                     # Shared config, IO, metrics, CLI helpers
  corpus/                     # Cleaning, split-building, rendering
  training/                   # Stage A, Stage B, synthetic data, export
  apps/
    football/                 # Football repository, storage, export logic
site/                         # SolidStart football app
configs/                      # Training and eval configs
tests/                        # Repo-local tests
docs/                         # Technical documentation and setup guides
.hack/                        # Hackathon copy, demo script, submission notes
```

## Fastest Demo Path

### 1. Install dependencies

```bash
uv sync
cd site && npm install && cd ..
```

### 2. Initialize and populate the football database

```bash
uv run scripts/init_football_db.py
uv run scripts/pipeline_football.py --scrape-limit 10 --translate-limit 10
```

### 3. Run the site

```bash
cd site
npm run dev
```

Open `http://localhost:3000`.

### 4. Show the feedback flywheel

1. Open a translated article.
2. Use the `Coach the Translator` form and paragraph-level voting.
3. Visit `/fatele` to show community totals.
4. Export the collected signals:

```bash
uv run scripts/export_football_interactions.py
```

By default this writes normalized JSONL artifacts under `data/football/exports/interactions/`.

## Core Workflows

### Corpus pipeline

```bash
uv run scripts/scrape_bible.py --full
uv run scripts/scrape_articles.py --library
uv run scripts/scrape_daily_text.py --range 2017-01-01 2025-12-31
uv run scripts/clean_pipeline.py
uv run scripts/build_splits.py
uv run scripts/validate_splits.py
uv run scripts/render_training_data.py --include-unstructured
```

### Stage A training and eval

```bash
bash scripts/bootstrap_tinker.sh
uv run scripts/train_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base.json
uv run scripts/eval_stage_a_translation.py --config configs/stage_a_translation_qwen30b_base.json --parallel 64
```

### Stage B data path

```bash
uv run scripts/build_stage_b_sources.py --config configs/synthetic_stage_b_core.json
uv run scripts/generate_stage_b_synthetic_tvl.py --config configs/synthetic_stage_b_core.json
uv run scripts/build_stage_b_mix.py --config configs/stage_b_agent_qwen30b.json
```

## Documentation

Start with [docs/README.md](docs/README.md) for the curated doc index.

| Doc | Why you would read it |
|---|---|
| [docs/README.md](docs/README.md) | Best entrypoint into the repo docs |
| [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | Corpus sources, cleaning, splits, rendered dataset |
| [docs/TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md) | Stage A / Stage B training and eval flow |
| [docs/FOOTBALL_SETUP.md](docs/FOOTBALL_SETUP.md) | Local demo, football app, interaction export |
| [scripts/README.md](scripts/README.md) | Current CLI inventory and usage |
| [.hack/README.md](.hack/README.md) | Hackathon-only materials, pitch copy, demo script |

## Technical Notes

### Why `curl-impersonate`?

JW.org blocks standard HTTP clients with TLS fingerprinting. The repo uses Docker `curl-impersonate` via `scripts/fetch.py` to make the corpus scrapers reproducible.

### Why football?

Football gives the project a live, recurring content loop with natural user engagement. That makes it a practical way to collect translation preferences and correction signals without turning the product into a survey form.

### What is still incomplete?

- A fully closed reward-training or preference-optimization loop from the live football signals
- Broader cross-model benchmark coverage beyond the current shared subset
- Further cleanup of older unstructured-data scripts into a more unified package shape

## Verification

Current repo-local test suite:

```bash
uv run pytest tests/
```

Recent cleanup passes in this repo were verified against the local suite, including namespace compatibility and football interaction export coverage.

## License

Code and derived dataset assets live in this repository. Parallel text is derived from publicly available JW.org and other public bilingual sources documented in [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md).
