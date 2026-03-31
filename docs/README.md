# Documentation Guide

This repo has both engineering docs and hackathon-facing materials. Start from the path that matches what you need.

## Start Here

### For judges or first-time readers

1. [../README.md](../README.md) for the high-level project story
2. [FOOTBALL_SETUP.md](FOOTBALL_SETUP.md) for the live app and feedback flywheel
3. [DATA_PIPELINE.md](DATA_PIPELINE.md) for the dataset and decontamination story
4. [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) for the model-training path

### For hackathon submission work

The local-only `.hack/` folder is the judge-facing source of truth for wording and demo framing:

- [../.hack/devpost.md](../.hack/devpost.md)
- [../.hack/demo-script.md](../.hack/demo-script.md)
- [../.hack/submission-strategy.md](../.hack/submission-strategy.md)
- [../.hack/slide-deck-template.md](../.hack/slide-deck-template.md)

Use those files for presentation copy. Use the docs in this folder for implementation and setup details.

## Core Technical Docs

| Doc | Scope |
|---|---|
| [DATA_PIPELINE.md](DATA_PIPELINE.md) | Full corpus pipeline: sources, cleaning, splits, rendered training data |
| [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) | Stage A and Stage B training/eval flow |
| [FOOTBALL_SETUP.md](FOOTBALL_SETUP.md) | Local football app setup, feedback loop, interaction export |
| [SCRAPING_PLAYBOOK.md](SCRAPING_PLAYBOOK.md) | Scraper reproduction details and operational notes |
| [UNSTRUCTURED_DATA_PIPELINE.md](UNSTRUCTURED_DATA_PIPELINE.md) | OCR and unstructured-source ingestion |
| [UNSTRUCTURED_DATA_SOURCES.md](UNSTRUCTURED_DATA_SOURCES.md) | Inventory of raw assets under `unstruct_lang_data/` and their current usage status |
| [STAGE_C_NATIVE_GROUNDING_SPEC.md](STAGE_C_NATIVE_GROUNDING_SPEC.md) | Concrete Stage C data spec for grounded native-TVL SFT, preferences, and evals |
| [STAGE_C_RESEARCH_PLAN.md](STAGE_C_RESEARCH_PLAN.md) | Literature-backed short-horizon Stage C plan based on the papers in `research/` |
| [SELECTIVE_TRANSLATION_SPEC.md](SELECTIVE_TRANSLATION_SPEC.md) | Rules for preserving code, JSON, and tool schemas during translation |
| [LOCAL_MLX_TRAINING.md](LOCAL_MLX_TRAINING.md) | Local MLX export and training path |

## Historical / Planning Docs

These are still useful, but they are not always exact descriptions of the live tree:

| Doc | Use |
|---|---|
| [CODEBASE_ORGANIZATION_PLAN.md](CODEBASE_ORGANIZATION_PLAN.md) | Reorg rationale and remaining cleanup roadmap |
| [DATASET_COLLECTION_AND_ML_PIPELINE.md](DATASET_COLLECTION_AND_ML_PIPELINE.md) | Earlier end-to-end planning notes |
| [TVL_EN_TINKER_PLAN.md](TVL_EN_TINKER_PLAN.md) | Original Tinker planning notes |
| [football_site_plan.md](football_site_plan.md) | Deep product and extraction notes for the football app |

## Script Inventory

For the current supported CLI surface, read [../scripts/README.md](../scripts/README.md).
