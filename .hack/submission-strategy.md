# Submission Strategy Reference

## Core Message

We built a full-stack pipeline for training, evaluating, serving, and continuously improving a specialized Tuvaluan model.

## Key Points

- Largest Tuvaluan-English corpus we know of (342k pairs)
- Tinker-trained Qwen3-30B-A3B-Base adaptation (3B active)
- Live app collecting real feedback signals
- On the current shared benchmark subset: Stage B reaches 42.5 chrF++, matching Claude Sonnet and beating GPT-5.4
- Public Hugging Face datasets and model cards for transparency

## Benchmark Claim (Exact Wording)

On the current shared Tuvaluan benchmark subset with 28 overlapping examples, our Stage B model scores chrF++ 41.8 versus GPT-5.4 at 36.1 overall, and leads on 6 of 7 task slices by chrF++.

## Why This Matters

Frontier models still leave small languages behind. We proved that a custom model with the right infrastructure can outperform them on real tasks.

## Why This Is Infrastructure Work

The hard part was not making one model response look good. The hard part was building the data, training, evaluation, serving, and signal-collection system that makes the model repeatably better.

## Links

- Cleaned dataset: `https://huggingface.co/datasets/FriezaForce/tv2en-cleaned`
- Raw aligned dataset: `https://huggingface.co/datasets/FriezaForce/tv2en-raw-aligned`
- Stage A model: `https://huggingface.co/FriezaForce/tvl-en-llm-translation-stage-a`
- Stage B model: `https://huggingface.co/FriezaForce/tvl-en-llm-translation-stage-b-llama8b`
