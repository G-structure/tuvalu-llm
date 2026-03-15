# Submission Strategy

## Core Framing

Lead with this:

`We built a full-stack pipeline for training, evaluating, serving, and continuously improving a specialized Tuvaluan model.`

Do not lead with this:

`We care about language preservation.`

The preservation angle matters, but for SemiAnalysis it should be the reason the project matters, not the first thing they hear.

## What Judges Are Likely To Reward

- technical specificity
- tight problem framing
- credible metrics
- visible systems work
- practical usefulness beyond the weekend

## What To Say Early

- largest Tuvaluan-English corpus we know of
- public-source corpus including Jehovah's Witnesses publications, dictionaries, and textbooks
- Tinker-trained Qwen3-30B-A3B-Base adaptation
- 30B total / 3B active
- live app collecting real feedback signals
- current overlap benchmark beats GPT-5.4 overall
- Hugging Face datasets and model cards are being published for direct inspection

## What To Avoid

- vague "AI for good" framing without technical substance
- unsupported "SOTA" claims without a benchmark qualifier
- calling the football site just a gimmick or game
- saying you already have a full RL loop if the current implementation is really a feedback-signal pipeline

## Safe Benchmark Language

Use this exact style:

`On the current shared Tuvaluan benchmark subset with 28 overlapping examples, our Stage B model scores chrF++ 41.8 versus GPT-5.4 at 36.1 overall, and leads on 6 of 7 task slices by chrF++.`

If you rerun a bigger overlap benchmark, replace the numbers everywhere at once.

## Recommended Titles

- `TalaFutipolo: A Live Data Flywheel for Tuvaluan LLMs`
- `Full-Stack Infrastructure for a Tuvaluan LLM`
- `Training, Evaluating, and Improving a Tuvaluan Model`
- `A Feedback Loop for Low-Resource Language Models`

## Best One-Sentence Positioning

`We built infrastructure for the opposite of giant general models: a specialized low-resource-language model with its own corpus pipeline, eval stack, live app, and feedback flywheel.`

## Best Judge Answer To "Why Does This Matter?"

`Because frontier models still leave small languages behind, and we can now show that a custom model with the right infrastructure can outperform them on real tasks.`

## Best Judge Answer To "Why Is This Infrastructure?"

`Because the hard part was not making one model response look good. The hard part was building the data, training, evaluation, serving, and signal-collection system that makes the model repeatably better.`

## Best Sponsor Answer To "Why Give You Credits?"

`More Tinker credits directly translate into larger cross-model benchmarks, more live feedback-driven tuning, and a stronger public proof that customized models can serve languages frontier systems still underserve.`

## Public Artifact Language

Use this exact style if the uploads are still running:

`At submission time, we are publishing the project artifacts to Hugging Face under FriezaForce. The cleaned dataset and Stage A model card are live now, while the raw aligned dataset and Stage B model card are in the current upload queue.`

If everything finishes before submission, switch to:

`We published the project artifacts to Hugging Face under FriezaForce, including cleaned and raw datasets plus Stage A and Stage B model cards.`

Current links:

- `https://huggingface.co/datasets/FriezaForce/tv2en-cleaned`
- `https://huggingface.co/datasets/FriezaForce/tv2en-raw-aligned`
- `https://huggingface.co/FriezaForce/tvl-en-llm-translation-stage-a`
- `https://huggingface.co/FriezaForce/tvl-en-llm-translation-stage-b-llama8b`
