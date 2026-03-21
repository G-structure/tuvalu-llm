# Demo Script

## 30-Second Pitch

Most AI infrastructure is built for giant general-purpose models. We built the opposite: a full-stack pipeline for a specialized low-resource-language model. We created the largest Tuvaluan-English corpus we know of, used Thinking Machines' Tinker stack to adapt a 30B-total / 3B-active model, and shipped a live Tuvaluan football news product that collects real feedback signals. On our current shared Tuvaluan benchmark subset, our model beats GPT-5.4 overall and wins 6 of 7 task slices. This proves that custom models can outperform frontier models when the infrastructure is built for the community you actually want to serve.

## 60-Second Deep Dive

This project is evidence for why customization infrastructure matters. We used Tinker to turn a compact MoE base model into what we believe is the strongest Tuvaluan model we are aware of, then connected it to a live product and an eval dashboard. Instead of stopping at a checkpoint, we built the whole loop: data ingestion, decontaminated training, evaluation, serving, user feedback, and public artifact release through Hugging Face datasets and model cards. This is the kind of infrastructure story that shows custom-model tooling can unlock real performance gains for languages that giant general-purpose systems still underserve.

## 10-Second Intro

We built a Tinker-trained Tuvaluan model, a live feedback app, and a benchmark stack showing that a specialized 3B-active system can beat GPT-5.4 on our current shared Tuvaluan eval slice.

## Common Q&A

**Why infrastructure and not just a model?**
The novelty is the full system for building, evaluating, serving, and continuously improving a low-resource-language model in production. That is harder than training a single checkpoint.
