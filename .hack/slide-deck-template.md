# TalaFutipolo
## A Live Data Flywheel for Tuvaluan LLMs

Speaker prompt: Open with the systems framing, not the social-impact framing.

---

# The Thesis

- Frontier models are not automatically best for low-resource languages
- A specialized model plus the right data and eval pipeline can win
- We built the full stack needed to prove it

Speaker prompt: "This is an AI infrastructure project disguised as a language app."

---

# The Problem

- Tuvaluan has roughly 11,000 speakers and almost no modern NLP tooling
- General-purpose models underperform because the data flywheel barely exists
- Low-resource-language work usually dies at the benchmark or dataset stage

Speaker prompt: Emphasize that the bottleneck is not only model quality. It is the entire pipeline.

---

# What We Built

- Corpus pipeline
- Tinker-based model adaptation
- Eval runner and dashboard
- Live product for feedback collection

Speaker prompt: Say "from raw text to user signals" in one sentence.

---

# Data Layer

- 342,505 raw Tuvaluan-English pairs
- 377,122 rendered training examples
- ~74.6M training tokens
- Decontaminated train / validation / test splits

Speaker prompt: This is the largest Tuvaluan-English corpus we know of. Mention that the public-source mix includes Jehovah's Witnesses publications, dictionaries, textbooks, and other bilingual material.

---

# Model Layer

- Base model: Qwen3-30B-A3B-Base
- 30B total parameters, 3B active
- LoRA fine-tuning on Thinking Machines' Tinker stack
- Stage A translation adapter plus Stage B capability model

Speaker prompt: The point is not giant scale. The point is efficient specialization.

---

# Offline Eval

- Held-out Stage A translation result: chrF++ 64.5
- Held-out Stage A translation result: BLEU 46.7
- Bidirectional Tuvaluan <-> English evaluation
- Extra held-out textbook set outside core training flow

Speaker prompt: These are not cherry-picked generations. They come from repeatable eval code in the repo.

---

# Cross-Model Comparison

- Current shared benchmark subset: 28 overlapping examples
- Our Stage B model: chrF++ 41.8
- GPT-5.4 on same overlap: chrF++ 36.1
- Our model wins 6 of 7 task slices by chrF++

Speaker prompt: Keep the "shared subset" qualifier. It makes the claim stronger, not weaker.

---

# Live Product

- Talafutipolo: Tuvaluan-first football news reader
- Sources: Goal.com, FIFA.com, Sky Sports
- Cloudflare Pages + D1 + automated ingestion pipeline
- Paragraph-level thumbs up / thumbs down plus reveal and share signals

Speaker prompt: The app is the data flywheel, not just the demo wrapper.

---

# Why Football

- Daily fresh content
- Strong community interest
- Natural way to collect preference signals without making users fill forms
- Practical path from passive reading to active model improvement

Speaker prompt: This is how we make post-training data collection sustainable.

---

# Why This Fits SemiAnalysis

- Full-stack AI infrastructure, not just an API app
- Training, eval, serving, feedback, and deployment in one system
- Clear resource-efficiency story around a specialized 3B-active model
- Concrete metrics instead of vague "AI for good" claims

Speaker prompt: Tie directly to their "dirt to decode" framing by showing every link in the chain.

---

# Why This Is Good PR For Thinking Machines

- Demonstrates Tinker on a hard, underrepresented language problem
- Shows custom-model infrastructure beating a frontier model on a real eval slice
- Connects fine-tuning infra to a live public product, not just an internal benchmark
- Makes a strong case for more credits producing visible public results

Speaker prompt: "This is a public proof point for customized models."

---

# What We Need Next

- More benchmark coverage against frontier models
- More live feedback volume
- Better ranking and filtering of high-value signals
- Extension of the same stack to additional low-resource languages

Speaker prompt: End with momentum. Credits and compute obviously convert into better results.

---

# Closing

- Specialized models can beat frontier general models
- Low-resource languages need infrastructure, not just goodwill
- We built a repeatable blueprint

Speaker prompt: Close on technical ambition first, impact second.
