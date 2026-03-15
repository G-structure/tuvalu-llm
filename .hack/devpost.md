# Devpost Draft

## Recommended Title

`TalaFutipolo: A Live Data Flywheel for Tuvaluan LLMs`

## Short Description

We built a full-stack pipeline for low-resource-language AI: the largest Tuvaluan-English corpus we know of, a Tinker-trained 3B-active Tuvaluan model, a live football news product that collects feedback signals, an eval stack where our current model beats GPT-5.4 on the shared Tuvaluan benchmark subset, and public Hugging Face artifacts for datasets and models.

## Tagline

Training, evaluating, serving, and continuously improving a specialized Tuvaluan model.

## Inspiration

Most AI infrastructure is optimized for giant general-purpose models, but low-resource languages need the opposite: compact, specialized systems that can be adapted quickly, evaluated transparently, and improved with real user feedback.

Tuvaluan has roughly 11,000 speakers and almost no modern NLP infrastructure. We wanted to prove that a custom model can serve a small language community better than a frontier general model, and that the full stack required to do it is now practical.

## What It Does

Our project has four layers:

1. We built the largest Tuvaluan-English corpus we know of, with 342,505 raw pairs and 377,122 rendered training examples.
2. We fine-tuned a Tuvaluan model on Thinking Machines' Tinker stack using Qwen3-30B-A3B-Base, a 30B-total / 3B-active MoE model.
3. We shipped a live Tuvaluan-first football news app, Talafutipolo, that serves translated content and collects paragraph-level user feedback and implicit interaction signals.
4. We built a benchmark and dashboard pipeline to compare our model against frontier systems on Tuvaluan tasks.

This is not just a model demo. It is an end-to-end workflow for creating and improving specialized models for low-resource languages.

## Why This Fits The Hackathon

This hackathon is about the full AI infrastructure stack, not just the API layer. Our project spans data acquisition, cleaning, decontaminated splitting, model adaptation, evaluation, deployment, and feedback collection.

Instead of asking how to make one giant model slightly better for everyone, we asked how to build infrastructure that makes a small custom model much better for a specific community. That is a different but equally important AI systems problem.

## How We Built It

We started by scraping and aligning Tuvaluan-English text from multiple public sources, including Jehovah's Witnesses publications, dictionaries, textbooks, and other bilingual material, then built a cleaning and split pipeline to reduce leakage and preserve held-out evaluation quality. The final Stage A translation dataset contains 377,122 rendered examples.

For training, we used Thinking Machines' Tinker API to LoRA fine-tune Qwen3-30B-A3B-Base. The resulting translation model reached chrF++ 64.5 and BLEU 46.7 on our decontaminated held-out translation test set.

On top of that, we built Talafutipolo, a Tuvaluan-first football news site on SolidStart and Cloudflare. It ingests articles from Goal.com, FIFA.com, and Sky Sports, translates them into Tuvaluan, and captures feedback such as thumbs up, thumbs down, reveals, shares, and island-tagged participation. That gives us a live path to collect post-training signals rather than stopping at a static benchmark.

We also built an evaluation runner and dashboard so we can compare our model to frontier systems on translation, textbook, chat, QA, and summarization tasks in Tuvaluan.

At submission time, we are also publishing the project artifacts to Hugging Face under the `FriezaForce` account so judges can inspect the data and model cards directly. The cleaned dataset and Stage A model card are already live, while the raw aligned dataset and Stage B model card are queued in the current upload run.

## Results

The strongest translation result we can defend today is:

- chrF++ 64.5 / BLEU 46.7 on our held-out Stage A translation test set

The strongest cross-model comparison we can defend today is:

- on the current shared benchmark subset with 28 overlapping examples, our Stage B model scores chrF++ 41.8 versus GPT-5.4 at 36.1 overall
- on that same shared subset, our model beats GPT-5.4 on 6 of 7 task slices by chrF++
- our model also substantially outperforms our own Stage A baseline on that subset, showing that the full low-resource adaptation pipeline matters

We are careful about claim scope here: the overlap benchmark is still small, so we present it as an early but real cross-model signal, not a universal claim over every possible Tuvaluan workload.

Public artifacts at submission time:

- cleaned dataset live: `https://huggingface.co/datasets/FriezaForce/tv2en-cleaned`
- raw aligned dataset upload in progress: `https://huggingface.co/datasets/FriezaForce/tv2en-raw-aligned`
- Stage A translation model card live: `https://huggingface.co/FriezaForce/tvl-en-llm-translation-stage-a`
- Stage B bilingual model card upload in progress: `https://huggingface.co/FriezaForce/tvl-en-llm-translation-stage-b-llama8b`

## Challenges We Ran Into

Low-resource language work has two hard problems at once: data scarcity and evaluation credibility.

On the data side, Tuvaluan sources are fragmented across websites, Jehovah's Witnesses publications, scanned PDFs, dictionaries, and mixed-format educational material. On the modeling side, it is easy to overclaim from anecdotal generations, so we invested heavily in decontamination, held-out test sets, and comparable benchmarks.

The product challenge was different: if the app felt like a data-collection form, no one would use it. The football format gave us a real reason for users to read, react, and generate useful feedback signals.

## Accomplishments That We're Proud Of

- We built what we believe is the best-performing Tuvaluan model we are aware of on our current benchmark suite.
- We turned a low-resource-language project into a full-stack systems project, not just a model checkpoint.
- We proved that a specialized 3B-active model can beat a much larger frontier model on a real low-resource-language eval slice.
- We created a live feedback loop that can keep improving the system after the hackathon.

## What We Learned

General-purpose models are not automatically the right answer for underserved languages. When the problem is narrow, the evaluation is careful, and the feedback loop is designed well, specialization wins.

We also learned that infrastructure matters as much as modeling. Without better data pipelines, eval pipelines, and user-facing collection loops, low-resource languages stay invisible to modern AI.

## Why Thinking Machines Should Care

This project is a strong proof point for Tinker and custom-model infrastructure.

We used Tinker to adapt a compact MoE model into a high-performing Tuvaluan system, then wrapped it in a real product and benchmark harness. That is exactly the kind of story that shows why fine-tuning infrastructure matters: not just for enterprise copilots, but for preserving languages that frontier general models leave behind.

Additional credits would convert directly into better public benchmarks, broader evaluation coverage, more feedback-driven tuning, and a stronger public case that customized models can serve communities the frontier stack misses.

## What's Next

- Expand the shared cross-model benchmark so the GPT-5.4 comparison covers a larger eval set
- Close the loop from live feedback into ranking, filtering, and post-training data generation
- Improve Stage B bilingual capability performance beyond translation into more general Tuvaluan assistant tasks
- Finish the current Hugging Face upload run and publish merged model weights for easier downstream use
- Use the same pipeline as a blueprint for additional low-resource languages

## Notes Before Submission

- Keep the `28 overlapping examples` qualifier on the GPT-5.4 comparison unless you rerun the larger benchmark and update the numbers.
- If you have a live demo URL and eval dashboard URL, put them near the top of the submission.
- If the raw dataset and Stage B uploads finish before you submit, change `upload in progress` to `live` everywhere at once.
- If you have a team name, replace the working title with that branding only if it stays technical.
