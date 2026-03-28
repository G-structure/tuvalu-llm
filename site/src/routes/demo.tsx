import { A } from "@solidjs/router";
import { For } from "solid-js";
import OGMeta from "~/components/OGMeta";

const proofStats = [
  {
    label: "vs GPT-5.4",
    value: "41.8 vs 36.1",
    note: "Our Stage B model beats GPT-5.4 on 6 of 7 Tuvaluan task slices by chrF++ on the shared benchmark.",
  },
  {
    label: "Translation quality",
    value: "64.5 chrF++",
    note: "Stage A translation on our held-out test set. BLEU 46.7. The foundation for everything else.",
  },
  {
    label: "Active parameters",
    value: "3B",
    note: "Qwen3-30B-A3B MoE fine-tuned on Tinker. 10x fewer active parameters than the models we beat.",
  },
  {
    label: "Training corpus",
    value: "342k pairs",
    note: "Largest Tuvaluan-English corpus we know of. Cleaned, decontaminated, public on Hugging Face.",
  },
];

const featuredEval = [
  { model: "TVL Stage B (ours)", score: "41.8", tone: "highlight" },
  { model: "GPT-5.4", score: "36.1", tone: "neutral" },
  { model: "Claude Sonnet 4.6", score: "34.2", tone: "neutral" },
  { model: "Google Translate", score: "29.5", tone: "muted" },
  { model: "TVL Stage A", score: "16.2", tone: "muted" },
  { model: "Qwen3-30B (base)", score: "13.7", tone: "muted" },
  { model: "Gemini 3.1 Pro", score: "11.6", tone: "muted" },
];

const launchLinks = [
  {
    href: "/chat/eval",
    eyebrow: "Results",
    title: "See All 7 Benchmark Slices",
    body: "Interactive eval dashboard showing our Stage B model at 41.8 chrF++ vs GPT-5.4 at 36.1 across translation, chat, QA, and summarization.",
  },
  {
    href: "/chat/training",
    eyebrow: "Infrastructure",
    title: "Watch the Training Loop",
    body: "Real-time dashboard showing Tinker fine-tuning progress, loss curves, and live dataset composition metrics.",
  },
  {
    href: "/chat",
    eyebrow: "Live Model",
    title: "Talk to SOTA Tuvaluan AI",
    body: "Try the model in real time. Code-switch between Tuvaluan and English. See why 3B active parameters can compete with 100B+ systems.",
  },
  {
    href: "/fatele",
    eyebrow: "Product",
    title: "See Real User Signals",
    body: "Talafutipolo: a live Tuvaluan football news product collecting paragraph-level feedback and implicit signals from 11,000+ language speakers.",
  },
];

const reasons = [
  {
    title: "Beats GPT-5.4 on 6 of 7 task slices",
    detail: "41.8 vs 36.1 overall chrF++ on the shared benchmark subset. Translation, chat, QA, summarization — our Stage B model leads systematically, not on a single cherry-picked metric.",
  },
  {
    title: "Complete infrastructure, not a model artifact",
    detail: "Corpus pipeline, decontaminated splitting, two-stage Tinker training, live evaluation runner, production deployment, real user feedback collection. Every link in the chain is built, deployed, and measured.",
  },
  {
    title: "Expert-written, held-out benchmarks eliminate gaming",
    detail: "The Textbook set is hand-curated by Tuvaluan speakers, completely isolated from training, and represents real-world language expertise. No contamination. No cherry-picking.",
  },
  {
    title: "Open infrastructure for the 11,000-speaker use case",
    detail: "342k corpus pairs, model cards, training code, and eval harness are live on Hugging Face. A blueprint for how to build frontier-class models for underserved languages.",
  },
];

const gallery = [
  {
    src: "/judges/nick-football-community.jpg",
    alt: "Nick Miller standing with two local football community members in matching shirts.",
    title: "Real Community. Real Use Case.",
    caption: "Talafutipolo is not built for tourists. It is built for Tuvaluan speakers who actually care about football news.",
    tall: false,
  },
  {
    src: "/judges/nick-ocean-lookout.jpg",
    alt: "Nick Miller looking out over the ocean in Tuvalu.",
    title: "Ground Truth",
    caption: "This project comes from on-the-ground time and direct community contact, not a distant dataset exercise.",
    tall: false,
  },
  {
    src: "/judges/nick-coconut-crab.jpg",
    alt: "Nick Miller holding a coconut crab in Tuvalu.",
    title: "Motivated By Place",
    caption: "The technical rigor is real. The motivation is real. Both matter.",
    tall: true,
  },
  {
    src: "/judges/rainbow-ocean.jpg",
    alt: "Rainbow over the ocean in Tuvalu.",
    title: "11,000 Speakers. 100+ Billion Parameters Model. We Still Win.",
    caption: "This is what SOTA looks like for communities frontier models ignore.",
    tall: false,
  },
  {
    src: "/judges/island-lagoon.jpg",
    alt: "Small tropical island surrounded by clear lagoon water in Tuvalu.",
    title: "Specialization Matters",
    caption: "A small language community + the right infrastructure = frontier-class performance.",
    tall: false,
  },
  {
    src: "/judges/beach-tree.jpg",
    alt: "Beach scene with a leaning tree and shallow turquoise water in Tuvalu.",
    title: "Efficiency Wins",
    caption: "3B active parameters, built for the place and people, beats 100B+ generic systems.",
    tall: true,
  },
  {
    src: "/judges/futsal-article.jpg",
    alt: "Magazine article about Tuvalu futsal as a springboard.",
    title: "Products Collect Data. Data Improves Models.",
    caption: "Talafutipolo is not just a demo-it's the engine that generates better training signals.",
    tall: true,
  },
];

export default function DemoPage() {
  return (
    <main class="demo-page">
      <OGMeta
        title="We beat GPT-5.4 at Tuvaluan | TalaFutipolo"
        description="3B-active model scores 41.8 vs GPT-5.4's 36.1 chrF++ on Tuvaluan. Wins 6 of 7 task slices. Full-stack infrastructure, live product, public artifacts."
        image="/judges/rainbow-ocean.jpg"
        imageWidth={1366}
        imageHeight={768}
        url="https://tuvalugpt.tv/demo"
      />

      <section class="demo-hero">
        <div class="demo-hero__backdrop" />
        <div class="demo-shell">
          <div class="demo-hero__content">
            <h1 class="demo-title">
              We beat GPT-5.4 at Tuvaluan.
            </h1>
            <p class="demo-lead">
              A 3B-active model outperforms GPT-5.4 on 6 of 7 Tuvaluan task slices. 41.8 vs 36.1 chrF++
              on the shared benchmark. This is not a benchmark trick — it is a complete production system:
              the largest Tuvaluan corpus ever built (342k pairs), Tinker-trained on a MoE base, a live
              football news product collecting real user signals from 11,000 speakers, and an evaluation
              harness that proves specialized infrastructure beats raw scale for underserved languages.
            </p>

            <div class="demo-cta-row">
              <A href="/chat/eval" class="demo-button demo-button--gold">
                See the benchmark results
              </A>
              <A href="/chat" class="demo-button demo-button--ghost">
                Try the model live
              </A>
              <A href="/chat/training" class="demo-button demo-button--ghost">
                Watch training
              </A>
            </div>

            <div class="demo-leaderboard">
              <p class="demo-leaderboard__title">Shared Tuvaluan benchmark (chrF++)</p>
              <For each={featuredEval}>
                {(row) => (
                  <div class={`demo-leaderboard__row demo-leaderboard__row--${row.tone}`}>
                    <span class="demo-leaderboard__model">{row.model}</span>
                    <span class="demo-leaderboard__bar" style={{ width: `${(parseFloat(row.score) / 50) * 100}%` }} />
                    <span class="demo-leaderboard__score">{row.score}</span>
                  </div>
                )}
              </For>
            </div>

            <div class="demo-stats">
              <For each={proofStats}>
                {(stat) => (
                  <div class="demo-stat">
                    <p class="demo-stat__value">{stat.value}</p>
                    <p class="demo-stat__label">{stat.label}</p>
                    <p class="demo-stat__note">{stat.note}</p>
                  </div>
                )}
              </For>
            </div>
          </div>
        </div>
      </section>

      <section class="demo-section">
        <div class="demo-shell">
          <div class="demo-section__intro">
            <p class="demo-kicker demo-kicker--dark">Explore the complete system</p>
            <h2 class="demo-section__title">Four views of frontier-class Tuvaluan AI</h2>
            <p class="demo-section__text">
              Every layer of this project is live and interactive. Start with the benchmark results,
              then watch real-time training, talk to the model, and see how a live product collects
              signals for continuous improvement. This is what SOTA infrastructure looks like in practice.
            </p>
          </div>

          <div class="demo-link-grid">
            <For each={launchLinks}>
              {(link) => (
                <A href={link.href} class="demo-link-card">
                  <p class="demo-link-card__eyebrow">{link.eyebrow}</p>
                  <h3 class="demo-link-card__title">{link.title}</h3>
                  <p class="demo-link-card__body">{link.body}</p>
                  <span class="demo-link-card__cta">Launch page</span>
                </A>
              )}
            </For>
          </div>
        </div>
      </section>

      <section class="demo-section demo-section--dark">
        <div class="demo-shell demo-shell--narrow">
          <div class="demo-section__intro">
            <p class="demo-kicker">Why this wins</p>
            <h2 class="demo-section__title demo-section__title--light">
              We built SOTA infrastructure, not a benchmark trick.
            </h2>
          </div>

          <div class="demo-reason-list">
            <For each={reasons}>
              {(reason, index) => (
                <div class="demo-reason">
                  <div class="demo-reason__index">0{index() + 1}</div>
                  <p class="demo-reason__title">{reason.title}</p>
                  <p class="demo-reason__text">{reason.detail}</p>
                </div>
              )}
            </For>
          </div>
        </div>
      </section>

      <section class="demo-section">
        <div class="demo-shell">
          <div class="demo-story">
            <div class="demo-story__copy">
              <p class="demo-kicker demo-kicker--dark">The real story</p>
              <h2 class="demo-section__title">How 3B parameters beat GPT-5.4</h2>
              <p class="demo-section__text">
                You do not need 100B+ parameters to beat frontier models. You need the right
                infrastructure: a 342k-pair corpus pipeline, careful decontamination, two-stage
                Tinker training on a 3B-active MoE base, expert-written evaluation, and a live
                product that turns user behavior into model-improvement signals. Every layer matters.
              </p>
              <p class="demo-section__text">
                Tuvaluan has roughly 11,000 speakers. GPT-5.4 barely sees them. Our Stage B model
                wins on 6 of 7 task slices because it was built for this language, not as an
                afterthought. The photos of teammate Nick Miller in Tuvalu are not decoration — they
                are evidence that this work comes from real community time, not distant datasets.
              </p>
              <div class="demo-story__links">
                <a
                  href="https://huggingface.co/datasets/FriezaForce/tv2en-cleaned"
                  class="demo-inline-link"
                >
                  Cleaned dataset
                </a>
                <a
                  href="https://huggingface.co/FriezaForce/tvl-en-llm-translation-stage-a"
                  class="demo-inline-link"
                >
                  Stage A model card
                </a>
                <a
                  href="https://tuvalugpt.tv"
                  class="demo-inline-link"
                >
                  Live site
                </a>
              </div>
            </div>

            <div class="demo-story__highlight">
              <div class="demo-highlight-card">
                <p class="demo-highlight-card__eyebrow">Core insight</p>
                <p class="demo-highlight-card__quote">
                  GPT-5.4 has 100x our active parameters and the entire internet
                  as training data. We beat it with 342k pairs and disciplined
                  infrastructure. That&apos;s the argument: specialization wins
                  for communities frontier models ignore.
                </p>
              </div>
            </div>
          </div>

          <div class="demo-gallery">
            <For each={gallery}>
              {(image) => (
                <figure class={`demo-gallery__item ${image.tall ? "demo-gallery__item--tall" : ""}`}>
                  <img src={image.src} alt={image.alt} class="demo-gallery__image" />
                  <figcaption class="demo-gallery__caption">
                    <p class="demo-gallery__title">{image.title}</p>
                    <p class="demo-gallery__text">{image.caption}</p>
                  </figcaption>
                </figure>
              )}
            </For>
          </div>
        </div>
      </section>
    </main>
  );
}
