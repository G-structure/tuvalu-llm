import { A } from "@solidjs/router";
import { For } from "solid-js";
import OGMeta from "~/components/OGMeta";

const proofStats = [
  {
    label: "Raw corpus pairs",
    value: "342,505",
    note: "Largest Tuvaluan-English corpus we know of",
  },
  {
    label: "Rendered training examples",
    value: "377,122",
    note: "Decontaminated and ready for model adaptation",
  },
  {
    label: "Stage A translation result",
    value: "64.5 chrF++",
    note: "46.7 BLEU on held-out translation test",
  },
  {
    label: "Headline eval slice",
    value: "42.5 chrF++",
    note: "Textbook TVL to EN: Stage B is 0.1 behind Claude Sonnet and ahead of GPT-5.4",
  },
];

const featuredEval = [
  { model: "Claude Sonnet", score: "42.6", tone: "neutral" },
  { model: "TVL (Stage B)", score: "42.5", tone: "highlight" },
  { model: "GPT-5.4", score: "41.8", tone: "neutral" },
  { model: "Google Translate", score: "29.5", tone: "muted" },
  { model: "TVL Stage A", score: "16.2", tone: "muted" },
  { model: "Qwen3-30B", score: "13.7", tone: "muted" },
  { model: "Gemini 3.1 Pro", score: "11.6", tone: "muted" },
];

const launchLinks = [
  {
    href: "/chat/eval",
    eyebrow: "Proof",
    title: "Open the Eval Dashboard",
    body: "Show the head-to-head benchmark slices and the current GPT-5.4 comparison.",
  },
  {
    href: "/chat/training",
    eyebrow: "Training",
    title: "Open the Live Training Page",
    body: "Walk judges through the current adapter progress, losses, and dataset mix.",
  },
  {
    href: "/chat",
    eyebrow: "Product",
    title: "Open the Chat Experience",
    body: "Let judges talk directly to the bilingual model in Tuvaluan and English.",
  },
  {
    href: "/fatele",
    eyebrow: "Signals",
    title: "Open Community Feedback",
    body: "Show that the product already captures real human signals instead of stopping at a static benchmark.",
  },
];

const reasons = [
  "This is not a single model screenshot. It is a full stack: scraping, cleaning, decontamination, training, evaluation, deployment, and feedback collection.",
  "The project targets a real blind spot in modern AI. Tuvaluan has roughly 11,000 speakers and almost no modern NLP infrastructure.",
  "We can defend the numbers. On the Textbook TVL to EN slice, Stage B scores 42.5 chrF++, essentially tied for first, above GPT-5.4, and dramatically above generic translation baselines.",
  "The product loop matters. Talafutipolo translates live football news into Tuvaluan and turns reading behavior into future model-improvement signals.",
];

const gallery = [
  {
    src: "/judges/nick-football-community.jpg",
    alt: "Nick Miller standing with two local football community members in matching shirts.",
    title: "Nick With The Football Community",
    caption: "The product is grounded in the actual football culture the site is built around.",
    tall: false,
  },
  {
    src: "/judges/nick-ocean-lookout.jpg",
    alt: "Nick Miller looking out over the ocean in Tuvalu.",
    title: "Nick In Tuvalu",
    caption: "This project comes from real on-the-ground time, not a distant dataset exercise.",
    tall: false,
  },
  {
    src: "/judges/nick-coconut-crab.jpg",
    alt: "Nick Miller holding a coconut crab in Tuvalu.",
    title: "Field Notes, Not Just Benchmarks",
    caption: "The story is technical, but the motivation is direct contact with place and community.",
    tall: true,
  },
  {
    src: "/judges/rainbow-ocean.jpg",
    alt: "Rainbow over the ocean in Tuvalu.",
    title: "Tuvalu Atmosphere",
    caption: "The demo page should feel like Tuvalu, not a generic ML dashboard.",
    tall: false,
  },
  {
    src: "/judges/island-lagoon.jpg",
    alt: "Small tropical island surrounded by clear lagoon water in Tuvalu.",
    title: "Low-Resource, High-Leverage",
    caption: "A small language community can still justify world-class infrastructure.",
    tall: false,
  },
  {
    src: "/judges/beach-tree.jpg",
    alt: "Beach scene with a leaning tree and shallow turquoise water in Tuvalu.",
    title: "Built For A Specific Place",
    caption: "Specialization beats generic scale when the system is tuned to the real use case.",
    tall: true,
  },
  {
    src: "/judges/futsal-article.jpg",
    alt: "Magazine article about Tuvalu futsal as a springboard.",
    title: "Football Is The Right Product Surface",
    caption: "Football creates a natural reason for readers to engage, react, and teach the model.",
    tall: true,
  },
];

export default function DemoPage() {
  return (
    <main class="demo-page">
      <OGMeta
        title="Judge Demo"
        description="A judge-facing Talafutipolo demo page for the Tuvaluan model project."
        image="/judges/rainbow-ocean.jpg"
        imageWidth={1366}
        imageHeight={768}
        url="https://talafutipolo.pages.dev/demo"
      />

      <section class="demo-hero">
        <div class="demo-hero__backdrop" />
        <div class="demo-shell">
          <div class="demo-hero__content">
            <p class="demo-kicker">SemiAnalysis x Fluidstack Hackathon</p>
            <h1 class="demo-title">
              Talafutipolo is a live data flywheel for Tuvaluan AI.
            </h1>
            <p class="demo-lead">
              We built the opposite of a generic frontier demo: a full-stack,
              low-resource-language system with its own corpus pipeline, model
              training loop, live football product, and public evaluation
              evidence.
            </p>

            <div class="demo-cta-row">
              <A href="/chat/eval" class="demo-button demo-button--gold">
                View eval first
              </A>
              <A href="/chat/training" class="demo-button demo-button--ghost">
                Watch training
              </A>
              <A href="/chat" class="demo-button demo-button--ghost">
                Try the chat
              </A>
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
            <p class="demo-kicker demo-kicker--dark">How to judge this project</p>
            <h2 class="demo-section__title">Three clicks to the proof</h2>
            <p class="demo-section__text">
              Start with evaluation, move to training, then end on the live
              product. The core claim is simple: customized infrastructure can
              outperform giant general models for languages the default stack
              ignores.
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

          <div class="demo-featured-eval">
            <div class="demo-featured-eval__copy">
              <p class="demo-kicker demo-kicker--dark">Eval to lead with</p>
              <h3 class="demo-featured-eval__title">
                Textbook TVL to EN is the most persuasive benchmark slice.
              </h3>
              <p class="demo-section__text">
                This is the comparison to put in front of judges first. Our
                Stage B model reaches 42.5 chrF++, just 0.1 behind Claude
                Sonnet, above GPT-5.4, and far above Google Translate, Stage A,
                Qwen3-30B, and Gemini 3.1 Pro.
              </p>
            </div>

            <div class="demo-featured-eval__table">
              <div class="demo-table">
                <div class="demo-table__head">Textbook TVL → EN (chrF++)</div>
                <For each={featuredEval}>
                  {(row) => (
                    <div class={`demo-table__row demo-table__row--${row.tone}`}>
                      <span>{row.model}</span>
                      <strong>{row.score}</strong>
                    </div>
                  )}
                </For>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="demo-section demo-section--dark">
        <div class="demo-shell demo-shell--narrow">
          <div class="demo-section__intro">
            <p class="demo-kicker">Why this should win</p>
            <h2 class="demo-section__title demo-section__title--light">
              The project is technical, credible, and useful after the weekend.
            </h2>
          </div>

          <div class="demo-reason-list">
            <For each={reasons}>
              {(reason, index) => (
                <div class="demo-reason">
                  <div class="demo-reason__index">0{index() + 1}</div>
                  <p class="demo-reason__text">{reason}</p>
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
              <p class="demo-kicker demo-kicker--dark">Project story</p>
              <h2 class="demo-section__title">A language model built for a real community</h2>
              <p class="demo-section__text">
                Talafutipolo started with a hard systems question: how do you
                build modern language infrastructure for a community that
                frontier models barely see? Our answer was to combine a public
                corpus pipeline, Tinker-based fine-tuning, a live football news
                product, and a feedback loop that keeps generating better data.
              </p>
              <p class="demo-section__text">
                These photos of teammate Nick Miller in Tuvalu help make the
                point visually. The project is not abstract. It comes from real
                place, real people, and a product surface that actually makes
                sense for readers.
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
                  href="https://talafutipolo.pages.dev"
                  class="demo-inline-link"
                >
                  Live site
                </a>
              </div>
            </div>

            <div class="demo-story__highlight">
              <div class="demo-highlight-card">
                <p class="demo-highlight-card__eyebrow">Judge soundbite</p>
                <p class="demo-highlight-card__quote">
                  “The hard part was not making one answer look good. The hard
                  part was building the data, training, evaluation, serving, and
                  signal-collection system that makes the model repeatably
                  better.”
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
