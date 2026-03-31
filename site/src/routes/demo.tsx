import { A } from "@solidjs/router";
import { createSignal, For, onCleanup, onMount, Show } from "solid-js";
import OGMeta from "~/components/OGMeta";
import { absoluteUrl } from "~/lib/site";

// ─── Data ───

const featuredEval = [
  { model: "TVL Stage B (ours)", score: "41.8", tone: "highlight" },
  { model: "GPT-5.4", score: "36.1", tone: "neutral" },
  { model: "Claude Sonnet 4.6", score: "34.2", tone: "neutral" },
  { model: "Google Translate", score: "29.5", tone: "muted" },
  { model: "Qwen3-30B (base)", score: "13.7", tone: "muted" },
  { model: "Gemini 3.1 Pro", score: "11.6", tone: "muted" },
];

const milestones = [
  { value: "342k", label: "parallel pairs", detail: "Largest Tuvaluan-English corpus ever assembled" },
  { value: "3B", label: "active params", detail: "MoE fine-tuned on Tinker, 10x smaller than what we beat" },
  { value: "6/7", label: "task slices won", detail: "Translation, chat, QA, summarization vs GPT-5.4" },
  { value: "3rd", label: "place at GTC", detail: "SemiAnalysis Hackathon at NVIDIA GTC 2026" },
];

const exploreLinks = [
  {
    href: "/chat/eval",
    eyebrow: "Benchmark",
    title: "See the eval results",
    body: "41.8 vs 36.1 chrF++ across 7 task slices. Interactive dashboard, per-model breakdowns.",
  },
  {
    href: "/chat",
    eyebrow: "Live model",
    title: "Talk to Tuvaluan AI",
    body: "Try real-time code-switching between Tuvaluan and English.",
  },
  {
    href: "/",
    eyebrow: "Product",
    title: "Read Tuvaluan football news",
    body: "Live translated articles from Goal.com, FIFA, and Sky Sports.",
  },
  {
    href: "/fatele",
    eyebrow: "Community",
    title: "See the Kominiti dashboard",
    body: "Real user feedback signals from across the Tuvaluan islands.",
  },
];

const gallery = [
  {
    src: "/judges/nick-football-community.webp",
    alt: "Nick Miller with Tuvaluan football community members in matching team shirts.",
    title: "Built with the community, not for them.",
    tall: false,
    pos: "center 20%",
    w: 800, h: 603,
  },
  {
    src: "/judges/rainbow-ocean.webp",
    alt: "Double rainbow arching over the Pacific Ocean in Tuvalu.",
    title: "11,000 speakers. Nine atolls. One language to save.",
    tall: false,
    pos: "center center",
    w: 800, h: 451,
  },
  {
    src: "/judges/nick-coconut-crab.webp",
    alt: "Nick Miller holding a coconut crab on a Tuvaluan beach.",
    title: "Fieldwork means getting your hands dirty.",
    tall: true,
    pos: "center 15%",
    w: 800, h: 1067,
  },
  {
    src: "/judges/island-lagoon.webp",
    alt: "A tiny coral atoll surrounded by clear turquoise lagoon water.",
    title: "The entire country is 26 km\u00B2. Smaller than Manhattan.",
    tall: false,
    pos: "center center",
    w: 800, h: 1067,
  },
  {
    src: "/judges/beach-tree.webp",
    alt: "Leaning palm tree over shallow turquoise water on a Tuvaluan beach.",
    title: "4.6 metres above sea level. That's the highest point.",
    tall: true,
    pos: "center center",
    w: 800, h: 1067,
  },
  {
    src: "/judges/futsal-article.webp",
    alt: "Magazine spread about Tuvalu's futsal team — the lowest-ranked in the world.",
    title: "Football is the language everyone shares. We started there.",
    tall: true,
    pos: "center 10%",
    w: 800, h: 1423,
  },
];

const reviews = [
  { src: "/reviews/review-1.webp", alt: "Tuvaluan reviewer says 'Very good excellent' after testing the model", w: 600, h: 623 },
  { src: "/reviews/review-2.webp", alt: "Tuvaluan reviewer says 'You got a very good training you learn Tuvalu by your self — Good Tecnology'", w: 600, h: 583 },
  { src: "/reviews/review-3.webp", alt: "Tuvaluan reviewer says 'Yeah that's perfect' after a translation test", w: 600, h: 531 },
];

const upcomingLanguages = [
  "Tuvaluan", "Tokelauan", "Niuean", "Cook Islands Maori",
  "Rotuman", "Wallisian", "Futunan",
];

// ─── Newsletter signup ───

function NewsletterSignup(props: { variant?: "hero" | "section" }) {
  const [email, setEmail] = createSignal("");
  const [state, setState] = createSignal<"idle" | "submitting" | "success" | "error">("idle");
  const isHero = () => props.variant === "hero";

  async function handleSubmit(e: SubmitEvent) {
    e.preventDefault();
    const addr = email().trim();
    if (!addr || !addr.includes("@")) return;

    setState("submitting");
    try {
      const resp = await fetch("/api/newsletter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: addr }),
      });
      if (resp.ok) {
        setState("success");
      } else {
        setState("error");
      }
    } catch {
      setState("error");
    }
  }

  return (
    <Show
      when={state() !== "success"}
      fallback={
        <div class={`newsletter-success ${isHero() ? "newsletter-success--hero" : ""}`}>
          <p class="newsletter-success__icon">&#10003;</p>
          <p class="newsletter-success__text">You're in. We'll share updates as we build.</p>
        </div>
      }
    >
      <form onSubmit={handleSubmit} class={`newsletter-form ${isHero() ? "newsletter-form--hero" : ""}`}>
        <div class="newsletter-form__row">
          <input
            type="email"
            placeholder="you@example.com"
            value={email()}
            onInput={(e) => setEmail(e.currentTarget.value)}
            class="newsletter-form__input"
            required
          />
          <button
            type="submit"
            class="newsletter-form__button"
            disabled={state() === "submitting"}
          >
            {state() === "submitting" ? "..." : "Join"}
          </button>
        </div>
        <Show when={state() === "error"}>
          <p class="newsletter-form__error">Something went wrong. Try again?</p>
        </Show>
        <p class="newsletter-form__fine">No spam. Updates on the Language Lab, new languages, and open-source releases.</p>
      </form>
    </Show>
  );
}

// ─── Twitter embed loader ───

function TweetEmbed(props: { tweetUrl: string }) {
  let containerRef: HTMLDivElement | undefined;

  onMount(() => {
    // Load Twitter widgets script if not already present
    if (!(window as any).twttr) {
      const script = document.createElement("script");
      script.src = "https://platform.twitter.com/widgets.js";
      script.async = true;
      script.charset = "utf-8";
      document.head.appendChild(script);
      script.onload = () => {
        (window as any).twttr?.widgets?.load(containerRef);
      };
    } else {
      (window as any).twttr?.widgets?.load(containerRef);
    }
  });

  return (
    <div ref={containerRef} class="tweet-embed">
      <blockquote class="twitter-tweet" data-theme="dark" data-dnt="true">
        <a href={props.tweetUrl}>Loading tweet...</a>
      </blockquote>
    </div>
  );
}

// ─── Review carousel ───

function ReviewCarousel() {
  const [current, setCurrent] = createSignal(0);
  let interval: ReturnType<typeof setInterval>;

  onMount(() => {
    interval = setInterval(() => {
      setCurrent((c) => (c + 1) % reviews.length);
    }, 4000);
  });

  onCleanup(() => clearInterval(interval));

  const go = (idx: number) => {
    setCurrent(idx);
    clearInterval(interval);
    interval = setInterval(() => {
      setCurrent((c) => (c + 1) % reviews.length);
    }, 4000);
  };

  return (
    <div>
      <div class="review-carousel">
        <div class="review-carousel__track" style={{ transform: `translateX(-${current() * 100}%)` }}>
          <For each={reviews}>
            {(r) => (
              <div class="review-carousel__slide">
                <img src={r.src} alt={r.alt} width={r.w} height={r.h} class="review-carousel__img" loading="lazy" decoding="async" />
              </div>
            )}
          </For>
        </div>
      </div>
      <div class="review-carousel__dots">
        <For each={reviews}>
          {(_, i) => (
            <button
              class={`review-carousel__dot ${i() === current() ? "review-carousel__dot--active" : ""}`}
              onClick={() => go(i())}
              aria-label={`Review ${i() + 1}`}
            />
          )}
        </For>
      </div>
    </div>
  );
}

// ─── Page ───

export default function DemoPage() {
  return (
    <main class="demo-page">
      <OGMeta
        title="We beat GPT-5.4 at Tuvaluan — now we're building a Language Lab"
        description="3rd place at GTC 2026. NVIDIA DGX Spark going to Tuvalu. A 3B model that beats GPT-5.4 on 6/7 Tuvaluan task slices. Now building an open Language Lab for dying languages."
        image="/judges/rainbow-ocean.webp"
        imageWidth={1366}
        imageHeight={768}
        url={absoluteUrl("/demo")}
      />

      {/* ═══ HERO ═══ */}
      <section class="demo-hero">
        <div class="demo-hero__backdrop" />
        <div class="demo-shell">
          <div class="demo-hero__content">
            <div class="demo-hero__badge">
              3rd Place &mdash; SemiAnalysis Hackathon @ NVIDIA GTC 2026
            </div>
            <h1 class="demo-title">
              We beat GPT-5.4 at Tuvaluan.
              <span class="demo-title__sub">Now we're giving the hardware to Tuvalu.</span>
            </h1>
            <p class="demo-lead">
              Our 3B-active model outperformed GPT-5.4 on 6 of 7 Tuvaluan language tasks.
              We won an NVIDIA DGX Spark at GTC 2026 — and we're sending it to Tuvalu so
              11,000 speakers can run their own sovereign AI agents in their own language.
            </p>

            <div class="demo-cta-row">
              <a href="#newsletter" class="demo-button demo-button--gold">
                Follow the mission
              </a>
              <A href="/chat/eval" class="demo-button demo-button--ghost">
                See benchmark results
              </A>
              <A href="/chat" class="demo-button demo-button--ghost">
                Try the model
              </A>
              <A href="/blog/technical-deepdive" class="demo-button demo-button--ghost">
                Read the technical deep dive
              </A>
            </div>

            <div class="demo-milestones">
              <For each={milestones}>
                {(m) => (
                  <div class="demo-milestone">
                    <p class="demo-milestone__value">{m.value}</p>
                    <p class="demo-milestone__label">{m.label}</p>
                    <p class="demo-milestone__detail">{m.detail}</p>
                  </div>
                )}
              </For>
            </div>
          </div>
        </div>
      </section>

      {/* ═══ THE RESULT ═══ */}
      <section class="demo-section">
        <div class="demo-shell">
          <div class="demo-result-grid">
            <div class="demo-result-grid__copy">
              <p class="demo-kicker demo-kicker--dark">The result</p>
              <h2 class="demo-section__title">
                3B parameters. Beats frontier models.
              </h2>
              <p class="demo-section__text">
                We built the largest Tuvaluan-English corpus ever assembled (342k pairs),
                trained a two-stage model on Thinking Machines' Tinker platform, and evaluated
                it against GPT-5.4, Claude Sonnet, Gemini, and Google Translate on 7 task slices.
              </p>
              <p class="demo-section__text">
                Our model leads on 6 of 7. Specialization beats scale when the infrastructure
                is right.
              </p>
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
          </div>
        </div>
      </section>

      {/* ═══ GTC WIN + TWEET ═══ */}
      <section class="demo-section demo-section--dark">
        <div class="demo-shell">
          <div class="demo-gtc-grid">
            <div class="demo-gtc-grid__copy">
              <p class="demo-kicker">GTC 2026</p>
              <h2 class="demo-section__title demo-section__title--light">
                3rd place. DGX Spark. Going to Tuvalu.
              </h2>
              <p class="demo-section__text">
                The SemiAnalysis hackathon at NVIDIA GTC 2026 challenged teams to build
                real AI systems, not demos. We placed 3rd and won an NVIDIA DGX Spark.
              </p>
              <p class="demo-section__text">
                We're not keeping it. The DGX Spark is going to Tuvalu — so a nation
                of 11,000 people can run AI agents that speak their language, on their
                own hardware, under their own control. Sovereign AI starts with sovereign
                infrastructure.
              </p>
            </div>
            <div class="demo-gtc-grid__tweet">
              <TweetEmbed tweetUrl="https://twitter.com/SemiAnalysis_/status/2033375212700340715" />
            </div>
          </div>
        </div>
      </section>

      {/* ═══ WHY TUVALU ═══ */}
      <section class="demo-section demo-section--why">
        <div class="demo-shell">
          <div class="demo-why-grid">
            <div class="demo-why-grid__copy">
              <p class="demo-kicker demo-kicker--dark">Why Tuvalu</p>
              <h2 class="demo-section__title">
                A country with 15 years left.
              </h2>
              <p class="demo-section__text">
                Tuvalu is a nation of 11,000 people spread across nine coral atolls in the
                Pacific. The highest point is 4.6 meters above sea level. At current rates
                of sea level rise, most of the country will be uninhabitable within 15 years.
              </p>
              <p class="demo-section__text">
                When a country disappears, its language disappears with it. Tuvaluan has no
                backup — no large diaspora, no written literary tradition at scale, no presence
                in any major AI system. If the land goes underwater and the language has no
                digital infrastructure, it is gone forever.
              </p>
              <p class="demo-section__text">
                That is why we chose Tuvaluan. Not because it was easy — it was the hardest
                possible test case. If we can build sovereign AI for a language this small,
                this endangered, and this invisible to frontier models, the playbook works
                for every language.
              </p>
              <blockquote class="demo-quote">
                <p class="demo-quote__text">
                  "We will not stand idly by as the water rises around us."
                </p>
                <cite class="demo-quote__cite">
                  Simon Kofe, Tuvalu Foreign Minister — COP26, standing knee-deep in the rising sea
                </cite>
              </blockquote>
            </div>
            <div class="demo-why-grid__video">
              <div class="demo-video-wrap">
                <iframe
                  src="https://www.youtube.com/embed/jBBsv0QyscE?si=_zY1GnLk0b9vNZKX&controls=0"
                  title="Tuvalu's foreign minister gives COP26 speech standing in rising seawater"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                  referrerpolicy="strict-origin-when-cross-origin"
                  allowfullscreen
                  loading="lazy"
                />
              </div>
              <p class="demo-video-caption">
                Tuvalu's foreign minister Simon Kofe addresses COP26 from the rising ocean.
                Over 1 million views. The world watched — then looked away.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ═══ TUVALUAN APPROVED ═══ */}
      <section class="demo-section demo-section--dark demo-section--reviews">
        <div class="demo-shell">
          <div class="demo-reviews-grid">
            <div class="demo-reviews-grid__copy">
              <p class="demo-kicker">Tuvaluan approved</p>
              <h2 class="demo-section__title demo-section__title--light">
                Real feedback from real speakers.
              </h2>
              <div class="demo-reviews__stars">
                {"★★★★★"}
              </div>
              <p class="demo-section__text">
                We don't just run benchmarks — we send our translations to native Tuvaluan
                speakers and ask them to judge. These are real text messages from community
                members reviewing our model's output.
              </p>
              <p class="demo-section__text" style={{ "font-style": "italic", opacity: 0.8 }}>
                "You got a very good training you learn Tuvalu by your self — Good Tecnology"
              </p>
            </div>
            <div class="demo-reviews-grid__carousel">
              <ReviewCarousel />
            </div>
          </div>
        </div>
      </section>

      {/* ═══ EXPLORE THE SYSTEM ═══ */}
      <section class="demo-section">
        <div class="demo-shell">
          <div class="demo-section__intro">
            <p class="demo-kicker demo-kicker--dark">Explore the system</p>
            <h2 class="demo-section__title">Everything is live and open</h2>
          </div>

          <div class="demo-link-grid">
            <For each={exploreLinks}>
              {(link) => (
                <A href={link.href} class="demo-link-card">
                  <p class="demo-link-card__eyebrow">{link.eyebrow}</p>
                  <h3 class="demo-link-card__title">{link.title}</h3>
                  <p class="demo-link-card__body">{link.body}</p>
                  <span class="demo-link-card__cta">Open &rarr;</span>
                </A>
              )}
            </For>
          </div>
        </div>
      </section>

      {/* ═══ GALLERY ═══ */}
      <section class="demo-section demo-section--sand">
        <div class="demo-shell">
          <div class="demo-section__intro">
            <p class="demo-kicker demo-kicker--dark">On the ground</p>
            <h2 class="demo-section__title">This work comes from a real place</h2>
          </div>
          <div class="demo-gallery">
            <For each={gallery}>
              {(image) => (
                <figure class={`demo-gallery__item ${image.tall ? "demo-gallery__item--tall" : ""}`}>
                  <img src={image.src} alt={image.alt} width={image.w} height={image.h} class="demo-gallery__image" loading="lazy" decoding="async" style={{ "object-position": image.pos }} />
                  <figcaption class="demo-gallery__caption">
                    <p class="demo-gallery__title">{image.title}</p>
                  </figcaption>
                </figure>
              )}
            </For>
          </div>
        </div>
      </section>

      {/* ═══ LANGUAGE LAB ═══ */}
      <section class="demo-section demo-section--dark demo-section--lab">
        <div class="demo-shell demo-shell--narrow">
          <div class="demo-lab">
            <p class="demo-kicker">What's next</p>
            <h2 class="demo-section__title demo-section__title--light demo-lab__title">
              The Language Lab
            </h2>
            <p class="demo-lab__tagline">
              Preserving dying languages. Enabling sovereign AI.
            </p>
            <p class="demo-section__text">
              Tuvaluan was the proof. Now we're building an open-source Language Lab — a
              nonprofit 501(c)(3) that gives endangered language communities the tools,
              models, and hardware to run their own AI systems.
            </p>
            <p class="demo-section__text">
              The playbook is proven: build the corpus, train a specialized model, ship
              a real product, collect feedback, improve. We did it for Tuvaluan with
              342k pairs and 3B active parameters. We're doing it next for the Pacific
              languages closest to disappearing.
            </p>

            <div class="demo-lab__languages">
              <p class="demo-lab__languages-label">First target languages</p>
              <div class="demo-lab__language-tags">
                <For each={upcomingLanguages}>
                  {(lang) => <span class="demo-lab__tag">{lang}</span>}
                </For>
              </div>
            </div>

            <div class="demo-lab__pillars">
              <div class="demo-lab__pillar">
                <p class="demo-lab__pillar-num">01</p>
                <p class="demo-lab__pillar-title">Open corpus infrastructure</p>
                <p class="demo-lab__pillar-text">
                  Scalable pipelines for scraping, aligning, cleaning, and decontaminating
                  parallel text for any low-resource language. Published to Hugging Face.
                </p>
              </div>
              <div class="demo-lab__pillar">
                <p class="demo-lab__pillar-num">02</p>
                <p class="demo-lab__pillar-title">Community-owned models</p>
                <p class="demo-lab__pillar-text">
                  Specialized models trained on community data, evaluated by native speakers,
                  and deployed on community hardware. Not a cloud API — actual sovereignty.
                </p>
              </div>
              <div class="demo-lab__pillar">
                <p class="demo-lab__pillar-num">03</p>
                <p class="demo-lab__pillar-title">Sovereign hardware</p>
                <p class="demo-lab__pillar-text">
                  The DGX Spark goes to Tuvalu. Future hardware goes to future communities.
                  AI agents that speak your language should run on your infrastructure.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══ NEWSLETTER ═══ */}
      <section class="demo-section demo-section--cta" id="newsletter">
        <div class="demo-shell demo-shell--narrow">
          <div class="demo-newsletter">
            <h2 class="demo-newsletter__title">
              Help us save the next language.
            </h2>
            <p class="demo-newsletter__text">
              Join the Language Lab mailing list. Get updates on new languages, open-source
              releases, deployment milestones, and ways to contribute — whether you're a
              linguist, an engineer, or someone who believes every language deserves AI.
            </p>
            <NewsletterSignup variant="section" />
            <div class="demo-newsletter__links">
              <a href="https://huggingface.co/datasets/FriezaForce/tv2en-cleaned" class="demo-inline-link demo-inline-link--dark">
                Dataset on HF
              </a>
              <a href="https://huggingface.co/FriezaForce/tvl-en-llm-translation-stage-a" class="demo-inline-link demo-inline-link--dark">
                Model card
              </a>
              <a href="https://github.com/G-structure/tuvalu-llm" class="demo-inline-link demo-inline-link--dark">
                GitHub
              </a>
            </div>
            <p class="demo-newsletter__contact">
              Need to reach us directly? Email <a href="mailto:contact@sanative.ai">contact@sanative.ai</a>
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
