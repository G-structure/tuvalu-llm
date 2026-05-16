import { createAsync, cache, useParams, A } from "@solidjs/router";
import { createSignal, For, Show } from "solid-js";
import { HttpStatusCode } from "@solidjs/start";
import { getArticle } from "~/lib/db";
import type { Article } from "~/lib/types";
import { formatDate } from "~/lib/time";
import OGMeta from "~/components/OGMeta";
import StructuredData from "~/components/StructuredData";
import {
  footballNewsArticleStructuredData,
  NOINDEX_ROBOTS,
  sourceName,
} from "~/lib/seo";
import { absoluteFootballUrl, FOOTBALL_META, SITE_ORIGINS } from "~/lib/site";
import type { LanguageMode } from "~/components/LanguageToggle";
import LanguageToggle from "~/components/LanguageToggle";
import CoachTranslatorCard from "~/components/CoachTranslatorCard";
import { promptForIslandIfUnknown } from "~/components/IslandSelector";
import { ensureCommunitySessionId, getKnownIsland } from "~/lib/community";

const loadArticle = cache(async (id: string) => {
  "use server";
  return (await getArticle(id)) || null;
}, "article");

export const route = {
  load: ({ params }: { params: { id: string } }) => loadArticle(params.id),
};

function splitParagraphs(body: string): string[] {
  if (body.includes("<p")) {
    const matches = body.match(/<p[^>]*>([\s\S]*?)<\/p>/gi);
    if (matches) {
      return matches
        .map((m) => m.replace(/<[^>]+>/g, "").trim())
        .filter((p) => p.length > 0);
    }
  }
  return body
    .split(/\n\n+/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
}

function SourceName(props: { sourceId: string }) {
  return <>{sourceName(props.sourceId)}</>;
}

async function sendSignal(articleId: string, signalType: string, paragraphIndex?: number) {
  const island = getKnownIsland();
  const session = ensureCommunitySessionId();
  fetch("/api/signal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      article_id: articleId,
      signal_type: signalType,
      paragraph_index: paragraphIndex,
      session_id: session,
      island,
    }),
  }).catch(() => {});
}

function BilingualParagraph(props: {
  tvl: string;
  en: string;
  mode: LanguageMode;
  index: number;
  articleId: string;
}) {
  const [showEn, setShowEn] = createSignal(false);
  const [vote, setVote] = createSignal<"thumbs_up" | "thumbs_down" | null>(null);

  const handleVote = async (type: "thumbs_up" | "thumbs_down") => {
    if (vote() === type) return;
    const island = getKnownIsland();
    const session = ensureCommunitySessionId();
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        article_id: props.articleId,
        paragraph_idx: props.index,
        feedback_type: type,
        island,
        session_id: session,
      }),
    }).catch(() => null);

    if (!response?.ok) return;
    setVote(type);

    if (!island) {
      await promptForIslandIfUnknown();
    }
  };

  const handleReveal = () => {
    const wasHidden = !showEn();
    setShowEn(!showEn());
    if (wasHidden) {
      sendSignal(props.articleId, "reveal", props.index);
    }
  };

  return (
    <div class="article-body-block">
      <Show when={props.mode === "tv" || props.mode === "tv+en"}>
        <div class="article-body-block__row">
          <p>{props.tvl}</p>
          <div class="article-votes">
            <button
              type="button"
              onClick={() => handleVote("thumbs_up")}
              class={`article-vote ${
                vote() === "thumbs_up" ? "article-vote--active" : ""
              }`}
              title="Tonu! (Good translation)"
              aria-label="Good translation"
              aria-pressed={vote() === "thumbs_up"}
            >
              👍🏾
            </button>
            <button
              type="button"
              onClick={() => handleVote("thumbs_down")}
              class={`article-vote ${
                vote() === "thumbs_down" ? "article-vote--active" : ""
              }`}
              title="Seki tonu (Bad translation)"
              aria-label="Bad translation"
              aria-pressed={vote() === "thumbs_down"}
            >
              👎🏾
            </button>
          </div>
        </div>
      </Show>

      <Show when={props.mode === "tv"}>
        <button
          onClick={handleReveal}
          class="article-reveal"
        >
          {showEn() ? "Funa te English" : "Fakakite English"}
        </button>
        <Show when={showEn()}>
          <p class="article-translation">
            {props.en}
          </p>
        </Show>
      </Show>

      <Show when={props.mode === "tv+en"}>
        <p class="article-translation">
          {props.en}
        </p>
      </Show>

      <Show when={props.mode === "en"}>
        <p>{props.en}</p>
      </Show>
    </div>
  );
}

export default function ArticlePage() {
  const params = useParams();
  const article = createAsync(() => loadArticle(params.id), { deferStream: true });
  const [langMode, setLangMode] = createSignal<LanguageMode>("tv");

  return (
    <Show when={article() !== undefined} fallback={<main class="site-page" />}>
      <Show
        when={article()}
        fallback={
          <main class="site-page">
            <HttpStatusCode code={404} />
            <OGMeta
              title="Article not found"
              description="This article may have been removed or the ID is invalid."
              url={absoluteFootballUrl(`/articles/${params.id}`)}
              image={FOOTBALL_META.articleFallbackOgImage}
              imageOrigin={SITE_ORIGINS.football}
              imageWidth={FOOTBALL_META.articleFallbackImageWidth}
              imageHeight={FOOTBALL_META.articleFallbackImageHeight}
              imageAlt={FOOTBALL_META.articleFallbackImageAlt}
              siteName={FOOTBALL_META.productName}
              titleSuffix={FOOTBALL_META.productName}
              robots={NOINDEX_ROBOTS}
            />
            <div class="site-shell site-empty">
              <strong>Article not found</strong>
              <span>This article may have been removed or the ID is invalid.</span>
            </div>
          </main>
        }
      >
      {(a) => {
        const title = () =>
          langMode() === "en"
            ? a().title_en
            : a().title_tvl || a().title_en;
        const description = () =>
          a().og_description_tvl || a().og_description_en || "";
        const enParagraphs = () => splitParagraphs(a().body_en);
        const tvlParagraphs = () =>
          a().body_tvl ? splitParagraphs(a().body_tvl!) : [];
        const hasTvl = () => tvlParagraphs().length > 0;
        const effectiveMode = () => (hasTvl() ? langMode() : "en");
        const hasArticleImage = () => !!a().image_url;
        const imageSrc = () => a().image_url || FOOTBALL_META.articleFallbackOgImage;
        const imageWidth = () =>
          a().image_width ||
          (hasArticleImage()
            ? FOOTBALL_META.defaultOgImageWidth
            : FOOTBALL_META.articleFallbackImageWidth);
        const imageHeight = () =>
          a().image_height ||
          (hasArticleImage()
            ? FOOTBALL_META.defaultOgImageHeight
            : FOOTBALL_META.articleFallbackImageHeight);
        const imageAlt = () =>
          a().image_alt ||
          (hasArticleImage() ? title() : FOOTBALL_META.articleFallbackImageAlt);

        return (
          <main class="site-page article-page lagoon-subpage">
            <OGMeta
              title={a().title_tvl || a().title_en}
              description={description() || undefined}
              image={imageSrc()}
              imageOrigin={SITE_ORIGINS.football}
              imageWidth={imageWidth()}
              imageHeight={imageHeight()}
              imageAlt={imageAlt()}
              publishedAt={a().published_at}
              category={a().category || undefined}
              type="article"
              url={absoluteFootballUrl(`/articles/${a().id}`)}
              siteName={FOOTBALL_META.productName}
              titleSuffix={FOOTBALL_META.productName}
            />
            <StructuredData data={footballNewsArticleStructuredData(a())} />

            {/* Top bar with back + language toggle */}
            <div class="site-shell site-shell--wide article-topbar">
              <A
                href="/"
                class="article-back-link"
              >
                &larr; Foki
              </A>
              <Show when={hasTvl()}>
                <LanguageToggle
                  mode={langMode()}
                  onChange={setLangMode}
                />
              </Show>
            </div>

            {/* Hero image */}
            <article class="site-shell site-shell--article article-shell article-shell--lagoon">
              {/* Title — TVL first */}
              <p class="site-kicker">Football story</p>
              <h1 class="article-title">
                {title()}
              </h1>

              {/* EN subtitle when showing TVL title */}
              <Show
                when={
                  hasTvl() &&
                  effectiveMode() !== "en" &&
                  a().title_en !== a().title_tvl
                }
              >
                <p class="article-subtitle">
                  {a().title_en}
                </p>
              </Show>

              {/* TVL subtitle when showing EN title */}
              <Show
                when={
                  hasTvl() &&
                  effectiveMode() === "en" &&
                  a().title_tvl
                }
              >
                <p class="article-subtitle">
                  {a().title_tvl}
                </p>
              </Show>

              {/* Meta line */}
              <div class="article-meta">
                <span>{formatDate(a().published_at)}</span>
                <span>&middot;</span>
                <SourceName sourceId={a().source_id} />
                <Show when={a().author}>
                  <>
                    <span>&middot;</span>
                    <span>{a().author}</span>
                  </>
                </Show>
                <Show when={a().category}>
                  <>
                    <span>&middot;</span>
                    <span class="capitalize">
                      {a().category!.replace(/-/g, " ")}
                    </span>
                  </>
                </Show>
              </div>

              <div class="article-info-strip" aria-label="Story tools">
                <div>
                  <span>Reading mode</span>
                  <strong>{effectiveMode().toUpperCase()}</strong>
                </div>
                <div>
                  <span>Source</span>
                  <strong><SourceName sourceId={a().source_id} /></strong>
                </div>
                <div>
                  <span>Community</span>
                  <strong>Coach ready</strong>
                </div>
              </div>

              <figure class="article-hero-image">
                <img
                  src={imageSrc()}
                  alt={imageAlt()}
                  width={imageWidth()}
                  height={imageHeight()}
                  loading="eager"
                  fetchpriority="high"
                  decoding="async"
                />
              </figure>

              {/* Body */}
              <div class="article-body">
                <Show
                  when={hasTvl()}
                  fallback={
                    /* English only — no translation available */
                    <For each={enParagraphs()}>
                      {(p) => (
                        <p>
                          {p}
                        </p>
                      )}
                    </For>
                  }
                >
                  {/* Bilingual paragraphs */}
                  <For each={tvlParagraphs()}>
                    {(tvlP, i) => (
                      <BilingualParagraph
                        tvl={tvlP}
                        en={enParagraphs()[i()] || ""}
                        mode={effectiveMode()}
                        index={i()}
                        articleId={a().id}
                      />
                    )}
                  </For>
                </Show>
              </div>

              <Show when={hasTvl()}>
                <CoachTranslatorCard
                  articleId={a().id}
                  paragraphCount={tvlParagraphs().length}
                  initialMode={effectiveMode()}
                />
              </Show>

              {/* Source attribution + share */}
              <div class="article-footer">
                <a
                  href={a().url}
                  target="_blank"
                  rel="noopener noreferrer"
                  class="article-source-link"
                >
                  Read original at <SourceName sourceId={a().source_id} />
                </a>
                <button
                  onClick={() => {
                    sendSignal(a().id, "share");
                    if (navigator.share) {
                      navigator.share({
                        title: a().title_tvl || a().title_en,
                        text:
                          a().og_description_tvl ||
                          a().og_description_en ||
                          undefined,
                        url: window.location.href,
                      });
                    } else {
                      navigator.clipboard.writeText(window.location.href);
                    }
                  }}
                  class="site-button site-button--primary"
                  aria-label="Fakasoa (Share)"
                >
                  Fakasoa
                </button>
              </div>
            </article>
          </main>
        );
      }}
      </Show>
    </Show>
  );
}
