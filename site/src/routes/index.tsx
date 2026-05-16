import { createAsync, cache, useSearchParams, A } from "@solidjs/router";
import { For, Show } from "solid-js";
import { getArticles, getCategories, getFateleStats } from "~/lib/db";
import type { Article } from "~/lib/types";
import ArticleCard from "~/components/ArticleCard";
import CategoryPills from "~/components/CategoryPills";
import OGMeta from "~/components/OGMeta";
import StructuredData from "~/components/StructuredData";
import { timeAgo } from "~/lib/time";
import {
  footballCollectionPage,
  footballWebsite,
  languageLabOrganization,
} from "~/lib/seo";
import { absoluteFootballUrl, FOOTBALL_META, SITE_ORIGINS } from "~/lib/site";

const PER_PAGE = 20;

const loadHome = cache(async (page: number) => {
  "use server";
  const offset = (page - 1) * PER_PAGE;
  const [articles, categories, stats] = await Promise.all([
    getArticles(PER_PAGE + 1, offset),
    getCategories(),
    getFateleStats(),
  ]);
  return { articles, categories, stats, page };
}, "home");

function storyTitle(article: Article) {
  return article.title_tvl || article.title_en;
}

function storyCategory(article: Article) {
  return (article.category || "Talafuti").replace(/-/g, " ");
}

function storyImage(article: Article, fallback: string) {
  return article.image_url || fallback;
}

export const route = {
  load: ({ location }: { location: { query: Record<string, string> } }) => {
    const page = Math.max(1, parseInt(location.query.page || "1", 10) || 1);
    return loadHome(page);
  },
};

export default function Home() {
  const [searchParams] = useSearchParams();
  const page = () => Math.max(1, parseInt(searchParams.page || "1", 10) || 1);
  const data = createAsync(() => loadHome(page()));
  const canonicalUrl = () => absoluteFootballUrl(page() === 1 ? "/" : `/?page=${page()}`);
  const description =
    "Tala futipolo mai te lalolagi i te gagana Tuvalu. Football news from around the world in Tuvaluan and English.";

  return (
    <main class="site-page home-dashboard-page">
      <OGMeta
        title={FOOTBALL_META.productName}
        description={description}
        url={canonicalUrl()}
        image={FOOTBALL_META.defaultOgImage}
        imageOrigin={SITE_ORIGINS.football}
        imageWidth={FOOTBALL_META.defaultOgImageWidth}
        imageHeight={FOOTBALL_META.defaultOgImageHeight}
        imageAlt={FOOTBALL_META.defaultOgImageAlt}
        siteName={FOOTBALL_META.productName}
        titleSuffix="Tala Futipolo i te Gagana Tuvalu"
      />
      <StructuredData
        data={[
          languageLabOrganization(),
          footballWebsite(),
          footballCollectionPage({
            name: FOOTBALL_META.productName,
            description,
            url: canonicalUrl(),
            image: FOOTBALL_META.defaultOgImage,
          }),
        ]}
      />

      <Show when={data()}>
        {(d) => {
          const articles = () => d().articles.slice(0, PER_PAGE);
          const homePillCategories = () =>
            d().categories.filter((category) => category.slug.toLowerCase() !== "kominiti");
          const hasNext = () => d().articles.length > PER_PAGE;
          const hasPrev = () => d().page > 1;

          return (
            <>
              <section class="home-dashboard" aria-label="Fenua Intelligence dashboard">
                <div class="site-shell site-shell--wide home-dashboard__layout">
                  <section class="home-chat-panel">
                    <div class="home-chat-panel__content">
                      <p class="home-chat-panel__eyebrow">TVL Chat</p>
                      <h1>
                        TVL Chat <span aria-hidden="true" class="home-star" />
                      </h1>
                      <p>AI for Tuvalu. By Tuvalu.</p>
                      <div class="home-chat-panel__actions">
                        <A href="/chat" class="home-pill-button home-pill-button--dark">
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z" />
                          </svg>
                          Fesili i te AI
                        </A>
                        <A href="/chat" class="home-pill-button home-pill-button--light">
                          Ask AI
                        </A>
                      </div>
                      <A href="/chat" class="home-chat-prompt">
                        <span>Fesili i te AI...</span>
                        <span class="home-chat-prompt__send" aria-hidden="true">
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
                            <path d="m9 18 6-6-6-6" />
                          </svg>
                        </span>
                      </A>
                      <div class="home-chat-panel__chips">
                        <span>Fakamatalaaga e uiga ki Tuvalu</span>
                        <span>Tulafono o te Pasefika</span>
                        <span>Kaupulega o te Kaupule</span>
                      </div>
                    </div>
                    <div class="home-chat-panel__secure">
                      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <rect width="18" height="11" x="3" y="11" rx="2" />
                        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                      </svg>
                      Secure. Private. Built for Tuvalu.
                    </div>
                  </section>

                  <Show when={articles().length > 0}>
                    <A href={`/articles/${articles()[0].id}`} class="home-lead-card">
                      <img
                        src={storyImage(articles()[0], "/judges/nick-football-community.webp")}
                        alt={articles()[0].image_alt || storyTitle(articles()[0])}
                        width={articles()[0].image_width || undefined}
                        height={articles()[0].image_height || undefined}
                        loading="eager"
                        fetchpriority="high"
                        decoding="async"
                      />
                      <div class="home-lead-card__shade" />
                      <div class="home-lead-card__content">
                        <span class="home-lead-card__badge">Latest</span>
                        <h2>{storyTitle(articles()[0])}</h2>
                        <p>
                          {storyCategory(articles()[0])} · {timeAgo(articles()[0].published_at)}
                        </p>
                        <span class="home-lead-card__cta">
                          Lau Fakamatalaaga
                          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <path d="M5 12h14" />
                            <path d="m12 5 7 7-7 7" />
                          </svg>
                        </span>
                      </div>
                      <div class="home-lead-card__scorebar">
                        <span class="home-flag-mark" aria-hidden="true" />
                        <span>Tuvaluan</span>
                        <strong>TVL</strong>
                        <span>English</span>
                        <strong>EN</strong>
                      </div>
                    </A>
                  </Show>

                  <aside class="home-side-rail">
                    <section class="home-rail-card">
                      <div class="home-rail-card__head">
                        <h2>Kominiti</h2>
                        <p>Fenua tu'atasi, lotou fakatasi.</p>
                      </div>
                      <div class="home-community-list">
                        <div class="home-community-row">
                          <span class="home-community-row__icon">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                              <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
                              <circle cx="9" cy="7" r="4" />
                              <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
                              <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                            </svg>
                          </span>
                          <span>
                            <strong>{d().stats.total_this_month}</strong>
                            <em>Signals this month</em>
                          </span>
                        </div>
                        <div class="home-community-row">
                          <span class="home-community-row__icon">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                              <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z" />
                            </svg>
                          </span>
                          <span>
                            <strong>{d().stats.article_feedback_count}</strong>
                            <em>Article coach notes</em>
                          </span>
                        </div>
                        <div class="home-community-row">
                          <span class="home-community-row__icon">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                              <path d="M8 2v4" />
                              <path d="M16 2v4" />
                              <rect width="18" height="18" x="3" y="4" rx="2" />
                              <path d="M3 10h18" />
                            </svg>
                          </span>
                          <span>
                            <strong>{d().categories.length}</strong>
                            <em>Article beats</em>
                          </span>
                        </div>
                        <div class="home-community-row">
                          <span class="home-community-row__icon">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                              <path d="M8 21h8" />
                              <path d="M12 17v4" />
                              <path d="M7 4h10v5a5 5 0 0 1-10 0z" />
                              <path d="M5 7H3a3 3 0 0 0 3 3" />
                              <path d="M19 7h2a3 3 0 0 1-3 3" />
                            </svg>
                          </span>
                          <span>
                            <strong>{d().stats.corrections_count}</strong>
                            <em>Fakaleiga fou</em>
                          </span>
                        </div>
                      </div>
                    </section>

                    <section class="home-rail-card home-rail-card--learning">
                      <div class="home-rail-card__head">
                        <h2>Fenua learning loop</h2>
                        <p>Real signals from chat corrections and article feedback.</p>
                      </div>
                      <div class="home-learning-list">
                        <div class="home-learning-row">
                          <span>Collected</span>
                          <strong>{d().stats.total_this_month}</strong>
                          <em>database-backed signals this month</em>
                        </div>
                        <div class="home-learning-row">
                          <span>Corrections</span>
                          <strong>{d().stats.corrections_count}</strong>
                          <em>translation fixes ready for review</em>
                        </div>
                        <div class="home-learning-row">
                          <span>Notes</span>
                          <strong>{d().stats.article_feedback_count}</strong>
                          <em>article feedback forms submitted</em>
                        </div>
                      </div>
                      <A href="/fatele" class="home-rail-link">
                        View signal dashboard
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                          <path d="m9 18 6-6-6-6" />
                        </svg>
                      </A>
                    </section>
                  </aside>

                  <section id="latest" class="home-latest-board">
                    <div class="home-latest-board__head">
                      <h2>Talafutipolo & Latest</h2>
                      <Show when={homePillCategories().length > 0}>
                        <CategoryPills categories={homePillCategories()} />
                      </Show>
                    </div>
                    <Show when={articles().length > 1}>
                      <div class="home-card-grid">
                        <For each={articles().slice(1, 5)}>
                          {(article) => <ArticleCard article={article} tile />}
                        </For>
                      </div>
                    </Show>
                  </section>
                </div>
              </section>

              <div class="site-shell site-shell--wide">
                {/* Empty state */}
                <Show when={articles().length === 0}>
                  <div class="site-empty">
                    <strong>Seki isi tala</strong>
                    <span>No articles yet</span>
                  </div>
                </Show>

                {/* Remaining articles as thumbnail rows */}
                <Show when={articles().length > 5}>
                  <section class="site-section home-more-list">
                    <div class="site-section-head">
                      <div>
                        <p class="site-kicker">Catch up</p>
                        <h2 class="site-section-title">More football news</h2>
                      </div>
                    </div>
                    <div class="site-grid">
                      <For each={articles().slice(5)}>
                        {(article) => <ArticleCard article={article} />}
                      </For>
                    </div>
                  </section>
                </Show>

                {/* Pagination */}
                <Show when={hasPrev() || hasNext()}>
                  <div class="site-pagination">
                    <Show when={hasPrev()}>
                      <A
                        href={d().page === 2 ? "/" : `/?page=${d().page - 1}`}
                        class="site-button site-button--ghost"
                      >
                        &larr; Foki
                      </A>
                    </Show>
                    <Show when={hasNext()}>
                      <A
                        href={`/?page=${d().page + 1}`}
                        class="site-button site-button--primary"
                      >
                        Faitau atu &darr;
                      </A>
                    </Show>
                  </div>
                </Show>
                </div>
            </>
          );
        }}
      </Show>
    </main>
  );
}
