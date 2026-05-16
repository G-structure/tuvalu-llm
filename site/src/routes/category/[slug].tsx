import { createAsync, cache, useParams, useSearchParams, A } from "@solidjs/router";
import { For, Show } from "solid-js";
import { getArticles, getCategories } from "~/lib/db";
import type { Article, Category } from "~/lib/types";
import ArticleCard from "~/components/ArticleCard";
import CategoryPills from "~/components/CategoryPills";
import OGMeta from "~/components/OGMeta";
import StructuredData from "~/components/StructuredData";
import {
  breadcrumbList,
  footballCollectionPage,
} from "~/lib/seo";
import {
  absoluteFootballUrl,
  FOOTBALL_CATEGORY_OG_IMAGES,
  FOOTBALL_META,
  SITE_ORIGINS,
} from "~/lib/site";

const PER_PAGE = 20;

const loadCategory = cache(async (slug: string, page: number) => {
  "use server";
  const offset = (page - 1) * PER_PAGE;
  const [articles, categories] = await Promise.all([
    getArticles(PER_PAGE + 1, offset, slug),
    getCategories(),
  ]);
  return { articles, categories, slug, page };
}, "category");

export const route = {
  load: ({ params, location }: { params: { slug: string }; location: { query: Record<string, string> } }) => {
    const page = Math.max(1, parseInt(location.query.page || "1", 10) || 1);
    return loadCategory(params.slug, page);
  },
};

export default function CategoryPage() {
  const params = useParams();
  const [searchParams] = useSearchParams();
  const page = () => Math.max(1, parseInt(searchParams.page || "1", 10) || 1);
  const data = createAsync(() => loadCategory(params.slug, page()));

  const displayName = () =>
    params.slug.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const socialTitle = () =>
    params.slug === "football" ? "Football News" : `${displayName()} Football News`;
  const socialImage = () =>
    FOOTBALL_CATEGORY_OG_IMAGES[params.slug] || FOOTBALL_META.defaultOgImage;
  const canonicalUrl = () =>
    absoluteFootballUrl(
      page() === 1 ? `/category/${params.slug}` : `/category/${params.slug}?page=${page()}`
    );
  const description = () => `${displayName()} football news in Tuvaluan and English`;

  return (
    <main class="site-page lagoon-subpage category-dashboard-page">
      <OGMeta
        title={socialTitle()}
        description={description()}
        url={canonicalUrl()}
        image={socialImage()}
        imageOrigin={SITE_ORIGINS.football}
        imageWidth={FOOTBALL_META.defaultOgImageWidth}
        imageHeight={FOOTBALL_META.defaultOgImageHeight}
        imageAlt={`Talafutipolo social card for ${socialTitle().toLowerCase()}.`}
        siteName={FOOTBALL_META.productName}
        titleSuffix={FOOTBALL_META.productName}
      />
      <StructuredData
        data={[
          footballCollectionPage({
            name: socialTitle(),
            description: description(),
            url: canonicalUrl(),
            image: socialImage(),
          }),
          breadcrumbList([
            { name: FOOTBALL_META.productName, url: absoluteFootballUrl("/") },
            { name: socialTitle(), url: canonicalUrl() },
          ]),
        ]}
      />

      <Show when={data()}>
        {(d) => {
          const articles = () => d().articles.slice(0, PER_PAGE);
          const hasNext = () => d().articles.length > PER_PAGE;
          const hasPrev = () => d().page > 1;

          return (
            <>
              <section class="site-hero site-hero--compact lagoon-subhero">
                <div class="site-shell site-shell--wide lagoon-subhero__grid">
                  <div>
                  <p class="site-kicker">Category</p>
                  <h1 class="site-title capitalize">{displayName()}</h1>
                  <p class="site-lede">{description()}</p>
                  </div>
                  <aside class="lagoon-subhero__panel">
                    <span>Active beat</span>
                    <strong>{articles().length}</strong>
                    <em>translated stories on this page</em>
                  </aside>
                </div>
              </section>

              <div class="site-shell site-shell--wide category-dashboard">
                <div class="category-dashboard__pills">
                  <CategoryPills categories={d().categories} />
                </div>

                <Show when={articles().length === 0}>
                  <div class="site-empty">
                    <strong>Seki isi tala</strong>
                    <span>No articles in this category</span>
                  </div>
                </Show>

                <Show when={articles().length > 0}>
                  <section class="category-feature-grid">
                    <ArticleCard article={articles()[0]} hero />
                    <aside class="category-insight-card">
                      <p class="site-kicker">Latest in {displayName()}</p>
                      <h2>Follow this beat in Tuvaluan first.</h2>
                      <p>
                        Stories here keep the same bilingual reading and coaching tools
                        as the main wire, with translations ready for community review.
                      </p>
                      <A href="/fatele" class="site-button site-button--gold">
                        Coach translations
                      </A>
                    </aside>
                  </section>
                </Show>

                <Show when={articles().length > 1}>
                  <section class="site-section category-tile-section">
                    <div class="home-card-grid">
                      <For each={articles().slice(1)}>
                        {(article) => <ArticleCard article={article} tile />}
                      </For>
                    </div>
                  </section>
                </Show>

                <Show when={hasPrev() || hasNext()}>
                  <div class="site-pagination">
                    <Show when={hasPrev()}>
                      <A
                        href={d().page === 2 ? `/category/${params.slug}` : `/category/${params.slug}?page=${d().page - 1}`}
                        class="site-button site-button--ghost"
                      >
                        &larr; Foki
                      </A>
                    </Show>
                    <Show when={hasNext()}>
                      <A
                        href={`/category/${params.slug}?page=${d().page + 1}`}
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
