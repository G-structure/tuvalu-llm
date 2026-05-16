import { createAsync, cache, useSearchParams, A } from "@solidjs/router";
import { For, Show } from "solid-js";
import { searchArticles } from "~/lib/db";
import ArticleCard from "~/components/ArticleCard";
import OGMeta from "~/components/OGMeta";
import { NOINDEX_ROBOTS } from "~/lib/seo";
import { absoluteFootballUrl, FOOTBALL_META, SITE_ORIGINS } from "~/lib/site";

const loadSearch = cache(async (q: string) => {
  "use server";
  if (!q || q.length < 2) return [];
  // Cap query length to prevent pathological LIKE patterns
  const trimmed = q.slice(0, 200);
  return await searchArticles(trimmed, 30);
}, "search");

export const route = {
  load: ({ location }: { location: { query: Record<string, string> } }) => {
    const q = location.query.q || "";
    return loadSearch(q);
  },
};

export default function SearchPage() {
  const [searchParams] = useSearchParams();
  const q = () => searchParams.q || "";
  const results = createAsync(() => loadSearch(q()));

  return (
    <main class="site-page lagoon-subpage search-dashboard-page">
      <OGMeta
        title="Search Football News"
        description="Search football articles in Tuvaluan and English"
        url={absoluteFootballUrl("/search")}
        image={FOOTBALL_META.searchOgImage}
        imageOrigin={SITE_ORIGINS.football}
        imageWidth={FOOTBALL_META.defaultOgImageWidth}
        imageHeight={FOOTBALL_META.defaultOgImageHeight}
        imageAlt="Talafutipolo social card for searching football news."
        siteName={FOOTBALL_META.productName}
        titleSuffix={FOOTBALL_META.productName}
        robots={NOINDEX_ROBOTS}
      />

      <section class="site-hero site-hero--compact lagoon-subhero">
        <div class="site-shell site-shell--wide lagoon-subhero__grid">
          <div>
            <p class="site-kicker">Saili</p>
            <h1 class="site-title">Search the Tuvaluan football wire.</h1>
            <p class="site-lede">
              Find stories, sources, names, and translated football phrases across
              the Talafutipolo archive.
            </p>
          </div>
          <aside class="lagoon-subhero__panel">
            <span>Archive mode</span>
            <strong>TVL</strong>
            <em>Tuvaluan-first search</em>
          </aside>
        </div>
      </section>

      <div class="site-shell site-shell--wide search-dashboard">
        {/* Search form */}
        <section class="search-console">
          <form action="/search" method="get" class="search-form" role="search">
            <label for="search-input" class="sr-only">Search articles</label>
            <input
              id="search-input"
              type="search"
              name="q"
              value={q()}
              maxLength={200}
              placeholder="Saili tala... (Search articles)"
              class="search-form__input"
            />
            <button
              type="submit"
              class="site-button site-button--primary search-form__button"
            >
              Saili
            </button>
          </form>
          <div class="search-console__chips" aria-label="Suggested searches">
            <A href="/search?q=world%20cup">World Cup</A>
            <A href="/search?q=tuvalu">Tuvalu</A>
            <A href="/search?q=training">Training</A>
          </div>
        </section>

        {/* Results */}
        <Show when={q().length >= 2}>
          <Show
            when={results() && results()!.length > 0}
            fallback={
              <div class="site-empty">
                <strong>Seki kitea</strong>
                <span>No results for "{q()}"</span>
              </div>
            }
          >
            <div class="site-section-head search-results-head">
              <div>
                <p class="site-kicker">Results</p>
                <h2 class="site-section-title">
                  {results()!.length} tala ne kitea
                </h2>
              </div>
            </div>
            <div class="home-card-grid search-results-grid">
              <For each={results()!}>
                {(article) => <ArticleCard article={article} tile />}
              </For>
            </div>
          </Show>
        </Show>
      </div>
    </main>
  );
}
