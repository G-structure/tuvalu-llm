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
    <main class="max-w-3xl mx-auto pb-16 px-4">
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

      <div class="pt-6 pb-4">
        <h1 class="text-xl font-bold text-gray-900">Saili (Search)</h1>
      </div>

      {/* Search form */}
      <form action="/search" method="get" class="mb-6" role="search">
        <div class="flex gap-2">
          <label for="search-input" class="sr-only">Search articles</label>
          <input
            id="search-input"
            type="search"
            name="q"
            value={q()}
            maxLength={200}
            placeholder="Saili tala... (Search articles)"
            class="flex-1 px-4 py-3 border border-[var(--sky-dark)] rounded-lg text-sm bg-white focus:outline-none focus:border-[var(--ocean-bright)]"
          />
          <button
            type="submit"
            class="px-5 py-3 bg-[var(--ocean-deep)] text-white rounded-lg text-sm font-medium cursor-pointer border-none hover:bg-[var(--ocean)] transition-colors"
          >
            Saili
          </button>
        </div>
      </form>

      {/* Results */}
      <Show when={q().length >= 2}>
        <Show
          when={results() && results()!.length > 0}
          fallback={
            <div class="p-8 text-center text-gray-400">
              <p class="text-lg font-medium">Seki kitea</p>
              <p class="mt-2 text-sm">
                No results for "{q()}"
              </p>
            </div>
          }
        >
          <p class="text-sm text-gray-500 mb-4">
            {results()!.length} tala ne kitea (results found)
          </p>
          <div class="divide-y divide-gray-100">
            <For each={results()!}>
              {(article) => <ArticleCard article={article} />}
            </For>
          </div>
        </Show>
      </Show>
    </main>
  );
}
