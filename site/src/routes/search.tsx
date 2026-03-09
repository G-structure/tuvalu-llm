import { createAsync, cache, useSearchParams, A } from "@solidjs/router";
import { For, Show } from "solid-js";
import { searchArticles } from "~/lib/db";
import ArticleCard from "~/components/ArticleCard";
import OGMeta from "~/components/OGMeta";

const loadSearch = cache(async (q: string) => {
  "use server";
  if (!q || q.length < 2) return [];
  return await searchArticles(q, 30);
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
        title="Saili | TALAFUTIPOLO"
        description="Search football articles in Tuvaluan and English"
      />

      <div class="pt-6 pb-4">
        <h1 class="text-xl font-bold text-gray-900">Saili (Search)</h1>
      </div>

      {/* Search form */}
      <form action="/search" method="get" class="mb-6">
        <div class="flex gap-2">
          <input
            type="text"
            name="q"
            value={q()}
            placeholder="Saili tala... (Search articles)"
            class="flex-1 px-4 py-3 border border-gray-200 rounded-lg text-sm bg-white focus:outline-none focus:border-gray-400"
          />
          <button
            type="submit"
            class="px-5 py-3 bg-[#1a1a2e] text-white rounded-lg text-sm font-medium cursor-pointer border-none hover:bg-[#2a2a4e] transition-colors"
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
