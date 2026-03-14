import { createAsync, cache, useSearchParams, A } from "@solidjs/router";
import { For, Show } from "solid-js";
import { getArticles, getCategories } from "~/lib/db";
import type { Article, Category } from "~/lib/types";
import ArticleCard from "~/components/ArticleCard";
import CategoryPills from "~/components/CategoryPills";
import OGMeta from "~/components/OGMeta";

const PER_PAGE = 20;

const loadHome = cache(async (page: number) => {
  "use server";
  const offset = (page - 1) * PER_PAGE;
  const [articles, categories] = await Promise.all([
    getArticles(PER_PAGE + 1, offset),
    getCategories(),
  ]);
  return { articles, categories, page };
}, "home");

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

  return (
    <main class="max-w-3xl mx-auto pb-8">
      <OGMeta
        title="Talafutipolo Tuvalu — Tala Futipolo i te Gagana Tuvalu"
        description="Tala futipolo mai te lalolagi i te gagana Tuvalu. Football news from around the world in the Tuvaluan language."
      />

      <Show when={data()}>
        {(d) => {
          const articles = () => d().articles.slice(0, PER_PAGE);
          const hasNext = () => d().articles.length > PER_PAGE;
          const hasPrev = () => d().page > 1;

          return (
            <>
              {/* Category filter pills */}
              <Show when={d().categories.length > 0}>
                <CategoryPills categories={d().categories} />
              </Show>

              {/* Empty state */}
              <Show when={articles().length === 0}>
                <div class="p-8 text-center text-gray-400">
                  <p class="text-lg font-medium">Seki isi tala</p>
                  <p class="mt-2 text-sm">No articles yet</p>
                </div>
              </Show>

              {/* Hero card for latest article */}
              <Show when={articles().length > 0}>
                <div class="px-4 pt-2">
                  <ArticleCard article={articles()[0]} hero />
                </div>
              </Show>

              {/* Remaining articles as thumbnail rows */}
              <Show when={articles().length > 1}>
                <div class="mt-4 divide-y divide-gray-100">
                  <For each={articles().slice(1)}>
                    {(article) => <ArticleCard article={article} />}
                  </For>
                </div>
              </Show>

              {/* Pagination */}
              <Show when={hasPrev() || hasNext()}>
                <div class="px-4 mt-4 flex gap-3">
                  <Show when={hasPrev()}>
                    <A
                      href={d().page === 2 ? "/" : `/?page=${d().page - 1}`}
                      class="flex-1 py-3 text-center text-sm font-medium text-[var(--ocean-deep)] bg-white rounded-lg no-underline hover:bg-[var(--sky-dark)] transition-colors border border-[var(--sky-dark)]"
                    >
                      &larr; Foki
                    </A>
                  </Show>
                  <Show when={hasNext()}>
                    <A
                      href={`/?page=${d().page + 1}`}
                      class="flex-1 py-3 text-center text-sm font-medium text-[var(--ocean-deep)] bg-white rounded-lg no-underline hover:bg-[var(--sky-dark)] transition-colors border border-[var(--sky-dark)]"
                    >
                      Faitau atu &darr;
                    </A>
                  </Show>
                </div>
              </Show>
            </>
          );
        }}
      </Show>
    </main>
  );
}
