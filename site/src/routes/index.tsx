import { createAsync, cache, useSearchParams, A } from "@solidjs/router";
import { For, Show } from "solid-js";
import { getArticles, getCategories } from "~/lib/db";
import type { Article, Category } from "~/lib/types";
import ArticleCard from "~/components/ArticleCard";
import CategoryPills from "~/components/CategoryPills";
import OGMeta from "~/components/OGMeta";
import { absoluteUrl } from "~/lib/site";

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
        title="Talafutipolo"
        description="Tala futipolo mai te lalolagi i te gagana Tuvalu. Football news from around the world in the Tuvaluan language."
        url={absoluteUrl("/")}
        siteName="Talafutipolo"
        titleSuffix="Tala Futipolo i te Gagana Tuvalu"
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

              <Show when={articles().length > 0}>
                <div class="px-4 pt-3">
                  <div class="rounded-2xl border border-[var(--gold)]/50 bg-[var(--ocean-deep)] text-white p-4">
                    <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <p class="text-xs uppercase tracking-[0.2em] text-[var(--gold)]">
                          Kominiti
                        </p>
                        <h2 class="mt-1 text-lg font-bold">
                          Help coach the Tuvaluan model
                        </h2>
                        <p class="mt-1 text-sm text-[var(--sky-dark)]">
                          Open a story, vote on the translation, choose your
                          preferred reading mode, and leave a better phrasing.
                        </p>
                      </div>
                      <div class="flex gap-2">
                        <A
                          href={`/articles/${articles()[0].id}`}
                          class="rounded-xl bg-[var(--gold)] px-4 py-2 text-sm font-semibold text-[var(--ocean-deep)] no-underline"
                        >
                          Coach latest story
                        </A>
                        <A
                          href="/fatele"
                          class="rounded-xl border border-white/20 px-4 py-2 text-sm font-semibold text-white no-underline"
                        >
                          View community
                        </A>
                      </div>
                    </div>
                  </div>
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
