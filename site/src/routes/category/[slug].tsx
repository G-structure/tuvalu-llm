import { createAsync, cache, useParams, useSearchParams, A } from "@solidjs/router";
import { For, Show } from "solid-js";
import { getArticles, getCategories } from "~/lib/db";
import type { Article, Category } from "~/lib/types";
import ArticleCard from "~/components/ArticleCard";
import CategoryPills from "~/components/CategoryPills";
import OGMeta from "~/components/OGMeta";

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

  return (
    <main class="max-w-3xl mx-auto pb-8">
      <OGMeta
        title={`${displayName()} | TALAFUTIPOLO`}
        description={`${displayName()} football news in Tuvaluan and English`}
      />

      <Show when={data()}>
        {(d) => {
          const articles = () => d().articles.slice(0, PER_PAGE);
          const hasNext = () => d().articles.length > PER_PAGE;
          const hasPrev = () => d().page > 1;

          return (
            <>
              {/* Category filter pills */}
              <CategoryPills categories={d().categories} />

              {/* Category heading */}
              <div class="px-4 pt-2 pb-2">
                <h1 class="text-xl font-bold text-gray-900 capitalize">
                  {displayName()}
                </h1>
              </div>

              {/* Empty state */}
              <Show when={articles().length === 0}>
                <div class="p-8 text-center text-gray-400">
                  <p class="text-lg font-medium">Seki isi tala</p>
                  <p class="mt-1 text-sm">No articles in this category</p>
                </div>
              </Show>

              {/* Hero card for first article */}
              <Show when={articles().length > 0}>
                <div class="px-4">
                  <ArticleCard article={articles()[0]} hero />
                </div>
              </Show>

              {/* Remaining articles */}
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
                      href={d().page === 2 ? `/category/${params.slug}` : `/category/${params.slug}?page=${d().page - 1}`}
                      class="flex-1 py-3 text-center text-sm font-medium text-[var(--ocean-deep)] bg-white rounded-lg no-underline hover:bg-[var(--sky-dark)] transition-colors border border-[var(--sky-dark)]"
                    >
                      &larr; Foki
                    </A>
                  </Show>
                  <Show when={hasNext()}>
                    <A
                      href={`/category/${params.slug}?page=${d().page + 1}`}
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
