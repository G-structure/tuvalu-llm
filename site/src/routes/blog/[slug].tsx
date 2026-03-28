import { A, useParams } from "@solidjs/router";
import { Show } from "solid-js";
import { createAsync, cache } from "@solidjs/router";
import { parseBlogPost, type BlogPostFull } from "~/lib/blog";
import OGMeta from "~/components/OGMeta";

const getPost = cache(async (slug: string): Promise<BlogPostFull | null> => {
  "use server";
  const fs = await import("fs");
  const path = await import("path");

  const filePath = path.join(process.cwd(), "public", "blog", "posts", `${slug}.md`);
  if (!fs.existsSync(filePath)) return null;

  const raw = fs.readFileSync(filePath, "utf-8");
  return parseBlogPost(slug, raw);
}, "blog-post");

export const route = {
  load: ({ params }: { params: { slug: string } }) => getPost(params.slug),
};

export default function BlogPostPage() {
  const params = useParams<{ slug: string }>();
  const post = createAsync(() => getPost(params.slug));

  return (
    <Show when={post()} fallback={<NotFound />}>
      {(p) => (
        <main class="blog-page">
          <OGMeta
            title={p().title}
            description={p().description}
            image={p().image}
            url={`https://tuvalugpt.tv/blog/${p().slug}`}
          />

          <article class="blog-shell blog-post">
            <header class="blog-post__header">
              <A href="/blog" class="blog-post__back">&larr; All posts</A>
              <time class="blog-post__date">{formatDate(p().date)}</time>
              <h1 class="blog-post__title">{p().title}</h1>
            </header>

            <div class="blog-content" innerHTML={p().html} />

            <footer class="blog-post__footer">
              <A href="/blog" class="blog-post__back">&larr; All posts</A>
            </footer>
          </article>
        </main>
      )}
    </Show>
  );
}

function NotFound() {
  return (
    <main class="blog-page">
      <div class="blog-shell" style={{ "text-align": "center", "padding-top": "4rem" }}>
        <h1>Post not found</h1>
        <A href="/blog">&larr; Back to blog</A>
      </div>
    </main>
  );
}

function formatDate(d: string): string {
  if (!d) return "";
  try {
    return new Date(d + "T00:00:00").toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  } catch {
    return d;
  }
}
