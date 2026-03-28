import { A } from "@solidjs/router";
import { For } from "solid-js";
import { createAsync, cache } from "@solidjs/router";
import { parseBlogMeta, type BlogPost } from "~/lib/blog";
import OGMeta from "~/components/OGMeta";

const getBlogPosts = cache(async (): Promise<BlogPost[]> => {
  "use server";
  const fs = await import("fs");
  const path = await import("path");

  const postsDir = path.join(process.cwd(), "public", "blog", "posts");
  if (!fs.existsSync(postsDir)) return [];

  const files = fs.readdirSync(postsDir).filter((f: string) => f.endsWith(".md"));
  const posts: BlogPost[] = [];

  for (const file of files) {
    const raw = fs.readFileSync(path.join(postsDir, file), "utf-8");
    const slug = file.replace(/\.md$/, "");
    posts.push(parseBlogMeta(slug, raw));
  }

  // Sort by date descending
  posts.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  return posts;
}, "blog-posts");

export const route = { load: () => getBlogPosts() };

export default function BlogIndex() {
  const posts = createAsync(() => getBlogPosts());

  return (
    <main class="blog-page">
      <OGMeta
        title="Blog — Language Lab"
        description="Technical deep dives, project updates, and stories from building AI for endangered languages."
        url="https://tuvalugpt.tv/blog"
      />

      <div class="blog-shell">
        <header class="blog-index-header">
          <h1 class="blog-index-title">Blog</h1>
          <p class="blog-index-desc">
            Technical deep dives, project updates, and stories from building AI for endangered languages.
          </p>
        </header>

        <div class="blog-index-list">
          <For each={posts()} fallback={<p>No posts yet.</p>}>
            {(post) => (
              <A href={`/blog/${post.slug}`} class="blog-card">
                {post.image && (
                  <img
                    src={post.image}
                    alt=""
                    class="blog-card__img"
                    loading="lazy"
                    decoding="async"
                  />
                )}
                <div class="blog-card__body">
                  <time class="blog-card__date">{formatDate(post.date)}</time>
                  <h2 class="blog-card__title">{post.title}</h2>
                  <p class="blog-card__desc">{post.description}</p>
                  {post.tags && (
                    <div class="blog-card__tags">
                      <For each={post.tags}>
                        {(tag) => <span class="blog-card__tag">{tag}</span>}
                      </For>
                    </div>
                  )}
                </div>
              </A>
            )}
          </For>
        </div>
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
