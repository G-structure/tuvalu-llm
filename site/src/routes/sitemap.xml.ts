import {
  getAllPosts,
  getAllTags,
  getArchiveGroups,
} from "~/lib/blog-data";
import { getAllAuthors } from "~/lib/blog-authors";
import { getArticleCount, getArticles, getCategories } from "~/lib/db";
import { absoluteChatUrl, absoluteFootballUrl, absoluteUrl } from "~/lib/site";

function xmlEscape(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export async function GET() {
  const [articleCount, categories] = await Promise.all([
    getArticleCount(),
    getCategories(),
  ]);
  const articles = articleCount > 0 ? await getArticles(articleCount, 0) : [];
  const posts = getAllPosts();
  const tags = getAllTags();
  const archive = getArchiveGroups();
  const authors = getAllAuthors();

  const entries = [
    { loc: absoluteFootballUrl("/"), lastmod: undefined },
    { loc: absoluteUrl("/blog"), lastmod: posts[0]?.updatedAt || posts[0]?.publishedAt },
    { loc: absoluteUrl("/blog/archive"), lastmod: archive[0]?.posts[0]?.publishedAt },
    { loc: absoluteUrl("/demo"), lastmod: undefined },
    { loc: absoluteFootballUrl("/fatele"), lastmod: undefined },
    { loc: absoluteFootballUrl("/search"), lastmod: undefined },
    { loc: absoluteChatUrl("/chat"), lastmod: undefined },
    { loc: absoluteChatUrl("/chat/eval"), lastmod: undefined },
    { loc: absoluteChatUrl("/chat/training"), lastmod: undefined },
    ...posts.map((post) => ({
      loc: absoluteUrl(`/blog/${post.slug}`),
      lastmod: post.updatedAt || post.publishedAt,
    })),
    ...tags.map((tag) => ({
      loc: absoluteUrl(`/blog/tag/${tag.slug}`),
      lastmod: getAllPosts()
        .find((post) => post.tagSlugs.includes(tag.slug))
        ?.publishedAt,
    })),
    ...authors.map((author) => ({
      loc: absoluteUrl(`/blog/author/${author.slug}`),
      lastmod: getAllPosts()
        .find((post) => post.authors.some((item) => item.slug === author.slug))
        ?.publishedAt,
    })),
    ...categories.map((category) => ({
      loc: absoluteFootballUrl(`/category/${category.slug}`),
      lastmod: undefined,
    })),
    ...articles.map((article) => ({
      loc: absoluteFootballUrl(`/articles/${article.id}`),
      lastmod: article.published_at || undefined,
    })),
  ];

  const body = entries
    .map(
      (entry) => `  <url>
    <loc>${xmlEscape(entry.loc)}</loc>${entry.lastmod ? `\n    <lastmod>${xmlEscape(entry.lastmod)}</lastmod>` : ""}
  </url>`
    )
    .join("\n");

  return new Response(
    `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${body}\n</urlset>`,
    {
      status: 200,
      headers: {
        "Content-Type": "application/xml; charset=utf-8",
        "Cache-Control": "public, max-age=3600",
      },
    }
  );
}
