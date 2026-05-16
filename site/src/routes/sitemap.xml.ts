import {
  getAllPosts,
  getAllTags,
  getArchiveGroups,
} from "~/lib/blog-data";
import { getAllAuthors } from "~/lib/blog-authors";
import { getArticleCount, getArticles, getCategories } from "~/lib/db";
import type { APIEvent } from "@solidjs/start/server";
import {
  absoluteChatUrl,
  absoluteFootballImageUrl,
  absoluteFootballUrl,
  absoluteImageUrl,
  absoluteUrl,
  FOOTBALL_CATEGORY_OG_IMAGES,
  FOOTBALL_META,
  SITE_META,
  SITE_ORIGINS,
} from "~/lib/site";

interface SitemapEntry {
  loc: string;
  lastmod?: string;
  image?: string;
  imageTitle?: string;
}

function xmlEscape(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function urlEntry(entry: SitemapEntry): string {
  const image = entry.image
    ? `\n    <image:image>\n      <image:loc>${xmlEscape(entry.image)}</image:loc>${entry.imageTitle ? `\n      <image:title>${xmlEscape(entry.imageTitle)}</image:title>` : ""}\n    </image:image>`
    : "";
  return `  <url>
    <loc>${xmlEscape(entry.loc)}</loc>${entry.lastmod ? `\n    <lastmod>${xmlEscape(entry.lastmod)}</lastmod>` : ""}${image}
  </url>`;
}

export async function GET(event: APIEvent) {
  const host = new URL(event.request.url).hostname;
  const isFootballHost = host === new URL(SITE_ORIGINS.football).hostname;
  const [articleCount, categories] = await Promise.all([
    getArticleCount(),
    getCategories(),
  ]);
  const articles = articleCount > 0 ? await getArticles(articleCount, 0) : [];
  const posts = getAllPosts();
  const tags = getAllTags();
  const archive = getArchiveGroups();
  const authors = getAllAuthors();

  const footballEntries: SitemapEntry[] = [
    {
      loc: absoluteFootballUrl("/"),
      image: absoluteFootballImageUrl(FOOTBALL_META.defaultOgImage),
      imageTitle: FOOTBALL_META.productName,
    },
    {
      loc: absoluteFootballUrl("/fatele"),
      image: absoluteFootballImageUrl(FOOTBALL_META.communityOgImage),
      imageTitle: "Kominiti",
    },
    {
      loc: absoluteFootballUrl("/legal"),
      image: absoluteFootballImageUrl(FOOTBALL_META.defaultOgImage),
      imageTitle: "Privacy Policy and Terms of Service",
    },
    ...categories.map((category) => ({
      loc: absoluteFootballUrl(`/category/${category.slug}`),
      image: absoluteFootballImageUrl(
        FOOTBALL_CATEGORY_OG_IMAGES[category.slug] || FOOTBALL_META.defaultOgImage
      ),
      imageTitle: category.slug.replace(/-/g, " "),
    })),
    ...articles.map((article) => ({
      loc: absoluteFootballUrl(`/articles/${article.id}`),
      lastmod: article.published_at || undefined,
      image: absoluteFootballImageUrl(article.image_url || FOOTBALL_META.articleFallbackOgImage),
      imageTitle: article.title_tvl || article.title_en,
    })),
  ];

  const organizationEntries: SitemapEntry[] = [
    {
      loc: absoluteUrl("/demo"),
      image: absoluteImageUrl("/judges/rainbow-ocean.webp"),
      imageTitle: SITE_META.productName,
    },
    {
      loc: absoluteUrl("/legal"),
      image: absoluteImageUrl(SITE_META.defaultOgImage),
      imageTitle: "Privacy Policy and Terms of Service",
    },
    {
      loc: absoluteChatUrl("/chat"),
      image: absoluteImageUrl("/social/tuvalu-chat.jpg"),
      imageTitle: "TVL Chat",
    },
    {
      loc: absoluteChatUrl("/chat/eval"),
      image: absoluteImageUrl("/social/chat-eval.jpg"),
      imageTitle: "TVL Model Evaluation",
    },
    {
      loc: absoluteChatUrl("/chat/training"),
      image: absoluteImageUrl("/social/chat-training.jpg"),
      imageTitle: "TVL Training Dashboard",
    },
    {
      loc: absoluteUrl("/blog"),
      lastmod: posts[0]?.updatedAt || posts[0]?.publishedAt,
      image: absoluteImageUrl(SITE_META.defaultOgImage),
      imageTitle: SITE_META.publicationName,
    },
    {
      loc: absoluteUrl("/blog/archive"),
      lastmod: archive[0]?.posts[0]?.publishedAt,
      image: absoluteImageUrl("/social/blog-archive.jpg"),
      imageTitle: `${SITE_META.publicationName} archive`,
    },
    ...posts.map((post) => ({
      loc: absoluteUrl(`/blog/${post.slug}`),
      lastmod: post.updatedAt || post.publishedAt,
      image: absoluteImageUrl(post.socialImage || post.image || SITE_META.defaultOgImage),
      imageTitle: post.title,
    })),
    ...tags.map((tag) => ({
      loc: absoluteUrl(`/blog/tag/${tag.slug}`),
      lastmod: posts
        .find((post) => post.tagSlugs.includes(tag.slug))
        ?.publishedAt,
      image: absoluteImageUrl(
        `/social/blog-tag-${tag.slug}.jpg`
      ),
      imageTitle: `${tag.name} — ${SITE_META.publicationName}`,
    })),
    ...authors.map((author) => ({
      loc: absoluteUrl(`/blog/author/${author.slug}`),
      lastmod: posts
        .find((post) => post.authors.some((item) => item.slug === author.slug))
        ?.publishedAt,
      image: absoluteImageUrl("/social/blog-author-language-lab.jpg"),
      imageTitle: `${author.name} — ${SITE_META.publicationName}`,
    })),
  ];

  const entries = isFootballHost ? footballEntries : organizationEntries;
  const body = entries.map(urlEntry).join("\n");

  return new Response(
    `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">\n${body}\n</urlset>`,
    {
      status: 200,
      headers: {
        "Content-Type": "application/xml; charset=utf-8",
        "Cache-Control": "public, max-age=3600",
      },
    }
  );
}
