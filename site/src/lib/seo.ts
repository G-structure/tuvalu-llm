import type { Article } from "./types";
import {
  absoluteFootballImageUrl,
  absoluteFootballUrl,
  absoluteImageUrl,
  absoluteUrl,
  FOOTBALL_META,
  SITE_META,
  SITE_ORIGINS,
} from "./site";

interface BreadcrumbItem {
  name: string;
  url: string;
}

export const INDEXABLE_ROBOTS =
  "index,follow,max-image-preview:large,max-snippet:-1,max-video-preview:-1";

export const NOINDEX_ROBOTS = "noindex,follow,max-image-preview:large";

export function languageLabOrganization() {
  return {
    "@context": "https://schema.org",
    "@type": "Organization",
    "@id": `${SITE_ORIGINS.organization}/#organization`,
    name: SITE_META.productName,
    url: SITE_ORIGINS.organization,
    logo: absoluteUrl("/icons/icon-512.png"),
    sameAs: [SITE_ORIGINS.football],
  };
}

export function footballWebsite() {
  return {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "@id": `${SITE_ORIGINS.football}/#website`,
    name: FOOTBALL_META.productName,
    alternateName: "Tala Futipolo",
    url: SITE_ORIGINS.football,
    publisher: {
      "@id": `${SITE_ORIGINS.organization}/#organization`,
    },
    inLanguage: ["tvl", "en"],
    potentialAction: {
      "@type": "SearchAction",
      target: `${absoluteFootballUrl("/search")}?q={search_term_string}`,
      "query-input": "required name=search_term_string",
    },
  };
}

export function languageLabWebsite() {
  return {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "@id": `${SITE_ORIGINS.organization}/#website`,
    name: SITE_META.productName,
    url: SITE_ORIGINS.organization,
    publisher: {
      "@id": `${SITE_ORIGINS.organization}/#organization`,
    },
    inLanguage: ["en", "tvl"],
  };
}

export function breadcrumbList(items: BreadcrumbItem[]) {
  return {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: items.map((item, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: item.name,
      item: item.url,
    })),
  };
}

export function footballCollectionPage(props: {
  name: string;
  description: string;
  url: string;
  image?: string | null;
}) {
  return {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    name: props.name,
    description: props.description,
    url: props.url,
    image: absoluteFootballImageUrl(props.image || FOOTBALL_META.defaultOgImage),
    isPartOf: {
      "@id": `${SITE_ORIGINS.football}/#website`,
    },
    publisher: {
      "@id": `${SITE_ORIGINS.organization}/#organization`,
    },
    inLanguage: ["tvl", "en"],
  };
}

export function footballNewsArticleStructuredData(article: Article) {
  const title = article.title_tvl || article.title_en;
  const description =
    article.og_description_tvl || article.og_description_en || article.title_en;
  const url = absoluteFootballUrl(`/articles/${article.id}`);
  const image = absoluteFootballImageUrl(
    article.image_url || FOOTBALL_META.articleFallbackOgImage
  );
  const keywords = parseArticleTags(article.tags);

  return [
    {
      "@context": "https://schema.org",
      "@type": "NewsArticle",
      mainEntityOfPage: {
        "@type": "WebPage",
        "@id": url,
      },
      headline: title,
      alternativeHeadline: article.title_tvl ? article.title_en : undefined,
      description,
      image: [image],
      datePublished: article.published_at,
      dateModified: article.published_at,
      author: article.author
        ? [{ "@type": "Person", name: article.author }]
        : [{ "@type": "Organization", name: sourceName(article.source_id) }],
      publisher: {
        "@id": `${SITE_ORIGINS.organization}/#organization`,
      },
      isPartOf: {
        "@id": `${SITE_ORIGINS.football}/#website`,
      },
      articleSection: article.category || "football",
      keywords,
      wordCount: article.word_count || undefined,
      inLanguage: article.title_tvl ? ["tvl", "en"] : ["en"],
      isBasedOn: article.url,
    },
    breadcrumbList([
      { name: FOOTBALL_META.productName, url: absoluteFootballUrl("/") },
      {
        name: article.category
          ? article.category.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
          : "Football",
        url: absoluteFootballUrl(`/category/${article.category || "football"}`),
      },
      { name: title, url },
    ]),
  ];
}

export function sourceName(sourceId: string): string {
  const map: Record<string, string> = {
    goal: "Goal.com",
    fifa: "FIFA.com",
    sky: "Sky Sports",
  };
  return map[sourceId] || sourceId;
}

export function blogImageUrl(path?: string | null): string {
  return absoluteImageUrl(path || SITE_META.defaultOgImage, SITE_ORIGINS.organization);
}

function parseArticleTags(value?: string | null): string[] | undefined {
  if (!value) return undefined;
  try {
    const parsed = JSON.parse(value);
    if (Array.isArray(parsed)) {
      return parsed.filter((item): item is string => typeof item === "string");
    }
  } catch {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return undefined;
}
