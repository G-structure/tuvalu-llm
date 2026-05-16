export const SITE_URL = "https://tuvalugpt.tv";

export const SITE_ORIGINS = {
  football: "https://futipolo.tv",
  organization: "https://tuvalugpt.tv",
  chat: "https://tuvalugpt.tv",
} as const;

export const FOOTBALL_META = {
  productName: "Talafutipolo",
  productTagline: "Tuvaluan football news in Tuvaluan and English.",
  defaultOgImage: "/social/talafutipolo-football.jpg",
  articleFallbackOgImage: "/social/football-free-kick-fallback.jpg",
  articleFallbackImageWidth: 1200,
  articleFallbackImageHeight: 675,
  articleFallbackImageAlt:
    "A cinematic football free-kick under stadium lights for Talafutipolo articles.",
  searchOgImage: "/social/football-search.jpg",
  communityOgImage: "/social/football-community.jpg",
  notFoundOgImage: "/social/football-not-found.jpg",
  defaultOgImageWidth: 1200,
  defaultOgImageHeight: 630,
  defaultOgImageAlt:
    "Talafutipolo social card for football news in Tuvaluan and English.",
  feeds: {
    articlesRss: "/feed.xml",
  },
} as const;

export const FOOTBALL_CATEGORY_OG_IMAGES: Record<string, string> = {
  football: "/social/category-football.jpg",
  "world-cup": "/social/category-world-cup.jpg",
  "premier-league": "/social/category-premier-league.jpg",
  transfers: "/social/category-transfers.jpg",
  scottish: "/social/category-scottish.jpg",
  championship: "/social/category-championship.jpg",
};

export const CHAT_META = {
  productName: "TVL Chat",
  productTagline:
    "A bilingual Tuvaluan-English language model for translation, chat, and evaluation.",
  defaultOgImage: "/social/tuvalu-chat.jpg",
  evalOgImage: "/social/chat-eval.jpg",
  trainingOgImage: "/social/chat-training.jpg",
  defaultOgImageWidth: 1200,
  defaultOgImageHeight: 630,
  defaultOgImageAlt:
    "TVL Chat social card for a bilingual Tuvaluan-English language model.",
} as const;

export const SITE_META = {
  productName: "Language Lab",
  productTagline: "Open-source AI infrastructure for endangered languages.",
  publicationName: "Language Lab Journal",
  publicationShortName: "Language Lab",
  publicationDescription:
    "Product updates, research notes, field reports, and open-source writing from the Language Lab team building Tuvaluan AI tools.",
  defaultOgImage: "/social/language-lab-blog.jpg",
  defaultOgImageWidth: 1200,
  defaultOgImageHeight: 630,
  defaultOgImageAlt:
    "Language Lab Journal social card for open-source AI infrastructure for endangered languages.",
  feeds: {
    blogRss: "/blog/feed.xml",
    blogJson: "/blog/feed.json",
    articlesRss: "/feed.xml",
  },
} as const;

export const BLOG_SECTION_OG_IMAGES = {
  archive: "/social/blog-archive.jpg",
  authorLanguageLab: "/social/blog-author-language-lab.jpg",
  tags: {
    evaluation: "/social/blog-tag-evaluation.jpg",
    technical: "/social/blog-tag-technical.jpg",
    training: "/social/blog-tag-training.jpg",
  },
} as const;

export function absoluteUrl(path = "/", origin = SITE_URL): string {
  return new URL(path, origin).toString();
}

export function absoluteFootballUrl(path = "/"): string {
  return absoluteUrl(path, SITE_ORIGINS.football);
}

export function absoluteChatUrl(path = "/"): string {
  return absoluteUrl(path, SITE_ORIGINS.chat);
}

export function absoluteImageUrl(path?: string | null, origin = SITE_URL): string {
  if (!path) return absoluteUrl(SITE_META.defaultOgImage);
  if (/^https?:\/\//i.test(path)) return path;
  return absoluteUrl(path, origin);
}

export function absoluteFootballImageUrl(path?: string | null): string {
  if (!path) return absoluteFootballUrl(FOOTBALL_META.defaultOgImage);
  if (/^https?:\/\//i.test(path)) return path;
  return absoluteFootballUrl(path);
}

export function absoluteChatImageUrl(path?: string | null): string {
  if (!path) return absoluteChatUrl(CHAT_META.defaultOgImage);
  if (/^https?:\/\//i.test(path)) return path;
  return absoluteChatUrl(path);
}

export function parseIsoLikeDate(value?: string | null): Date | null {
  if (!value) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return new Date(`${value}T12:00:00Z`);
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

export function formatLongDate(value?: string | null): string {
  const parsed = parseIsoLikeDate(value);
  if (!parsed) return value || "";
  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

export function formatShortDate(value?: string | null): string {
  const parsed = parseIsoLikeDate(value);
  if (!parsed) return value || "";
  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function formatMonthYear(value?: string | null): string {
  const parsed = parseIsoLikeDate(value);
  if (!parsed) return value || "";
  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
  });
}
