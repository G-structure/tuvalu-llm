export const SITE_URL = "https://tuvalugpt.tv";

export const SITE_META = {
  productName: "Talafutipolo",
  productTagline: "Tuvaluan football news in Tuvaluan and English.",
  publicationName: "Language Lab Journal",
  publicationShortName: "Language Lab",
  publicationDescription:
    "Product updates, research notes, field reports, and open-source writing from the Language Lab team building Tuvaluan AI tools.",
  defaultOgImage: "/social/language-lab-blog.jpg",
  defaultOgImageWidth: 1200,
  defaultOgImageHeight: 630,
  defaultOgImageAlt:
    "Language Lab Journal social card with an ocean sky backdrop and editorial branding.",
  feeds: {
    blogRss: "/blog/feed.xml",
    blogJson: "/blog/feed.json",
    articlesRss: "/feed.xml",
  },
} as const;

export function absoluteUrl(path = "/"): string {
  return new URL(path, SITE_URL).toString();
}

export function absoluteImageUrl(path?: string | null): string {
  if (!path) return absoluteUrl(SITE_META.defaultOgImage);
  if (/^https?:\/\//i.test(path)) return path;
  return absoluteUrl(path);
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
