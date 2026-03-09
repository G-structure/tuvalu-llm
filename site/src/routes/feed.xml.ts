import { getArticles } from "~/lib/db";

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export async function GET() {
  const articles = await getArticles(20, 0);

  const items = articles
    .map((a) => {
      const title = a.title_tvl || a.title_en;
      const description =
        a.og_description_tvl || a.og_description_en || a.title_en;
      const link = `https://talafutipolo.tv/articles/${a.id}`;
      const pubDate = a.published_at
        ? new Date(a.published_at).toUTCString()
        : "";

      return `    <item>
      <title>${escapeXml(title)}</title>
      <link>${escapeXml(link)}</link>
      <description>${escapeXml(description)}</description>
      <guid isPermaLink="true">${escapeXml(link)}</guid>${pubDate ? `\n      <pubDate>${pubDate}</pubDate>` : ""}
      <source url="https://talafutipolo.tv/feed.xml">TALAFUTIPOLO</source>
    </item>`;
    })
    .join("\n");

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>TALAFUTIPOLO - Tala Futipolo i te Gagana Tuvalu</title>
    <link>https://talafutipolo.tv</link>
    <description>Football news in the Tuvaluan language</description>
    <language>tvl</language>
    <atom:link href="https://talafutipolo.tv/feed.xml" rel="self" type="application/rss+xml"/>
${items}
  </channel>
</rss>`;

  return new Response(xml, {
    status: 200,
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8",
      "Cache-Control": "public, max-age=3600",
    },
  });
}
