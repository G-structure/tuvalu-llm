import { absoluteFootballUrl } from "~/lib/site";

export async function GET() {
  const body = `User-agent: *\nAllow: /\n\nSitemap: ${absoluteFootballUrl("/sitemap.xml")}\n`;
  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "public, max-age=3600",
    },
  });
}
