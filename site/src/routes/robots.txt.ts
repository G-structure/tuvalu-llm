import type { APIEvent } from "@solidjs/start/server";

export async function GET(event: APIEvent) {
  const origin = new URL(event.request.url).origin;
  const body = `User-agent: *\nAllow: /\n\nSitemap: ${new URL("/sitemap.xml", origin).toString()}\n`;
  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "public, max-age=3600",
    },
  });
}
