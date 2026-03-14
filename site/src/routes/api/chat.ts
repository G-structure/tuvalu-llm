import type { APIEvent } from "@solidjs/start/server";

function getBackendUrl(event: APIEvent): string {
  // Runtime: Cloudflare Pages secrets are in event.context.cloudflare.env
  const cfEnv = (event.context as any)?.cloudflare?.env;
  return cfEnv?.CHAT_BACKEND_URL || process.env.CHAT_BACKEND_URL || "http://localhost:8787";
}

export async function POST(event: APIEvent) {
  const backendUrl = getBackendUrl(event);
  const targetUrl = `${backendUrl}/api/chat`;

  try {
    const body = await event.request.json();

    const resp = await fetch(targetUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.text();
      return new Response(JSON.stringify({ error: err }), {
        status: resp.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(resp.body, {
      headers: { "Content-Type": "application/json" },
    });
  } catch (e: any) {
    return new Response(
      JSON.stringify({ error: e.message, target: targetUrl }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }
}
