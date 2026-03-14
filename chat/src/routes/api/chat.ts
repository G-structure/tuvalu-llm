import type { APIEvent } from "@solidjs/start/server";

const BACKEND_URL = process.env.CHAT_BACKEND_URL || "http://localhost:8787";

export async function POST(event: APIEvent) {
  const body = await event.request.json();

  const resp = await fetch(`${BACKEND_URL}/api/chat`, {
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
}
