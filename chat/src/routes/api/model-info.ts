import type { APIEvent } from "@solidjs/start/server";

const BACKEND_URL = process.env.CHAT_BACKEND_URL || "http://localhost:8787";

export async function GET(_event: APIEvent) {
  try {
    const resp = await fetch(`${BACKEND_URL}/api/model-info`);
    return new Response(resp.body, {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return new Response(JSON.stringify({ error: "Backend unavailable" }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }
}
