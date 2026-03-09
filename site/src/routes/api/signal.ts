import type { APIEvent } from "@solidjs/start/server";
import { insertSignal } from "~/lib/db";

const VALID_TYPES = new Set(["share", "reveal", "flag"]);

export async function POST(event: APIEvent) {
  try {
    const body = await event.request.json();

    if (!body.article_id || !VALID_TYPES.has(body.signal_type)) {
      return new Response(JSON.stringify({ error: "Invalid signal" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    await insertSignal({
      article_id: body.article_id,
      signal_type: body.signal_type,
      paragraph_index: body.paragraph_index,
      session_id: body.session_id || undefined,
      island: body.island || undefined,
    });

    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return new Response(JSON.stringify({ error: "Server error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
