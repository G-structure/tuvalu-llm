import type { APIEvent } from "@solidjs/start/server";
import { insertFeedback } from "~/lib/db";

export async function POST(event: APIEvent) {
  try {
    const body = await event.request.json();

    const validTypes = new Set(["thumbs_up", "thumbs_down"]);
    if (!body.article_id || typeof body.paragraph_idx !== "number" || !validTypes.has(body.feedback_type)) {
      return new Response(JSON.stringify({ error: "Invalid feedback" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    await insertFeedback({
      article_id: body.article_id,
      paragraph_idx: body.paragraph_idx,
      feedback_type: body.feedback_type,
      island: body.island || undefined,
      session_id: body.session_id || undefined,
    });

    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("Feedback API error:", e);
    return new Response(JSON.stringify({ error: "Server error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
