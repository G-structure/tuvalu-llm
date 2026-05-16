import type { APIEvent } from "@solidjs/start/server";
import { insertChatFeedback } from "~/lib/chat-conversations";

const VALID_RATINGS = new Set([
  "up",
  "down",
  "correction",
  "good",
  "needs_work",
  "sounded_funny",
  "fix_words",
]);

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function asString(value: unknown, max: number): string {
  return typeof value === "string" ? value.trim().slice(0, max) : "";
}

export async function POST(event: APIEvent) {
  try {
    const body = await event.request.json();
    const rating = asString((body as any)?.rating, 24);

    if (
      !asString((body as any)?.id, 120) ||
      !asString((body as any)?.conversation_id, 120) ||
      !asString((body as any)?.session_id, 200) ||
      !VALID_RATINGS.has(rating)
    ) {
      return json({ error: "Invalid feedback" }, 400);
    }

    await insertChatFeedback(
      {
        id: asString((body as any).id, 120),
        conversation_id: asString((body as any).conversation_id, 120),
        message_id: asString((body as any).message_id, 120) || null,
        session_id: asString((body as any).session_id, 200),
        rating: rating as
          | "up"
          | "down"
          | "correction"
          | "good"
          | "needs_work"
          | "sounded_funny"
          | "fix_words",
        correction_text: asString((body as any).correction_text, 1200) || null,
        selected_text: asString((body as any).selected_text, 1200) || null,
        island: asString((body as any).island, 80) || null,
        metadata: (body as any).metadata || null,
      },
      event
    );

    return json({ ok: true });
  } catch (e) {
    console.error("Chat feedback API error:", e);
    return json({ error: "Server error" }, 500);
  }
}
