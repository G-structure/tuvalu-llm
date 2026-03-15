import type { APIEvent } from "@solidjs/start/server";
import { insertArticleFeedbackForm } from "~/lib/db";

const VALID_MODES = new Set(["tv", "tv+en", "en"]);

export async function POST(event: APIEvent) {
  try {
    const body = await event.request.json();
    const helpful = body.helpful_score;
    const correctionText =
      typeof body.correction_text === "string" ? body.correction_text.trim() : "";
    const correctionIdx = body.correction_paragraph_idx;

    if (
      !body.article_id ||
      (helpful !== 0 && helpful !== 1) ||
      !VALID_MODES.has(body.mode_preference)
    ) {
      return new Response(JSON.stringify({ error: "Invalid feedback" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    if (
      correctionText &&
      (typeof correctionIdx !== "number" || Number.isNaN(correctionIdx) || correctionIdx < 0)
    ) {
      return new Response(JSON.stringify({ error: "Select a paragraph for corrections" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    if (correctionText.length > 1000) {
      return new Response(JSON.stringify({ error: "Correction is too long" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    await insertArticleFeedbackForm({
      article_id: body.article_id,
      helpful_score: helpful,
      mode_preference: body.mode_preference,
      correction_paragraph_idx: correctionText ? correctionIdx : undefined,
      correction_text: correctionText || undefined,
      island: body.island || undefined,
      session_id: body.session_id || undefined,
    });

    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("Article feedback API error:", e);
    return new Response(JSON.stringify({ error: "Server error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
