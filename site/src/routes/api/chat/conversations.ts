import type { APIEvent } from "@solidjs/start/server";
import {
  deleteChatConversation,
  listChatConversations,
  normalizeChatConversation,
  upsertChatConversation,
} from "~/lib/chat-conversations";

const MAX_BODY_BYTES = 96 * 1024;

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function asString(value: unknown, max: number): string {
  return typeof value === "string" ? value.trim().slice(0, max) : "";
}

export async function GET(event: APIEvent) {
  try {
    const url = new URL(event.request.url);
    const sessionId = asString(url.searchParams.get("session_id"), 200);
    const includeMessages = url.searchParams.get("include") === "messages";

    if (!sessionId) {
      return json({ error: "Missing session_id" }, 400);
    }

    const conversations = await listChatConversations(
      sessionId,
      includeMessages,
      event
    );

    return json({ conversations });
  } catch (e) {
    console.error("Chat conversations GET error:", e);
    return json({ error: "Server error" }, 500);
  }
}

export async function POST(event: APIEvent) {
  try {
    const contentLength = event.request.headers.get("content-length");
    if (contentLength && parseInt(contentLength, 10) > MAX_BODY_BYTES) {
      return json({ error: "Request too large" }, 413);
    }

    const body = await event.request.json();
    const raw = (body as any)?.conversation || body;
    const conversation = normalizeChatConversation(raw);

    if (!conversation.id || !conversation.session_id) {
      return json({ error: "Invalid conversation" }, 400);
    }

    await upsertChatConversation(conversation, event);
    return json({ ok: true, synced_at: new Date().toISOString() });
  } catch (e) {
    console.error("Chat conversations POST error:", e);
    return json({ error: "Server error" }, 500);
  }
}

export async function DELETE(event: APIEvent) {
  try {
    const url = new URL(event.request.url);
    const id = asString(url.searchParams.get("id"), 120);
    const sessionId = asString(url.searchParams.get("session_id"), 200);

    if (!id || !sessionId) {
      return json({ error: "Missing id or session_id" }, 400);
    }

    await deleteChatConversation(id, sessionId, event);
    return json({ ok: true });
  } catch (e) {
    console.error("Chat conversations DELETE error:", e);
    return json({ error: "Server error" }, 500);
  }
}
