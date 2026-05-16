import type { APIEvent } from "@solidjs/start/server";
import {
  normalizeChatConversation,
  upsertChatConversation,
} from "~/lib/chat-conversations";
import { getChatBackendUrl } from "~/lib/chat-backend";

const MAX_MESSAGE_LENGTH = 4000;
const MAX_MESSAGES = 50;
const MAX_BODY_BYTES = 64 * 1024; // 64 KB

function cleanText(value: unknown, max: number): string {
  return typeof value === "string" ? value.trim().slice(0, max) : "";
}

function validateChatBody(body: unknown):
  | {
      ok: true;
      payload: object;
      persistence: {
        conversation_id: string;
        session_id: string;
        title: string;
        consent_state: "sync_training" | "local_only";
        island: string | null;
        language_mode: string | null;
        source: string;
      };
    }
  | { ok: false; error: string } {
  if (!body || typeof body !== "object") return { ok: false, error: "Invalid request body" };
  const b = body as Record<string, unknown>;

  if (!Array.isArray(b.messages)) return { ok: false, error: "messages must be an array" };
  if (b.messages.length === 0) return { ok: false, error: "messages must not be empty" };
  if (b.messages.length > MAX_MESSAGES) return { ok: false, error: `Too many messages (max ${MAX_MESSAGES})` };

  const validRoles = new Set(["user", "assistant", "system"]);
  for (const msg of b.messages) {
    if (!msg || typeof msg !== "object") return { ok: false, error: "Invalid message" };
    if (!validRoles.has((msg as any).role)) return { ok: false, error: "Invalid message role" };
    if (typeof (msg as any).content !== "string") return { ok: false, error: "Message content must be a string" };
    if ((msg as any).content.length > MAX_MESSAGE_LENGTH) return { ok: false, error: `Message too long (max ${MAX_MESSAGE_LENGTH} chars)` };
  }

  if (b.temperature !== undefined && (typeof b.temperature !== "number" || b.temperature < 0 || b.temperature > 2)) {
    return { ok: false, error: "temperature must be 0-2" };
  }
  if (b.max_tokens !== undefined && (typeof b.max_tokens !== "number" || b.max_tokens < 1 || b.max_tokens > 4096)) {
    return { ok: false, error: "max_tokens must be 1-4096" };
  }

  return {
    ok: true,
    payload: {
      messages: b.messages.map((m: any) => ({ role: m.role, content: m.content })),
      temperature: b.temperature ?? 0.3,
      max_tokens: b.max_tokens ?? 1024,
    },
    persistence: {
      conversation_id: cleanText(b.conversation_id, 120),
      session_id: cleanText(b.session_id, 200),
      title: cleanText(b.title, 120) || "Untitled chat",
      consent_state:
        b.consent_state === "local_only" ? "local_only" : "sync_training",
      island: cleanText(b.island, 80) || null,
      language_mode: cleanText(b.language_mode, 40) || null,
      source: cleanText(b.source, 40) || "web",
    },
  };
}

export async function POST(event: APIEvent) {
  const backendUrl = getChatBackendUrl(event);
  const targetUrl = `${backendUrl}/api/chat`;

  try {
    const contentLength = event.request.headers.get("content-length");
    if (contentLength && parseInt(contentLength, 10) > MAX_BODY_BYTES) {
      return new Response(JSON.stringify({ error: "Request too large" }), {
        status: 413,
        headers: { "Content-Type": "application/json" },
      });
    }

    const body = await event.request.json();
    const validation = validateChatBody(body);
    if (!validation.ok) {
      return new Response(JSON.stringify({ error: validation.error }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const resp = await fetch(targetUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(validation.payload),
    });

    if (!resp.ok) {
      const status = resp.status >= 500 ? 502 : resp.status;
      return new Response(JSON.stringify({ error: "Chat request failed" }), {
        status,
        headers: { "Content-Type": "application/json" },
      });
    }

    const data = await resp.json().catch(() => null);
    const content =
      typeof data?.content === "string" ? data.content : "";

    if (!content) {
      return new Response(JSON.stringify({ error: "Invalid chat response" }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      });
    }

    const persist = validation.persistence;
    if (
      persist.conversation_id &&
      persist.session_id &&
      persist.consent_state === "sync_training"
    ) {
      const requestMessages = (body as any).messages || [];
      const assistantMessage = {
        id: cleanText((body as any).assistant_message_id, 120) || undefined,
        role: "assistant" as const,
        content,
        created_at: new Date().toISOString(),
      };

      upsertChatConversation(
        normalizeChatConversation({
          id: persist.conversation_id,
          session_id: persist.session_id,
          title: persist.title,
          messages: [...requestMessages, assistantMessage],
          source: persist.source,
          language_mode: persist.language_mode,
          island: persist.island,
          consent_state: persist.consent_state,
          updated_at: new Date().toISOString(),
          metadata: {
            model_info: data?.model_info || null,
            saved_from: "chat_proxy",
          },
        }),
        event
      ).catch((error) => {
        console.error("Chat persistence failed:", error);
      });
    }

    return new Response(JSON.stringify(data), {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return new Response(
      JSON.stringify({ error: "Chat service unavailable" }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }
}
