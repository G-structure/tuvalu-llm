import type { APIEvent } from "@solidjs/start/server";
import type { Message } from "./types";

export type ChatConsentState = "sync_training" | "local_only";

export interface ChatConversationPayload {
  id: string;
  session_id: string;
  title: string;
  messages: Message[];
  source?: string;
  language_mode?: string | null;
  island?: string | null;
  consent_state?: ChatConsentState;
  created_at?: string;
  updated_at?: string;
  metadata?: Record<string, unknown> | null;
}

export interface ChatConversationSummary {
  id: string;
  session_id: string;
  title: string;
  source: string;
  language_mode: string | null;
  island: string | null;
  consent_state: ChatConsentState;
  message_count: number;
  created_at: string;
  updated_at: string;
  synced_at: string | null;
  metadata_json: string | null;
}

export interface ChatConversationRecord extends ChatConversationSummary {
  messages: Message[];
}

let _devProxyReady: Promise<any> | null = null;
let _schemaReady: Promise<void> | null = null;

async function getDb(event?: APIEvent): Promise<D1Database> {
  const cfEnv = (event?.context as any)?.cloudflare?.env;
  const db =
    cfEnv?.DB ||
    (process.env as any).DB ||
    (globalThis as any).__env__?.DB;

  if (db) return db;

  if (!_devProxyReady) {
    _devProxyReady = (async () => {
      const { getPlatformProxy } = await import("wrangler");
      const proxy = await getPlatformProxy({
        persist: { path: ".wrangler/state/v3" },
      });
      (globalThis as any).__env__ = proxy.env;
      return proxy;
    })();
  }

  const proxy = await _devProxyReady;
  return proxy.env.DB;
}

async function ensureChatSchema(db: D1Database): Promise<void> {
  if (_schemaReady) {
    await _schemaReady;
    return;
  }

  _schemaReady = (async () => {
    await db
      .prepare(
        `CREATE TABLE IF NOT EXISTS chat_conversations (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          title TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'web',
          language_mode TEXT,
          island TEXT,
          consent_state TEXT NOT NULL DEFAULT 'sync_training',
          message_count INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now')),
          synced_at TEXT,
          metadata_json TEXT
        )`
      )
      .run();

    await db
      .prepare(
        `CREATE INDEX IF NOT EXISTS idx_chat_conversations_session_updated
         ON chat_conversations(session_id, updated_at DESC)`
      )
      .run();

    await db
      .prepare(
        `CREATE TABLE IF NOT EXISTS chat_messages (
          id TEXT PRIMARY KEY,
          conversation_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          sequence INTEGER NOT NULL,
          client_created_at TEXT,
          model_run TEXT,
          sampler_path TEXT,
          sampler_step TEXT,
          latency_ms INTEGER,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          metadata_json TEXT,
          UNIQUE(conversation_id, sequence)
        )`
      )
      .run();

    await db
      .prepare(
        `CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_sequence
         ON chat_messages(conversation_id, sequence ASC)`
      )
      .run();

    await db
      .prepare(
        `CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
         ON chat_messages(session_id, created_at DESC)`
      )
      .run();

    await db
      .prepare(
        `CREATE TABLE IF NOT EXISTS chat_feedback (
          id TEXT PRIMARY KEY,
          conversation_id TEXT NOT NULL,
          message_id TEXT,
          session_id TEXT NOT NULL,
          rating TEXT NOT NULL,
          correction_text TEXT,
          selected_text TEXT,
          island TEXT,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          metadata_json TEXT
        )`
      )
      .run();

    await db
      .prepare(
        `CREATE INDEX IF NOT EXISTS idx_chat_feedback_conversation
         ON chat_feedback(conversation_id, created_at DESC)`
      )
      .run();

    await db
      .prepare(
        `CREATE TABLE IF NOT EXISTS chat_training_examples (
          id TEXT PRIMARY KEY,
          conversation_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          task_family TEXT NOT NULL DEFAULT 'chat',
          language_mode TEXT,
          messages_json TEXT NOT NULL,
          metadata_json TEXT,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )`
      )
      .run();

    await db
      .prepare(
        `CREATE INDEX IF NOT EXISTS idx_chat_training_examples_session
         ON chat_training_examples(session_id, updated_at DESC)`
      )
      .run();
  })();

  await _schemaReady;
}

function cleanText(value: unknown, max: number): string {
  return typeof value === "string" ? value.trim().slice(0, max) : "";
}

function safeJson(value: unknown): string | null {
  if (!value) return null;
  try {
    return JSON.stringify(value);
  } catch {
    return null;
  }
}

function normalizeMessages(messages: Message[]): Message[] {
  return messages
    .filter(
      (message) =>
        message &&
        (message.role === "user" ||
          message.role === "assistant" ||
          message.role === "system") &&
        typeof message.content === "string" &&
        message.content.trim()
    )
    .slice(-80)
    .map((message) => ({
      id: cleanText(message.id, 120) || undefined,
      role: message.role,
      content: message.content.trim().slice(0, 4000),
      created_at: cleanText(message.created_at, 80) || undefined,
    }));
}

export function normalizeChatConversation(
  input: ChatConversationPayload
): ChatConversationPayload {
  const now = new Date().toISOString();
  const id = cleanText(input.id, 120);
  const sessionId = cleanText(input.session_id, 200);
  const title = cleanText(input.title, 120) || "Untitled chat";

  return {
    id,
    session_id: sessionId,
    title,
    messages: normalizeMessages(input.messages || []),
    source: cleanText(input.source, 40) || "web",
    language_mode: cleanText(input.language_mode, 40) || null,
    island: cleanText(input.island, 80) || null,
    consent_state:
      input.consent_state === "local_only" ? "local_only" : "sync_training",
    created_at: cleanText(input.created_at, 80) || now,
    updated_at: cleanText(input.updated_at, 80) || now,
    metadata: input.metadata || null,
  };
}

export function isPersistableConversation(
  input: ChatConversationPayload
): boolean {
  return !!input.id && !!input.session_id;
}

export async function upsertChatConversation(
  input: ChatConversationPayload,
  event?: APIEvent
): Promise<void> {
  const conversation = normalizeChatConversation(input);
  if (!isPersistableConversation(conversation)) {
    throw new Error("Invalid conversation payload");
  }

  const db = await getDb(event);
  await ensureChatSchema(db);

  const metadataJson = safeJson(conversation.metadata);
  const syncedAt = new Date().toISOString();

  await db
    .prepare(
      `INSERT INTO chat_conversations (
         id, session_id, title, source, language_mode, island, consent_state,
         message_count, created_at, updated_at, synced_at, metadata_json
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         session_id = excluded.session_id,
         title = excluded.title,
         source = excluded.source,
         language_mode = excluded.language_mode,
         island = excluded.island,
         consent_state = excluded.consent_state,
         message_count = excluded.message_count,
         updated_at = excluded.updated_at,
         synced_at = excluded.synced_at,
         metadata_json = excluded.metadata_json`
    )
    .bind(
      conversation.id,
      conversation.session_id,
      conversation.title,
      conversation.source,
      conversation.language_mode,
      conversation.island,
      conversation.consent_state,
      conversation.messages.length,
      conversation.created_at,
      conversation.updated_at,
      syncedAt,
      metadataJson
    )
    .run();

  await db
    .prepare(
      `DELETE FROM chat_messages
       WHERE conversation_id = ? AND session_id = ?`
    )
    .bind(conversation.id, conversation.session_id)
    .run();

  for (const [index, message] of conversation.messages.entries()) {
    await db
      .prepare(
        `INSERT INTO chat_messages (
           id, conversation_id, session_id, role, content, sequence,
           client_created_at, metadata_json
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(conversation_id, sequence) DO UPDATE SET
           id = excluded.id,
           role = excluded.role,
           content = excluded.content,
           client_created_at = excluded.client_created_at,
           metadata_json = excluded.metadata_json`
      )
      .bind(
        message.id || `${conversation.id}-${index}`,
        conversation.id,
        conversation.session_id,
        message.role,
        message.content,
        index,
        message.created_at || null,
        null
      )
      .run();
  }

  if (
    conversation.consent_state === "sync_training" &&
    conversation.messages.length >= 2
  ) {
    await db
      .prepare(
        `INSERT INTO chat_training_examples (
           id, conversation_id, session_id, task_family, language_mode,
           messages_json, metadata_json, updated_at
         ) VALUES (?, ?, ?, 'chat', ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           language_mode = excluded.language_mode,
           messages_json = excluded.messages_json,
           metadata_json = excluded.metadata_json,
           updated_at = excluded.updated_at`
      )
      .bind(
        `chat-${conversation.id}`,
        conversation.id,
        conversation.session_id,
        conversation.language_mode,
        JSON.stringify(conversation.messages),
        safeJson({
          title: conversation.title,
          island: conversation.island,
          source: conversation.source,
          consent_state: conversation.consent_state,
        }),
        conversation.updated_at
      )
      .run();
  }
}

export async function listChatConversations(
  sessionId: string,
  includeMessages: boolean,
  event?: APIEvent
): Promise<ChatConversationRecord[] | ChatConversationSummary[]> {
  const cleanSessionId = cleanText(sessionId, 200);
  if (!cleanSessionId) return [];

  const db = await getDb(event);
  await ensureChatSchema(db);

  const { results } = await db
    .prepare(
      `SELECT id, session_id, title, source, language_mode, island,
              consent_state, message_count, created_at, updated_at, synced_at,
              metadata_json
       FROM chat_conversations
       WHERE session_id = ?
       ORDER BY updated_at DESC
       LIMIT 40`
    )
    .bind(cleanSessionId)
    .all();

  const summaries = results as unknown as ChatConversationSummary[];
  if (!includeMessages) return summaries;

  const records: ChatConversationRecord[] = [];
  for (const summary of summaries) {
    const { results: rows } = await db
      .prepare(
        `SELECT id, role, content, client_created_at
         FROM chat_messages
         WHERE conversation_id = ? AND session_id = ?
         ORDER BY sequence ASC`
      )
      .bind(summary.id, cleanSessionId)
      .all();

    records.push({
      ...summary,
      messages: (rows as any[]).map((row) => ({
        id: row.id,
        role: row.role,
        content: row.content,
        created_at: row.client_created_at || undefined,
      })),
    });
  }

  return records;
}

export async function deleteChatConversation(
  id: string,
  sessionId: string,
  event?: APIEvent
): Promise<void> {
  const cleanId = cleanText(id, 120);
  const cleanSessionId = cleanText(sessionId, 200);
  if (!cleanId || !cleanSessionId) return;

  const db = await getDb(event);
  await ensureChatSchema(db);

  await db
    .prepare(`DELETE FROM chat_messages WHERE conversation_id = ? AND session_id = ?`)
    .bind(cleanId, cleanSessionId)
    .run();

  await db
    .prepare(`DELETE FROM chat_training_examples WHERE conversation_id = ? AND session_id = ?`)
    .bind(cleanId, cleanSessionId)
    .run();

  await db
    .prepare(`DELETE FROM chat_feedback WHERE conversation_id = ? AND session_id = ?`)
    .bind(cleanId, cleanSessionId)
    .run();

  await db
    .prepare(`DELETE FROM chat_conversations WHERE id = ? AND session_id = ?`)
    .bind(cleanId, cleanSessionId)
    .run();
}

export async function insertChatFeedback(
  feedback: {
    id: string;
    conversation_id: string;
    message_id?: string | null;
    session_id: string;
    rating:
      | "up"
      | "down"
      | "correction"
      | "good"
      | "needs_work"
      | "sounded_funny"
      | "fix_words";
    correction_text?: string | null;
    selected_text?: string | null;
    island?: string | null;
    metadata?: Record<string, unknown> | null;
  },
  event?: APIEvent
): Promise<void> {
  const db = await getDb(event);
  await ensureChatSchema(db);

  await db
    .prepare(
      `INSERT INTO chat_feedback (
         id, conversation_id, message_id, session_id, rating,
         correction_text, selected_text, island, metadata_json
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .bind(
      cleanText(feedback.id, 120),
      cleanText(feedback.conversation_id, 120),
      cleanText(feedback.message_id, 120) || null,
      cleanText(feedback.session_id, 200),
      feedback.rating,
      cleanText(feedback.correction_text, 1200) || null,
      cleanText(feedback.selected_text, 1200) || null,
      cleanText(feedback.island, 80) || null,
      safeJson(feedback.metadata)
    )
    .run();
}
