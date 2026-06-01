import type { APIEvent } from "@solidjs/start/server";

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_OPENROUTER_MODEL = "openai/gpt-5-nano";
const DEFAULT_TRANSLATION_BACKEND_URL = "https://api.cyberneticphysics.com/tvl-chat";
const DEFAULT_TINKER_API_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1";
const DEFAULT_TINKER_MODEL_NAME = "Qwen/Qwen3-30B-A3B";
const DEFAULT_TINKER_MODEL_PATH =
  "tinker://06e2f0d3-7d06-5c29-83a4-f44c0d29728c:train:0/sampler_weights/gen_eval_018000";
const MAX_BODY_BYTES = 96 * 1024;
const MAX_MESSAGE_LENGTH = 8000;
const MAX_HISTORY_MESSAGES = 30;
const MAX_TRANSLATION_TEXT_LENGTH = 12000;
const MAX_ERROR_TEXT_BYTES = 2048;
const OPENROUTER_TIMEOUT_MS = 45_000;
const TRANSLATION_TIMEOUT_MS = 60_000;
const MIN_GPT5_COMPLETION_TOKENS = 1200;

type Language = "tvl" | "en" | "mixed" | "unknown";
type TargetLanguage = "tvl" | "en" | "bilingual" | "same_as_user";
type DisplayLanguage = "tvl" | "en" | "bilingual";
type RequestedDisplayLanguage = DisplayLanguage | "auto";
type Intent =
  | "general_chat"
  | "translate"
  | "generate_in_language"
  | "explain_translation"
  | "rewrite"
  | "summarize"
  | "code"
  | "math";

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

interface ValidatedBody {
  sessionId: string;
  userMessage: string;
  providedHistory: ChatMessage[];
  requestedDisplayLanguage: RequestedDisplayLanguage;
  temperature: number;
  maxCompletionTokens: number;
}

interface RouteDecision {
  input_language: Language;
  intent: Intent;
  source_language: Language;
  target_language: TargetLanguage;
  needs_tvl_to_en: boolean;
  needs_en_to_tvl: boolean;
  needs_nano_answer: boolean;
  display_language: DisplayLanguage;
  translation_text: string;
  preserve_blocks: string[];
  reason: string;
}

interface StoredMessage {
  role: "user" | "assistant";
  content_en: string | null;
  content_tvl: string | null;
  content_display: string;
}

interface ModelsUsed {
  router: string;
  answer?: string;
  tvl_to_en?: string;
  en_to_tvl?: string;
}

let _devProxy: any = null;
let _schemaReady: Promise<void> | null = null;

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

async function readLimitedText(resp: Response, maxBytes: number): Promise<string> {
  const reader = resp.body?.getReader();
  if (!reader) return "";

  const chunks: Uint8Array[] = [];
  let total = 0;

  try {
    while (total < maxBytes) {
      const { value, done } = await reader.read();
      if (done || !value) break;
      const remaining = maxBytes - total;
      chunks.push(value.byteLength > remaining ? value.slice(0, remaining) : value);
      total += Math.min(value.byteLength, remaining);
    }
  } finally {
    reader.cancel().catch(() => {});
  }

  return new TextDecoder().decode(concatBytes(chunks));
}

function concatBytes(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const output = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

async function readJsonRequest(request: Request): Promise<{ ok: true; body: unknown } | { ok: false; status: number; error: string }> {
  const requestText = await readLimitedRequestText(request, MAX_BODY_BYTES);
  if (!requestText.ok) return { ok: false, status: 413, error: "Request too large" };

  try {
    return { ok: true, body: JSON.parse(requestText.text) };
  } catch {
    return { ok: false, status: 400, error: "Invalid JSON" };
  }
}

async function readLimitedRequestText(
  request: Request,
  maxBytes: number
): Promise<{ ok: true; text: string } | { ok: false }> {
  const reader = request.body?.getReader();
  if (!reader) return { ok: true, text: "" };

  const chunks: Uint8Array[] = [];
  let total = 0;

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done || !value) break;
      if (total + value.byteLength > maxBytes) return { ok: false };
      chunks.push(value);
      total += value.byteLength;
    }
  } finally {
    reader.cancel().catch(() => {});
  }

  return { ok: true, text: new TextDecoder().decode(concatBytes(chunks)) };
}

function getCfEnv(event: APIEvent): Record<string, any> {
  return ((event.context as any)?.cloudflare?.env ?? {}) as Record<string, any>;
}

function getEnvValue(event: APIEvent, key: string): string | undefined {
  const cfValue = getCfEnv(event)[key];
  if (typeof cfValue === "string" && cfValue) return cfValue;
  const processValue = (process.env as any)?.[key];
  if (typeof processValue === "string" && processValue) return processValue;
  const globalValue = (globalThis as any).__env__?.[key];
  if (typeof globalValue === "string" && globalValue) return globalValue;
  return undefined;
}

function getBackendUrl(event: APIEvent): string {
  return (getEnvValue(event, "CHAT_BACKEND_URL") || DEFAULT_TRANSLATION_BACKEND_URL).replace(/\/+$/, "");
}

function getTinkerApiBaseUrl(event: APIEvent): string {
  return (getEnvValue(event, "TINKER_API_BASE_URL") || DEFAULT_TINKER_API_BASE_URL).replace(/\/+$/, "");
}

function getTinkerModelName(event: APIEvent): string {
  return getEnvValue(event, "TINKER_MODEL_NAME") || DEFAULT_TINKER_MODEL_NAME;
}

function getTinkerModelPath(event: APIEvent): string {
  return (
    getEnvValue(event, "TINKER_MODEL_PATH") ||
    getEnvValue(event, "SAMPLER_PATH") ||
    DEFAULT_TINKER_MODEL_PATH
  );
}

function getRouterModel(event: APIEvent): string {
  return (
    getEnvValue(event, "OPENROUTER_ROUTER_MODEL") ||
    getEnvValue(event, "OPENROUTER_MODEL") ||
    DEFAULT_OPENROUTER_MODEL
  );
}

function getAnswerModel(event: APIEvent): string {
  return (
    getEnvValue(event, "OPENROUTER_CHAT_MODEL") ||
    getEnvValue(event, "OPENROUTER_ANSWER_MODEL") ||
    getEnvValue(event, "OPENROUTER_MODEL") ||
    DEFAULT_OPENROUTER_MODEL
  );
}

async function getDb(event: APIEvent): Promise<D1Database | null> {
  const cfEnv = getCfEnv(event);
  const cfDb = cfEnv.DB;
  if (cfDb) return cfDb as D1Database;
  if (cfEnv.CF_PAGES) return null;

  const processDb = (process.env as any)?.DB;
  if (processDb) return processDb as D1Database;

  const globalDb = (globalThis as any).__env__?.DB;
  if (globalDb) return globalDb as D1Database;

  try {
    const { getPlatformProxy } = await import("wrangler");
    _devProxy ||= await getPlatformProxy({ persist: { path: ".wrangler/state/v3" } });
    (globalThis as any).__env__ = _devProxy.env;
    return _devProxy.env.DB as D1Database;
  } catch {
    return null;
  }
}

async function ensureSchema(db: D1Database): Promise<void> {
  if (_schemaReady) {
    await _schemaReady;
    return;
  }

  _schemaReady = (async () => {
    await db
      .prepare(
        `CREATE TABLE IF NOT EXISTS chat_router_sessions (
          id TEXT PRIMARY KEY,
          display_language TEXT NOT NULL DEFAULT 'auto',
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )`
      )
      .run();

    await db
      .prepare(
        `CREATE TABLE IF NOT EXISTS chat_router_messages (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          role TEXT NOT NULL,
          original_language TEXT NOT NULL,
          display_language TEXT NOT NULL,
          content_original TEXT NOT NULL,
          content_en TEXT,
          content_tvl TEXT,
          content_display TEXT NOT NULL,
          route_intent TEXT,
          route_json TEXT,
          models_used_json TEXT,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          FOREIGN KEY (session_id) REFERENCES chat_router_sessions(id)
        )`
      )
      .run();

    await db
      .prepare(
        `CREATE INDEX IF NOT EXISTS idx_chat_router_messages_session_created
         ON chat_router_messages(session_id, created_at)`
      )
      .run();
  })().catch((e) => {
    _schemaReady = null;
    throw e;
  });

  await _schemaReady;
}

async function loadHistory(db: D1Database | null, sessionId: string): Promise<StoredMessage[]> {
  if (!db) return [];
  try {
    await ensureSchema(db);
    const { results } = await db
      .prepare(
        `SELECT role, content_en, content_tvl, content_display
         FROM chat_router_messages
         WHERE session_id = ?
         ORDER BY created_at DESC, rowid DESC
         LIMIT ?`
      )
      .bind(sessionId, MAX_HISTORY_MESSAGES)
      .all();

    return (results as unknown as StoredMessage[]).reverse();
  } catch (e) {
    console.error("Chat router history load failed:", e);
    return [];
  }
}

async function saveTurn(params: {
  db: D1Database | null;
  sessionId: string;
  requestedDisplayLanguage: RequestedDisplayLanguage;
  route: RouteDecision;
  modelsUsed: ModelsUsed;
  user: {
    original: string;
    contentEn: string | null;
    contentTvl: string | null;
    display: string;
  };
  assistant: {
    original: string;
    contentEn: string | null;
    contentTvl: string | null;
    display: string;
  };
}): Promise<void> {
  if (!params.db) return;

  try {
    await ensureSchema(params.db);
    await params.db
      .prepare(
        `INSERT INTO chat_router_sessions (id, display_language, updated_at)
         VALUES (?, ?, datetime('now'))
         ON CONFLICT(id) DO UPDATE SET
           display_language = excluded.display_language,
           updated_at = excluded.updated_at`
      )
      .bind(params.sessionId, params.requestedDisplayLanguage)
      .run();

    const routeJson = JSON.stringify(params.route);
    const modelsJson = JSON.stringify(params.modelsUsed);
    const nowPrefix = `${Date.now()}-${crypto.randomUUID()}`;

    await params.db
      .prepare(
        `INSERT INTO chat_router_messages (
          id, session_id, role, original_language, display_language,
          content_original, content_en, content_tvl, content_display,
          route_intent, route_json, models_used_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .bind(
        `${nowPrefix}-user`,
        params.sessionId,
        "user",
        params.route.input_language,
        params.requestedDisplayLanguage,
        params.user.original,
        params.user.contentEn,
        params.user.contentTvl,
        params.user.display,
        params.route.intent,
        routeJson,
        modelsJson
      )
      .run();

    await params.db
      .prepare(
        `INSERT INTO chat_router_messages (
          id, session_id, role, original_language, display_language,
          content_original, content_en, content_tvl, content_display,
          route_intent, route_json, models_used_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .bind(
        `${nowPrefix}-assistant`,
        params.sessionId,
        "assistant",
        params.route.display_language,
        params.requestedDisplayLanguage,
        params.assistant.original,
        params.assistant.contentEn,
        params.assistant.contentTvl,
        params.assistant.display,
        params.route.intent,
        routeJson,
        modelsJson
      )
      .run();
  } catch (e) {
    console.error("Chat router persistence failed:", e);
  }
}

function validateChatBody(body: unknown): { ok: true; payload: ValidatedBody } | { ok: false; error: string } {
  if (!body || typeof body !== "object") return { ok: false, error: "Invalid request body" };
  const b = body as Record<string, unknown>;

  let userMessage = "";
  let providedHistory: ChatMessage[] = [];

  if (typeof b.message === "string") {
    userMessage = b.message.trim();
  } else if (Array.isArray(b.messages)) {
    if (b.messages.length === 0) return { ok: false, error: "messages must not be empty" };
    if (b.messages.length > MAX_HISTORY_MESSAGES + 1) {
      return { ok: false, error: `Too many messages (max ${MAX_HISTORY_MESSAGES + 1})` };
    }

    const validRoles = new Set(["user", "assistant", "system"]);
    const messages: ChatMessage[] = [];
    for (const msg of b.messages) {
      if (!msg || typeof msg !== "object") return { ok: false, error: "Invalid message" };
      const role = (msg as any).role;
      const content = (msg as any).content;
      if (!validRoles.has(role)) return { ok: false, error: "Invalid message role" };
      if (typeof content !== "string") return { ok: false, error: "Message content must be a string" };
      if (content.length > MAX_MESSAGE_LENGTH) {
        return { ok: false, error: `Message too long (max ${MAX_MESSAGE_LENGTH} chars)` };
      }
      messages.push({ role, content });
    }

    const latest = messages[messages.length - 1];
    if (!latest || latest.role !== "user") {
      return { ok: false, error: "Last message must be a user message" };
    }
    userMessage = latest.content.trim();
    providedHistory = messages.slice(0, -1);
  } else {
    return { ok: false, error: "message or messages is required" };
  }

  if (!userMessage) return { ok: false, error: "message must not be empty" };
  if (userMessage.length > MAX_MESSAGE_LENGTH) {
    return { ok: false, error: `Message too long (max ${MAX_MESSAGE_LENGTH} chars)` };
  }

  const rawSessionId = typeof b.session_id === "string" ? b.session_id.trim() : "";
  if (rawSessionId && rawSessionId.length > 160) return { ok: false, error: "session_id is too long" };

  const requestedDisplayLanguage =
    b.display_language === "tvl" ||
    b.display_language === "en" ||
    b.display_language === "bilingual" ||
    b.display_language === "auto"
      ? b.display_language
      : "auto";

  const temperature = typeof b.temperature === "number" ? b.temperature : 0.7;
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
    return { ok: false, error: "temperature must be 0-2" };
  }

  const rawMaxTokens = b.max_completion_tokens ?? b.max_tokens;
  const maxCompletionTokens = typeof rawMaxTokens === "number" ? rawMaxTokens : 1024;
  if (!Number.isFinite(maxCompletionTokens) || maxCompletionTokens < 1 || maxCompletionTokens > 4096) {
    return { ok: false, error: "max_completion_tokens must be 1-4096" };
  }

  return {
    ok: true,
    payload: {
      sessionId: rawSessionId || crypto.randomUUID(),
      userMessage,
      providedHistory,
      requestedDisplayLanguage,
      temperature,
      maxCompletionTokens,
    },
  };
}

function routeResponseFormat() {
  return {
    type: "json_schema",
    json_schema: {
      name: "tvl_chat_route",
      strict: true,
      schema: {
        type: "object",
        additionalProperties: false,
        required: [
          "input_language",
          "intent",
          "source_language",
          "target_language",
          "needs_tvl_to_en",
          "needs_en_to_tvl",
          "needs_nano_answer",
          "display_language",
          "translation_text",
          "preserve_blocks",
          "reason",
        ],
        properties: {
          input_language: { type: "string", enum: ["tvl", "en", "mixed", "unknown"] },
          intent: {
            type: "string",
            enum: [
              "general_chat",
              "translate",
              "generate_in_language",
              "explain_translation",
              "rewrite",
              "summarize",
              "code",
              "math",
            ],
          },
          source_language: { type: "string", enum: ["tvl", "en", "mixed", "unknown"] },
          target_language: { type: "string", enum: ["tvl", "en", "bilingual", "same_as_user"] },
          needs_tvl_to_en: { type: "boolean" },
          needs_en_to_tvl: { type: "boolean" },
          needs_nano_answer: { type: "boolean" },
          display_language: { type: "string", enum: ["tvl", "en", "bilingual"] },
          translation_text: { type: "string" },
          preserve_blocks: {
            type: "array",
            items: { type: "string" },
          },
          reason: { type: "string" },
        },
      },
    },
  };
}

async function callOpenRouter(event: APIEvent, payload: Record<string, unknown>): Promise<string> {
  const apiKey = getEnvValue(event, "OPENROUTER_API_KEY");
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is not configured");

  const referer =
    getEnvValue(event, "OPENROUTER_REFERER") ||
    getEnvValue(event, "SITE_URL") ||
    "https://tvl-chat.pages.dev";
  const title = getEnvValue(event, "OPENROUTER_TITLE") || "TVL Chat";
  const requestPayload = normalizeOpenRouterPayload(payload);

  const resp = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": referer,
      "X-OpenRouter-Title": title,
    },
    body: JSON.stringify(requestPayload),
    signal: AbortSignal.timeout(OPENROUTER_TIMEOUT_MS),
  });

  if (!resp.ok) {
    const errorText = await readLimitedText(resp, MAX_ERROR_TEXT_BYTES).catch(() => "");
    throw new Error(`OpenRouter request failed: ${resp.status} ${errorText.slice(0, 500)}`);
  }

  const data = await resp.json();
  const content = data?.choices?.[0]?.message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part?.text === "string" ? part.text : ""))
      .join("")
      .trim();
  }
  throw new Error("OpenRouter response did not include message content");
}

function normalizeOpenRouterPayload(payload: Record<string, unknown>): Record<string, unknown> {
  const model = typeof payload.model === "string" ? payload.model : "";
  if (!model.startsWith("openai/gpt-5")) return payload;

  const maxCompletionTokens =
    typeof payload.max_completion_tokens === "number"
      ? Math.max(payload.max_completion_tokens, MIN_GPT5_COMPLETION_TOKENS)
      : MIN_GPT5_COMPLETION_TOKENS;

  return {
    ...payload,
    max_completion_tokens: maxCompletionTokens,
    reasoning:
      payload.reasoning && typeof payload.reasoning === "object"
        ? payload.reasoning
        : { effort: "minimal", exclude: true },
  };
}

async function routeTurn(
  event: APIEvent,
  userMessage: string,
  requestedDisplayLanguage: RequestedDisplayLanguage
): Promise<RouteDecision> {
  const model = getRouterModel(event);
  const content = await callOpenRouter(event, {
    model,
    messages: [
      {
        role: "system",
        content:
          "You are the routing layer for a bilingual Tuvaluan-English chat product. " +
          "Classify the latest user turn and decide which models are needed. " +
          "The specialized TVL model should be used only for Tuvaluan-English translation. " +
          "The OpenRouter Nano model should answer all non-translation tasks in English. " +
          "For direct translation requests, set needs_nano_answer=false and put only the exact source text to translate in translation_text. " +
          "For normal Tuvaluan chat, set needs_tvl_to_en=true, needs_nano_answer=true, needs_en_to_tvl=true, and display_language=tvl. " +
          "For English chat, set needs_nano_answer=true and display_language=en unless the user requests Tuvaluan or bilingual output. " +
          "Preserve code blocks, URLs, names, numbers, and quoted text.",
      },
      {
        role: "user",
        content: JSON.stringify({
          user_message: userMessage,
          requested_display_language: requestedDisplayLanguage,
        }),
      },
    ],
    temperature: 0,
    max_completion_tokens: 700,
    response_format: routeResponseFormat(),
  });

  const parsed = JSON.parse(content) as RouteDecision;
  return normalizeRoute(parsed, userMessage, requestedDisplayLanguage);
}

function normalizeRoute(
  route: RouteDecision,
  userMessage: string,
  requestedDisplayLanguage: RequestedDisplayLanguage
): RouteDecision {
  const englishInstructionForTuvaluan = isEnglishInstructionForTuvaluan(userMessage);
  const normalizedRoute: RouteDecision =
    englishInstructionForTuvaluan && route.intent === "generate_in_language"
      ? {
          ...route,
          input_language: "en",
          source_language: "en",
          target_language: "tvl",
          needs_tvl_to_en: false,
          needs_en_to_tvl: false,
          needs_nano_answer: true,
          display_language: "tvl",
          translation_text: userMessage,
          reason: `${route.reason} English instruction requesting Tuvaluan output; answer directly in Tuvaluan.`,
        }
      : route;
  const displayLanguage =
    requestedDisplayLanguage === "auto" ||
    (normalizedRoute.intent === "translate" &&
      normalizedRoute.target_language !== "same_as_user" &&
      requestedDisplayLanguage !== "bilingual")
      ? normalizedRoute.display_language
      : requestedDisplayLanguage;
  const translationText = normalizedRoute.translation_text.trim() || userMessage;

  return {
    ...normalizedRoute,
    display_language: displayLanguage,
    translation_text:
      translationText.length > MAX_TRANSLATION_TEXT_LENGTH
        ? translationText.slice(0, MAX_TRANSLATION_TEXT_LENGTH)
        : translationText,
    needs_en_to_tvl:
      normalizedRoute.needs_en_to_tvl ||
      (!shouldAnswerDirectlyInTuvaluan({ ...normalizedRoute, display_language: displayLanguage }) &&
        (displayLanguage === "tvl" || displayLanguage === "bilingual")),
  };
}

function translationMessages(direction: "tvl_to_en" | "en_to_tvl", text: string): ChatMessage[] {
  return direction === "tvl_to_en"
    ? [
        {
          role: "system",
          content:
            "You are a careful Tuvaluan-to-English translator. Translate faithfully. " +
            "Preserve code blocks, URLs, names, numbers, and formatting. Output only the English translation.",
        },
        {
          role: "user",
          content: text,
        },
      ]
    : [
        {
          role: "system",
          content:
            "You are a careful English-to-Tuvaluan translator. Translate faithfully. " +
            "Preserve code blocks, URLs, names, numbers, and formatting. Output only the Tuvaluan translation.",
        },
        {
          role: "user",
          content: text,
        },
      ];
}

function parseTinkerCompletion(data: any): string {
  const choice = data?.choices?.[0];
  if (typeof choice?.text === "string") return choice.text;
  if (typeof choice?.message?.content === "string") return choice.message.content;
  if (Array.isArray(choice?.message?.content)) {
    return choice.message.content
      .map((part: any) => (typeof part?.text === "string" ? part.text : ""))
      .join("");
  }
  throw new Error("Tinker response did not include completion content");
}

async function translateWithTinkerHttp(
  event: APIEvent,
  direction: "tvl_to_en" | "en_to_tvl",
  text: string
): Promise<{ content: string; modelInfo?: unknown }> {
  const apiKey = getEnvValue(event, "TINKER_API_KEY");
  if (!apiKey) throw new Error("TINKER_API_KEY is not configured");

  const modelPath = getTinkerModelPath(event);
  const messages = translationMessages(direction, text);
  const resp = await fetch(`${getTinkerApiBaseUrl(event)}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "X-Api-Key": apiKey,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: modelPath,
      messages,
      temperature: 0.2,
      max_tokens: 2048,
    }),
    signal: AbortSignal.timeout(TRANSLATION_TIMEOUT_MS),
  });

  if (!resp.ok) {
    const errorText = await readLimitedText(resp, MAX_ERROR_TEXT_BYTES).catch(() => "");
    throw new Error(`Tinker translation request failed: ${resp.status} ${errorText.slice(0, 500)}`);
  }

  const data = await resp.json();
  const content = parseTinkerCompletion(data).replace(/<\|im_end\|>$/g, "").trim();
  if (!content) throw new Error("Tinker response was empty");

  return {
    content,
    modelInfo: {
      model_name: getTinkerModelName(event),
      sampler_path: modelPath,
      transport: "tinker-http",
    },
  };
}

async function translateWithTvlBackend(
  event: APIEvent,
  direction: "tvl_to_en" | "en_to_tvl",
  text: string
): Promise<{ content: string; modelInfo?: unknown }> {
  const backendUrl = getBackendUrl(event);
  const messages = translationMessages(direction, text);

  const resp = await fetch(`${backendUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      temperature: 0.2,
      max_tokens: 2048,
    }),
    signal: AbortSignal.timeout(TRANSLATION_TIMEOUT_MS),
  });

  if (!resp.ok) {
    const errorText = await readLimitedText(resp, MAX_ERROR_TEXT_BYTES).catch(() => "");
    throw new Error(`TVL translation request failed: ${resp.status} ${errorText.slice(0, 500)}`);
  }

  const data = await resp.json();
  if (typeof data?.content !== "string") {
    throw new Error("TVL translation response did not include content");
  }

  return { content: data.content.trim(), modelInfo: data.model_info };
}

async function translateWithTvlModel(
  event: APIEvent,
  direction: "tvl_to_en" | "en_to_tvl",
  text: string
): Promise<{ content: string; modelInfo?: unknown }> {
  if (getEnvValue(event, "TINKER_API_KEY")) {
    return translateWithTinkerHttp(event, direction, text);
  }
  return translateWithTvlBackend(event, direction, text);
}

function buildEnglishHistory(dbHistory: StoredMessage[], providedHistory: ChatMessage[]): ChatMessage[] {
  if (dbHistory.length > 0) {
    return dbHistory
      .filter((msg) => msg.role === "user" || msg.role === "assistant")
      .map((msg) => ({
        role: msg.role,
        content: msg.content_en || msg.content_display,
      }));
  }

  return providedHistory
    .filter((msg) => msg.role === "user" || msg.role === "assistant")
    .slice(-MAX_HISTORY_MESSAGES);
}

async function answerWithNano(params: {
  event: APIEvent;
  history: ChatMessage[];
  userMessageEn: string;
  outputLanguage?: DisplayLanguage;
  temperature: number;
  maxCompletionTokens: number;
}): Promise<string> {
  const model = getAnswerModel(params.event);
  const outputInstruction =
    params.outputLanguage === "tvl"
      ? "Answer directly in Tuvaluan. Do not add an English explanation unless the user asks for one."
      : params.outputLanguage === "bilingual"
        ? "Answer bilingually with Tuvaluan first, then English."
        : "Answer in clear English. Another model may translate your answer into Tuvaluan.";
  const content = await callOpenRouter(params.event, {
    model,
    messages: [
      {
        role: "system",
        content:
          "You are the general reasoning model for TVL Chat. " +
          outputInstruction +
          " " +
          "When code, URLs, identifiers, names, markdown tables, formulas, or exact quoted text appear, preserve them carefully.",
      },
      ...params.history,
      { role: "user", content: params.userMessageEn },
    ],
    temperature: params.temperature,
    max_completion_tokens: params.maxCompletionTokens,
  });

  return content.trim();
}

function isEnglishInstructionForTuvaluan(text: string): boolean {
  const normalized = text.toLowerCase();
  if (!/\btuvaluan\b|\bte\s+gana\s+tuvalu\b|\btuvalu\b/.test(normalized)) return false;
  if (
    !/\b(in|to|into|with|using)\s+(tuvaluan|te\s+gana\s+tuvalu|tuvalu)\b/.test(normalized) &&
    !/\b(greet|write|say|compose|draft|create|make|translate|explain)\b/.test(normalized)
  ) {
    return false;
  }

  const latinLetters = text.match(/[A-Za-z]/g)?.length ?? 0;
  const nonAsciiLetters = text.match(/[^\x00-\x7F]/g)?.length ?? 0;
  return latinLetters > 0 && nonAsciiLetters / Math.max(latinLetters, 1) < 0.2;
}

function shouldAnswerDirectlyInTuvaluan(route: RouteDecision): boolean {
  return (
    route.input_language === "en" &&
    route.intent === "generate_in_language" &&
    route.target_language === "tvl" &&
    route.display_language === "tvl"
  );
}

function formatDisplay(displayLanguage: DisplayLanguage, contentEn: string | null, contentTvl: string | null): string {
  if (displayLanguage === "bilingual" && contentTvl && contentEn) {
    return `Tuvaluan:\n${contentTvl}\n\nEnglish:\n${contentEn}`;
  }
  if (displayLanguage === "tvl" && contentTvl) return contentTvl;
  return contentEn || contentTvl || "";
}

function getTranslationSource(route: RouteDecision, userMessage: string): string {
  return (route.translation_text || userMessage).trim();
}

function getTvlModelLabel(modelInfo: unknown): string {
  if (modelInfo && typeof modelInfo === "object") {
    const samplerPath = (modelInfo as any).sampler_path;
    if (typeof samplerPath === "string" && samplerPath) return samplerPath;
  }
  if (typeof modelInfo === "string" && modelInfo) return modelInfo;
  return "tvl-backend";
}

export async function GET(event: APIEvent) {
  return jsonResponse({
    ok: true,
    route: "/api/chat-router",
    router_model: getRouterModel(event),
    answer_model: getAnswerModel(event),
    tvl_model_name: getTinkerModelName(event),
    tvl_model_path: getTinkerModelPath(event),
    tvl_transport: getEnvValue(event, "TINKER_API_KEY") ? "tinker-http" : "vps-backend",
    translation_backend_fallback: getBackendUrl(event),
  });
}

export async function POST(event: APIEvent) {
  const contentLength = event.request.headers.get("content-length");
  if (contentLength && parseInt(contentLength, 10) > MAX_BODY_BYTES) {
    return jsonResponse({ error: "Request too large" }, 413);
  }

  const bodyResult = await readJsonRequest(event.request);
  if (!bodyResult.ok) return jsonResponse({ error: bodyResult.error }, bodyResult.status);

  const validation = validateChatBody(bodyResult.body);
  if (!validation.ok) return jsonResponse({ error: validation.error }, 400);

  const {
    sessionId,
    userMessage,
    providedHistory,
    requestedDisplayLanguage,
    temperature,
    maxCompletionTokens,
  } = validation.payload;

  const modelsUsed: ModelsUsed = {
    router: getRouterModel(event),
  };

  try {
    const route = await routeTurn(event, userMessage, requestedDisplayLanguage);
    let db: D1Database | null = null;

    let userContentEn: string | null = route.input_language === "en" ? userMessage : null;
    let userContentTvl: string | null = route.input_language === "tvl" ? userMessage : null;
    let assistantContentEn: string | null = null;
    let assistantContentTvl: string | null = null;

    if (route.needs_tvl_to_en) {
      const sourceText = getTranslationSource(route, userMessage);
      const translated = await translateWithTvlModel(event, "tvl_to_en", sourceText);
      modelsUsed.tvl_to_en = getTvlModelLabel(translated.modelInfo);
      userContentEn = translated.content;
      if (!userContentTvl && route.source_language === "tvl") userContentTvl = sourceText;
    }

    if (route.intent === "translate" && !route.needs_nano_answer) {
      const sourceText = getTranslationSource(route, userMessage);
      if (route.target_language === "tvl" || (route.target_language === "bilingual" && route.source_language !== "tvl")) {
        const translated = await translateWithTvlModel(event, "en_to_tvl", sourceText);
        modelsUsed.en_to_tvl = getTvlModelLabel(translated.modelInfo);
        assistantContentEn = sourceText;
        assistantContentTvl = translated.content;
      } else {
        assistantContentEn = userContentEn;
        if (assistantContentEn === null) {
          const translated = await translateWithTvlModel(event, "tvl_to_en", sourceText);
          modelsUsed.tvl_to_en = getTvlModelLabel(translated.modelInfo);
          assistantContentEn = translated.content;
        }
        assistantContentTvl = route.source_language === "tvl" ? sourceText : null;
      }
    } else {
      db = await getDb(event);
      const history = await loadHistory(db, sessionId);
      const englishHistory = buildEnglishHistory(history, providedHistory);
      const userMessageEn = userContentEn || userMessage;
      const directTvlAnswer = shouldAnswerDirectlyInTuvaluan(route);
      const nanoAnswer = await answerWithNano({
        event,
        history: englishHistory,
        userMessageEn,
        outputLanguage: directTvlAnswer ? "tvl" : "en",
        temperature,
        maxCompletionTokens,
      });
      modelsUsed.answer = getAnswerModel(event);

      if (directTvlAnswer) {
        assistantContentTvl = nanoAnswer;
      } else {
        assistantContentEn = nanoAnswer;
      }

      if (
        assistantContentEn &&
        (route.needs_en_to_tvl || route.display_language === "tvl" || route.display_language === "bilingual")
      ) {
        const translated = await translateWithTvlModel(event, "en_to_tvl", assistantContentEn);
        modelsUsed.en_to_tvl = getTvlModelLabel(translated.modelInfo);
        assistantContentTvl = translated.content;
      }
    }

    const display = formatDisplay(route.display_language, assistantContentEn, assistantContentTvl);
    const userDisplay = formatDisplay(route.input_language === "tvl" ? "tvl" : "en", userContentEn, userContentTvl);
    db ??= await getDb(event);

    await saveTurn({
      db,
      sessionId,
      requestedDisplayLanguage,
      route,
      modelsUsed,
      user: {
        original: userMessage,
        contentEn: userContentEn,
        contentTvl: userContentTvl,
        display: userDisplay || userMessage,
      },
      assistant: {
        original: display,
        contentEn: assistantContentEn,
        contentTvl: assistantContentTvl,
        display,
      },
    });

    return jsonResponse({
      ok: true,
      session_id: sessionId,
      content: display,
      content_en: assistantContentEn,
      content_tvl: assistantContentTvl,
      route,
      models_used: modelsUsed,
    });
  } catch (e: any) {
    console.error("Chat router API error:", e);
    const message =
      e?.message === "OPENROUTER_API_KEY is not configured"
        ? "OpenRouter API key is not configured"
        : "Chat router request failed";
    const status = e?.message === "OPENROUTER_API_KEY is not configured" ? 503 : 502;
    return jsonResponse({ error: message }, status);
  }
}
