import { createMemo, createSignal, For, onMount, Show } from "solid-js";
import type { Message } from "~/lib/types";
import { ensureCommunitySessionId, getKnownIsland } from "~/lib/community";
import ChatMessage from "~/components/chat/ChatMessage";
import ChatInput from "~/components/chat/ChatInput";
import TypingIndicator from "~/components/chat/TypingIndicator";
import ModelBadge from "~/components/chat/ModelBadge";
import OGMeta from "~/components/OGMeta";
import StructuredData from "~/components/StructuredData";
import { languageLabOrganization, languageLabWebsite } from "~/lib/seo";
import { absoluteChatUrl, CHAT_META, SITE_ORIGINS } from "~/lib/site";

const STORAGE_KEY = "fenua.chat.conversations.v1";
const ACTIVE_KEY = "fenua.chat.active.v1";
const SYNC_KEY = "fenua.chat.sync.v1";
const MAX_LOCAL_CONVERSATIONS = 24;

interface StoredConversation {
  id: string;
  title: string;
  messages: Message[];
  created_at: string;
  updated_at: string;
  synced_at?: string | null;
  consent_state: "sync_training" | "local_only";
}

const CHAT_SUGGESTIONS = [
  {
    label: "EN",
    title: "Translate",
    detail: "The ocean is beautiful today",
    prompt: "Translate: The ocean is beautiful today.",
  },
  {
    label: "TV",
    title: "Say it locally",
    detail: "A warm welcome for a visitor",
    prompt: "Greet a visitor warmly in Tuvaluan.",
  },
  {
    label: "AI",
    title: "Explain",
    detail: "A phrase from school or family",
    prompt: "Explain a Tuvaluan phrase in English.",
  },
  {
    label: "FT",
    title: "Match day",
    detail: "A football phrase in context",
    prompt: "Explain a football phrase.",
  },
];

function newId(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}_${crypto.randomUUID()}`;
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

function nowIso(): string {
  return new Date().toISOString();
}

function createConversation(): StoredConversation {
  const now = nowIso();
  return {
    id: newId("chat"),
    title: "New chat",
    messages: [],
    created_at: now,
    updated_at: now,
    consent_state: "sync_training",
  };
}

function titleFromMessages(messages: Message[]): string {
  const firstUser = messages.find((message) => message.role === "user");
  const raw = firstUser?.content.replace(/^translate:\s*/i, "").trim();
  if (!raw) return "New chat";
  return raw.length > 54 ? `${raw.slice(0, 51).trim()}...` : raw;
}

function readLocalConversations(): StoredConversation[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((conversation) => conversation?.id && conversation?.title)
      .map((conversation) => ({
        id: String(conversation.id),
        title: String(conversation.title || "New chat"),
        messages: Array.isArray(conversation.messages)
          ? conversation.messages.filter((message: Message) => message?.content)
          : [],
        created_at: String(conversation.created_at || nowIso()),
        updated_at: String(conversation.updated_at || nowIso()),
        synced_at: conversation.synced_at || null,
        consent_state:
          conversation.consent_state === "local_only"
            ? "local_only"
            : "sync_training",
      }))
      .slice(0, MAX_LOCAL_CONVERSATIONS);
  } catch {
    return [];
  }
}

function sortConversations(conversations: StoredConversation[]) {
  return [...conversations].sort(
    (a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at)
  );
}

function formatConversationTime(value: string): string {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return "Now";
  const diff = Date.now() - timestamp;
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (diff < minute) return "Now";
  if (diff < hour) return `${Math.max(1, Math.round(diff / minute))}m`;
  if (diff < day) return `${Math.round(diff / hour)}h`;
  return new Date(timestamp).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export default function Chat() {
  const [conversations, setConversations] = createSignal<StoredConversation[]>([]);
  const [activeId, setActiveId] = createSignal("");
  const [sessionId, setSessionId] = createSignal("");
  const [loading, setLoading] = createSignal(false);
  const [pendingConversationId, setPendingConversationId] = createSignal("");
  const [syncEnabled, setSyncEnabled] = createSignal(true);
  const [syncState, setSyncState] = createSignal<"saved" | "saving" | "offline">("saved");
  let messagesEnd: HTMLDivElement | undefined;

  const activeConversation = createMemo(() =>
    conversations().find((conversation) => conversation.id === activeId())
  );

  const messages = createMemo(() => activeConversation()?.messages || []);

  const isConstrainedNetwork = () => {
    const connection = (navigator as any).connection;
    return (
      !!connection?.saveData ||
      ["slow-2g", "2g"].includes(connection?.effectiveType)
    );
  };

  const persistLocal = (
    nextConversations = conversations(),
    nextActiveId = activeId()
  ) => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(nextConversations.slice(0, MAX_LOCAL_CONVERSATIONS))
    );
    if (nextActiveId) localStorage.setItem(ACTIVE_KEY, nextActiveId);
  };

  const scrollToBottom = () => {
    setTimeout(
      () =>
        messagesEnd?.scrollIntoView({
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
            ? "auto"
            : "smooth",
        }),
      50
    );
  };

  const replaceConversations = (
    nextConversations: StoredConversation[],
    nextActiveId = activeId()
  ) => {
    const sorted = sortConversations(nextConversations).slice(
      0,
      MAX_LOCAL_CONVERSATIONS
    );
    setConversations(sorted);
    if (nextActiveId) setActiveId(nextActiveId);
    persistLocal(sorted, nextActiveId);
  };

  const commitConversation = (updated: StoredConversation) => {
    const next = conversations().some((conversation) => conversation.id === updated.id)
      ? conversations().map((conversation) =>
          conversation.id === updated.id ? updated : conversation
        )
      : [updated, ...conversations()];
    replaceConversations(next, updated.id);
    return updated;
  };

  const syncConversation = async (conversation: StoredConversation) => {
    if (!syncEnabled() || !sessionId() || conversation.messages.length === 0) {
      return;
    }

    setSyncState("saving");
    try {
      const resp = await fetch("/api/chat/conversations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation: {
            id: conversation.id,
            session_id: sessionId(),
            title: conversation.title,
            messages: conversation.messages,
            source: "web",
            language_mode: "tvl-en",
            island: getKnownIsland(),
            consent_state: "sync_training",
            created_at: conversation.created_at,
            updated_at: conversation.updated_at,
            metadata: {
              local_cache: true,
              client: "solid-chat",
            },
          },
        }),
      });

      if (!resp.ok) throw new Error("sync failed");
      const data = await resp.json().catch(() => ({}));
      setSyncState("saved");
      if (data?.synced_at) {
        const refreshed = {
          ...conversation,
          synced_at: data.synced_at,
          consent_state: "sync_training" as const,
        };
        const next = conversations().map((item) =>
          item.id === refreshed.id ? refreshed : item
        );
        setConversations(next);
        persistLocal(next, activeId());
      }
    } catch {
      setSyncState("offline");
    }
  };

  const loadRemoteConversations = async (sid: string) => {
    try {
      const resp = await fetch(
        `/api/chat/conversations?session_id=${encodeURIComponent(sid)}&include=messages`
      );
      if (!resp.ok) return;
      const data = await resp.json();
      const remote: StoredConversation[] = (data.conversations || []).map(
        (conversation: any) => ({
          id: conversation.id,
          title: conversation.title || "New chat",
          messages: Array.isArray(conversation.messages)
            ? conversation.messages
            : [],
          created_at: conversation.created_at || nowIso(),
          updated_at: conversation.updated_at || conversation.created_at || nowIso(),
          synced_at: conversation.synced_at || null,
          consent_state:
            conversation.consent_state === "local_only"
              ? "local_only"
              : "sync_training",
        })
      );
      if (remote.length === 0) return;

      const byId = new Map<string, StoredConversation>();
      for (const conversation of [...conversations(), ...remote]) {
        const existing = byId.get(conversation.id);
        if (
          !existing ||
          Date.parse(conversation.updated_at) > Date.parse(existing.updated_at)
        ) {
          byId.set(conversation.id, conversation);
        }
      }
      const merged = sortConversations([...byId.values()]);
      const nextActive = activeId() || merged[0]?.id || "";
      replaceConversations(merged, nextActive);
    } catch {
      setSyncState("offline");
    }
  };

  onMount(() => {
    const sid = ensureCommunitySessionId();
    const local = readLocalConversations();
    const initial = local.length ? local : [createConversation()];
    const storedActive = localStorage.getItem(ACTIVE_KEY) || initial[0].id;
    const canSync = localStorage.getItem(SYNC_KEY) !== "local_only";

    setSessionId(sid);
    setSyncEnabled(canSync);
    replaceConversations(
      initial,
      initial.some((conversation) => conversation.id === storedActive)
        ? storedActive
        : initial[0].id
    );

    if (canSync) void loadRemoteConversations(sid);
  });

  const setSyncPreference = (enabled: boolean) => {
    setSyncEnabled(enabled);
    localStorage.setItem(SYNC_KEY, enabled ? "sync_training" : "local_only");
    setSyncState(enabled ? "saving" : "saved");
    const next = conversations().map((conversation) => ({
      ...conversation,
      consent_state: enabled ? ("sync_training" as const) : ("local_only" as const),
    }));
    replaceConversations(next);
    if (enabled) {
      for (const conversation of next) {
        if (conversation.messages.length > 0) void syncConversation(conversation);
      }
    }
  };

  const startNewChat = () => {
    const conversation = {
      ...createConversation(),
      consent_state: syncEnabled()
        ? ("sync_training" as const)
        : ("local_only" as const),
    };
    commitConversation(conversation);
  };

  const deleteConversation = async (id: string) => {
    const remaining = conversations().filter((conversation) => conversation.id !== id);
    const next = remaining.length ? remaining : [createConversation()];
    const nextActive = activeId() === id ? next[0].id : activeId();
    replaceConversations(next, nextActive);

    if (syncEnabled() && sessionId()) {
      fetch(
        `/api/chat/conversations?id=${encodeURIComponent(id)}&session_id=${encodeURIComponent(sessionId())}`,
        { method: "DELETE" }
      ).catch(() => {});
    }
  };

  const updateMessages = (
    conversationId: string,
    nextMessages: Message[]
  ): StoredConversation | undefined => {
    const current = conversations().find(
      (conversation) => conversation.id === conversationId
    );
    if (!current) return undefined;
    return commitConversation({
      ...current,
      title:
        current.title === "New chat"
          ? titleFromMessages(nextMessages)
          : current.title,
      messages: nextMessages,
      updated_at: nowIso(),
      consent_state: syncEnabled() ? "sync_training" : "local_only",
    });
  };

  const sendMessage = async (text: string) => {
    const current = activeConversation() || createConversation();
    if (!activeConversation()) commitConversation(current);

    const userMessage: Message = {
      id: newId("msg"),
      role: "user",
      content: text,
      created_at: nowIso(),
    };
    const requestMessages = [...current.messages, userMessage];
    const updated = updateMessages(current.id, requestMessages);
    if (updated) void syncConversation(updated);

    setLoading(true);
    setPendingConversationId(current.id);
    scrollToBottom();

    const assistantMessageId = newId("msg");

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: requestMessages,
          temperature: 0.3,
          max_tokens: isConstrainedNetwork() ? 512 : 1024,
          conversation_id: current.id,
          session_id: sessionId(),
          title: titleFromMessages(requestMessages),
          assistant_message_id: assistantMessageId,
          language_mode: "tvl-en",
          island: getKnownIsland(),
          consent_state: syncEnabled() ? "sync_training" : "local_only",
          source: "web",
        }),
      });

      if (!resp.ok) {
        const offline = addOfflineMessage(current.id, requestMessages);
        if (offline) void syncConversation(offline);
        return;
      }

      const data = await resp.json();
      const assistantMessage: Message = {
        id: assistantMessageId,
        role: "assistant",
        content: data.content,
        created_at: nowIso(),
      };
      const saved = updateMessages(current.id, [
        ...requestMessages,
        assistantMessage,
      ]);
      if (saved) void syncConversation(saved);
    } catch {
      const offline = addOfflineMessage(current.id, requestMessages);
      if (offline) void syncConversation(offline);
    } finally {
      setLoading(false);
      setPendingConversationId("");
      scrollToBottom();
    }
  };

  const addOfflineMessage = (
    conversationId: string,
    requestMessages: Message[]
  ) =>
    updateMessages(conversationId, [
      ...requestMessages,
      {
        id: newId("msg"),
        role: "assistant",
        content:
          "The local Fenua model is offline right now. Your chat is saved on this device and will sync when the site can reach storage again.",
        created_at: nowIso(),
      },
    ]);

  const sendFeedback = async (
    message: Message,
    rating: "good" | "needs_work" | "sounded_funny" | "fix_words",
    detail?: {
      reasons?: string[];
      note?: string;
      selected_text?: string;
      correction_text?: string;
    }
  ) => {
    const conversation = activeConversation();
    if (!conversation || !syncEnabled() || !sessionId()) return;

    fetch("/api/chat/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: newId("feedback"),
        conversation_id: conversation.id,
        message_id: message.id || null,
        session_id: sessionId(),
        rating,
        correction_text: detail?.correction_text || detail?.note || null,
        selected_text: detail?.selected_text || null,
        island: getKnownIsland(),
        metadata: {
          title: conversation.title,
          reasons: detail?.reasons || [],
          note: detail?.note || "",
        },
      }),
    }).catch(() => {});
  };

  return (
    <>
      <OGMeta
        title={CHAT_META.productName}
        description={CHAT_META.productTagline}
        url={absoluteChatUrl("/chat")}
        image={CHAT_META.defaultOgImage}
        imageOrigin={SITE_ORIGINS.chat}
        imageWidth={CHAT_META.defaultOgImageWidth}
        imageHeight={CHAT_META.defaultOgImageHeight}
        imageAlt={CHAT_META.defaultOgImageAlt}
        siteName={CHAT_META.productName}
        titleSuffix="Tuvaluan-English AI"
      />
      <StructuredData
        data={[
          languageLabOrganization(),
          languageLabWebsite(),
          {
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            name: CHAT_META.productName,
            applicationCategory: "AIApplication",
            operatingSystem: "Web",
            url: absoluteChatUrl("/chat"),
            description: CHAT_META.productTagline,
            publisher: {
              "@id": `${SITE_ORIGINS.organization}/#organization`,
            },
            inLanguage: ["tvl", "en"],
          },
        ]}
      />
      <div class="chat-theme chat-shell chat-shell--saved flex flex-col">
        <nav aria-label="Chat navigation" class="chat-nav">
          <div class="chat-nav__left">
            <a href="/" class="chat-nav__brand">
              <span class="chat-nav__brand-mark" aria-hidden="true" />
              <span>
                <strong>Fenua</strong>
                <em>Intelligence</em>
              </span>
            </a>
            <h1 class="chat-nav__title">TVL Chat</h1>
            <ModelBadge />
          </div>
          <div class="chat-nav__actions">
            <a href="/chat/training" class="chat-nav__link">
              Training
            </a>
            <button type="button" onClick={startNewChat} class="chat-nav__button">
              New chat
            </button>
          </div>
        </nav>

        <div class="chat-body">
          <aside class="chat-rail" aria-label="Saved conversations">
            <div class="chat-rail__header">
              <div>
                <p>Saved chats</p>
                <strong>{conversations().length}</strong>
              </div>
              <button type="button" onClick={startNewChat} aria-label="New chat">
                +
              </button>
            </div>

            <div class="chat-sync-card">
              <div>
                <span>{syncEnabled() ? "Sync on" : "Local only"}</span>
                <p>
                  {syncEnabled()
                    ? "Saved on device and D1."
                    : "Only this device stores chats."}
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={syncEnabled()}
                class={`chat-sync-toggle ${syncEnabled() ? "is-on" : ""}`}
                onClick={() => setSyncPreference(!syncEnabled())}
              >
                <span />
              </button>
            </div>

            <div class="chat-rail__list">
              <For each={conversations()}>
                {(conversation) => (
                  <button
                    type="button"
                    class={`chat-thread-card ${
                      activeId() === conversation.id ? "is-active" : ""
                    }`}
                    onClick={() => {
                      setActiveId(conversation.id);
                      persistLocal(conversations(), conversation.id);
                      scrollToBottom();
                    }}
                  >
                    <span class="chat-thread-card__title">
                      {conversation.title}
                    </span>
                    <span class="chat-thread-card__meta">
                      {conversation.messages.length} messages
                      <span>{formatConversationTime(conversation.updated_at)}</span>
                    </span>
                  </button>
                )}
              </For>
            </div>
          </aside>

          <section class="chat-workspace" aria-label="Current conversation">
            <div class="chat-thread-bar">
              <div>
                <p>Current conversation</p>
                <h2>{activeConversation()?.title || "New chat"}</h2>
              </div>
              <div class={`chat-sync-status chat-sync-status--${syncState()}`}>
                <span />
                {syncEnabled()
                  ? syncState() === "saving"
                    ? "Syncing"
                    : syncState() === "offline"
                      ? "Saved locally"
                      : "Synced"
                  : "Local only"}
              </div>
              <Show when={activeConversation()?.messages.length}>
                <button
                  type="button"
                  class="chat-thread-delete"
                  onClick={() => {
                    const id = activeId();
                    if (id) void deleteConversation(id);
                  }}
                >
                  Delete
                </button>
              </Show>
            </div>

            <div class="chat-main flex-1 overflow-y-auto">
              <Show
                when={messages().length > 0}
                fallback={
                  <div class="chat-empty">
                    <div class="chat-empty__inner">
                      <div class="chat-empty__intro">
                        <div class="chat-empty__mark" aria-hidden="true">
                          <span />
                        </div>
                        <p class="chat-empty__eyebrow">Fenua Intelligence</p>
                        <h2>Talofa. What do you want to say?</h2>
                        <p class="chat-empty__primary">
                          Tuvaluan and English, saved across conversations.
                        </p>
                      </div>
                      <div class="chat-suggestions">
                        <For each={CHAT_SUGGESTIONS}>
                          {(suggestion) => (
                            <Suggestion
                              suggestion={suggestion}
                              onClick={sendMessage}
                            />
                          )}
                        </For>
                      </div>
                    </div>
                  </div>
                }
              >
                <For each={messages()}>
                  {(msg) => (
                    <ChatMessage
                      message={msg}
                      onFeedback={
                        msg.role === "assistant" && syncEnabled()
                          ? sendFeedback
                          : undefined
                      }
                    />
                  )}
                </For>
                <Show when={loading() && pendingConversationId() === activeId()}>
                  <TypingIndicator />
                </Show>
                <div ref={messagesEnd} class="h-4" />
              </Show>
            </div>

            <div class="chat-privacy-strip">
              <span aria-hidden="true" />
              <p>
                Chats stay on this device first. Sync stores JSON transcripts in
                D1 so Fenua can learn from real Tuvaluan-English use.{" "}
                <a href="/legal#privacy">Privacy and Terms</a>
              </p>
            </div>
            <ChatInput
              onSend={sendMessage}
              disabled={loading() && pendingConversationId() === activeId()}
            />
          </section>
        </div>
      </div>
    </>
  );
}

function Suggestion(props: {
  suggestion: (typeof CHAT_SUGGESTIONS)[number];
  onClick: (t: string) => void;
}) {
  return (
    <button
      onClick={() => props.onClick(props.suggestion.prompt)}
      class="chat-suggestion"
    >
      <span class="chat-suggestion__icon" aria-hidden="true">
        {props.suggestion.label}
      </span>
      <span class="chat-suggestion__copy">
        <span class="chat-suggestion__title">{props.suggestion.title}</span>
        <span class="chat-suggestion__body">{props.suggestion.detail}</span>
      </span>
    </button>
  );
}
