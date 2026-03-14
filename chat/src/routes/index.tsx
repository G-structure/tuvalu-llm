import { createSignal, For, Show } from "solid-js";
import { Title } from "@solidjs/meta";
import type { Message } from "~/lib/types";
import ChatMessage from "~/components/ChatMessage";
import ChatInput from "~/components/ChatInput";
import TypingIndicator from "~/components/TypingIndicator";
import ModelBadge from "~/components/ModelBadge";

export default function Chat() {
  const [messages, setMessages] = createSignal<Message[]>([]);
  const [loading, setLoading] = createSignal(false);
  let messagesEnd: HTMLDivElement | undefined;

  const scrollToBottom = () => {
    setTimeout(() => messagesEnd?.scrollIntoView({ behavior: "smooth" }), 50);
  };

  const sendMessage = async (text: string) => {
    const userMsg: Message = { role: "user", content: text };
    const updated = [...messages(), userMsg];
    setMessages(updated);
    setLoading(true);
    scrollToBottom();

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: updated,
          temperature: 0.3,
          max_tokens: 1024,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: "Request failed" }));
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${err.error || resp.statusText}` },
        ]);
        return;
      }

      const data = await resp.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.content },
      ]);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${e.message}` },
      ]);
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <>
      <Title>TVL Chat</Title>
      <div class="h-screen flex flex-col bg-[var(--color-bg)]">
        {/* Header */}
        <nav class="flex items-center justify-between px-6 h-12 border-b border-[var(--color-border)]">
          <div class="flex items-center gap-3">
            <span class="text-[var(--color-accent)] text-[10px]">&#10038;</span>
            <h1 class="text-[14px] font-medium text-[var(--color-text)]">TVL Chat</h1>
            <ModelBadge />
          </div>
          <div class="flex items-center gap-4">
            <a
              href="/training"
              class="text-[12px] text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
            >
              Training
            </a>
            <button
              onClick={clearChat}
              class="text-[12px] text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
            >
              New chat
            </button>
          </div>
        </nav>

        {/* Messages area */}
        <div class="flex-1 overflow-y-auto">
          <Show
            when={messages().length > 0}
            fallback={
              <div class="h-full flex items-center justify-center">
                <div class="text-center max-w-md px-4">
                  <div class="w-14 h-14 rounded-full bg-[var(--color-surface-2)] flex items-center justify-center text-2xl mx-auto mb-5">
                    🏝️
                  </div>
                  <h2 class="text-[18px] font-semibold text-[var(--color-text)] mb-1.5">
                    Talofa!
                  </h2>
                  <p class="text-[13px] text-[var(--color-text-secondary)] mb-1">
                    Chat with a bilingual Tuvaluan-English LLM training live.
                  </p>
                  <p class="text-[12px] text-[var(--color-text-muted)] mb-8">
                    Te gagana o Tuvalu — from the islands to the world
                  </p>
                  <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    <Suggestion
                      text="Translate to Tuvaluan: The ocean is beautiful today."
                      onClick={sendMessage}
                    />
                    <Suggestion
                      text="Tell me a short story in Tuvaluan."
                      onClick={sendMessage}
                    />
                    <Suggestion
                      text="What is Tuvalu? Respond in Tuvaluan."
                      onClick={sendMessage}
                    />
                    <Suggestion
                      text="Explain a Python sort function in Tuvaluan."
                      onClick={sendMessage}
                    />
                  </div>
                </div>
              </div>
            }
          >
            <For each={messages()}>
              {(msg) => <ChatMessage message={msg} />}
            </For>
            <Show when={loading()}>
              <TypingIndicator />
            </Show>
            <div ref={messagesEnd} class="h-4" />
          </Show>
        </div>

        {/* Input */}
        <ChatInput onSend={sendMessage} disabled={loading()} />
      </div>
    </>
  );
}

function Suggestion(props: { text: string; onClick: (t: string) => void }) {
  return (
    <button
      onClick={() => props.onClick(props.text)}
      class="text-left text-[13px] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)] border border-[var(--color-border)] rounded-lg px-4 py-3 transition-colors"
    >
      {props.text}
    </button>
  );
}
