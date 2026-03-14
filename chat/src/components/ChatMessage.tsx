import { Show } from "solid-js";
import type { Message } from "~/lib/types";
import { renderMarkdown } from "~/lib/markdown";

export default function ChatMessage(props: { message: Message }) {
  const isUser = () => props.message.role === "user";

  return (
    <div class="py-5">
      <div class="max-w-3xl mx-auto flex gap-4 px-4">
        <div
          class={`shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-medium ${
            isUser()
              ? "bg-white/[0.08] text-[var(--color-text-secondary)]"
              : "bg-[var(--color-accent)]/15 text-[var(--color-accent)]"
          }`}
        >
          {isUser() ? "U" : "T"}
        </div>

        <div class="min-w-0 flex-1">
          <div class="text-[11px] font-medium mb-1.5 text-[var(--color-text-muted)]">
            {isUser() ? "You" : "TVL Model"}
          </div>
          <Show
            when={!isUser()}
            fallback={
              <p class="text-[14px] leading-relaxed whitespace-pre-wrap text-[var(--color-text)]">
                {props.message.content}
              </p>
            }
          >
            <div
              class="markdown-content text-[14px] leading-relaxed"
              innerHTML={renderMarkdown(props.message.content)}
            />
          </Show>
        </div>
      </div>
    </div>
  );
}
