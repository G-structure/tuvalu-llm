import { createSignal } from "solid-js";

export default function ChatInput(props: {
  onSend: (text: string) => void;
  disabled: boolean;
}) {
  const [text, setText] = createSignal("");

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    const t = text().trim();
    if (!t || props.disabled) return;
    props.onSend(t);
    setText("");
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div class="pb-5 pt-3 px-4">
      <form
        onSubmit={handleSubmit}
        class="max-w-3xl mx-auto relative"
      >
        <textarea
          value={text()}
          onInput={(e) => setText(e.currentTarget.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message TVL Chat..."
          disabled={props.disabled}
          rows={1}
          class="w-full bg-[var(--color-input-bg)] text-[var(--color-text)] rounded-xl pl-4 pr-12 py-3.5 text-[14px] resize-none focus:outline-none focus:ring-1 focus:ring-[var(--color-accent)]/40 border border-[var(--color-border)] placeholder:text-[var(--color-text-muted)] disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={props.disabled || !text().trim()}
          class="absolute right-2.5 bottom-2.5 bg-[var(--color-accent)] text-[#080f1a] w-7 h-7 rounded-md flex items-center justify-center transition-opacity disabled:opacity-15 hover:brightness-110"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <path d="M3 13L13 8L3 3V7L9 8L3 9V13Z" fill="currentColor" />
          </svg>
        </button>
      </form>
      <p class="text-center text-[11px] text-[var(--color-text-muted)] mt-2">
        TVL Chat is an experimental bilingual Tuvaluan-English model currently in training.
      </p>
    </div>
  );
}
