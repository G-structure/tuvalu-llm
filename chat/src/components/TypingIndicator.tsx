export default function TypingIndicator() {
  return (
    <div class="py-5">
      <div class="max-w-3xl mx-auto flex gap-4 px-4">
        <div class="shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-medium bg-[var(--color-accent)]/15 text-[var(--color-accent)]">
          T
        </div>
        <div class="pt-2">
          <div class="flex gap-1">
            <span class="typing-dot w-1.5 h-1.5 bg-[var(--color-text-muted)] rounded-full" />
            <span class="typing-dot w-1.5 h-1.5 bg-[var(--color-text-muted)] rounded-full" />
            <span class="typing-dot w-1.5 h-1.5 bg-[var(--color-text-muted)] rounded-full" />
          </div>
        </div>
      </div>
    </div>
  );
}
