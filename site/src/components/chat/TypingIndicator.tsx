export default function TypingIndicator() {
  return (
    <div class="chat-message chat-message--assistant">
      <div class="chat-message__inner">
        <div class="chat-message__avatar chat-message__avatar--assistant">
          FI
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
