export default function TypingIndicator() {
  return (
    <div class="chat-message chat-message--assistant chat-message--loading">
      <div class="chat-message__inner">
        <div class="chat-message__avatar chat-message__avatar--assistant">
          FI
        </div>
        <div
          class="chat-message__content chat-loading-message"
          role="status"
          aria-live="polite"
          aria-busy="true"
        >
          <div class="chat-message__label">
            <span>Fenua AI</span>
          </div>
          <div class="chat-loading-message__body">
            <span>Writing a reply</span>
            <span class="chat-loading-message__dots" aria-hidden="true">
              <span class="typing-dot" />
              <span class="typing-dot" />
              <span class="typing-dot" />
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
