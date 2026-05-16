import { createSignal } from "solid-js";

export default function ChatInput(props: {
  onSend: (text: string) => void;
  disabled: boolean;
  value?: string;
  editing?: boolean;
  onCancelEdit?: () => void;
  onValueChange?: (text: string) => void;
}) {
  const [localText, setLocalText] = createSignal("");
  let inputRef: HTMLTextAreaElement | undefined;
  const text = () => props.value ?? localText();
  const setText = (value: string) => {
    props.onValueChange?.(value);
    if (props.value === undefined) setLocalText(value);
  };

  const resizeInput = () => {
    if (!inputRef) return;
    inputRef.style.height = "0px";
    inputRef.style.height = `${Math.min(inputRef.scrollHeight, 190)}px`;
    inputRef.style.overflowY = inputRef.scrollHeight > 190 ? "auto" : "hidden";
  };

  const handleSubmit = (e: Event) => {
    e.preventDefault();
    const t = text().trim();
    if (!t || props.disabled) return;
    props.onSend(t);
    if (!props.editing) setText("");
    requestAnimationFrame(resizeInput);
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div class="chat-input-wrap">
      <form
        onSubmit={handleSubmit}
        class={`chat-input-form ${props.editing ? "is-editing" : ""}`}
      >
        {props.editing && (
          <div class="chat-input-edit-banner">
            <span>Editing your message</span>
            <button
              type="button"
              onClick={() => {
                setText("");
                props.onCancelEdit?.();
                requestAnimationFrame(resizeInput);
              }}
            >
              Cancel
            </button>
          </div>
        )}
        <label for="chat-input" class="sr-only">Message</label>
        <textarea
          ref={inputRef}
          id="chat-input"
          value={text()}
          onInput={(e) => {
            setText(e.currentTarget.value);
            resizeInput();
          }}
          onKeyDown={handleKeyDown}
          placeholder="Ask in Tuvaluan or English"
          disabled={props.disabled}
          rows={1}
          class="chat-input"
        />
        <button
          type="submit"
          disabled={props.disabled || !text().trim()}
          aria-label={props.editing ? "Save edited message" : "Send message"}
          class="chat-send-button"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M3 13L13 8L3 3V7L9 8L3 9V13Z" fill="currentColor" />
          </svg>
        </button>
      </form>
    </div>
  );
}
