import { createSignal, For, onMount, Show } from "solid-js";
import type { Message } from "~/lib/types";
import { renderMarkdown, initCopyButtons } from "~/lib/markdown";

const FEEDBACK_REASONS = [
  "Wrong word",
  "Wrong meaning",
  "Sounds unnatural",
  "Too formal",
  "Different dialect",
  "Not what I meant",
  "Other",
];

export default function ChatMessage(props: {
  message: Message;
  onFeedback?: (
    message: Message,
    rating: "good" | "needs_work" | "sounded_funny" | "fix_words",
    detail?: {
      reasons?: string[];
      note?: string;
      selected_text?: string;
      correction_text?: string;
    }
  ) => void;
}) {
  const isUser = () => props.message.role === "user";
  const [activePanel, setActivePanel] = createSignal<
    "needs_work" | "sounded_funny" | "fix_words" | null
  >(null);
  const [selectedReasons, setSelectedReasons] = createSignal<string[]>([]);
  const [note, setNote] = createSignal("");
  const [selectedText, setSelectedText] = createSignal("");
  const [correctionText, setCorrectionText] = createSignal("");
  const [saved, setSaved] = createSignal("");
  let contentRef: HTMLDivElement | undefined;

  onMount(() => {
    if (contentRef && !isUser()) {
      initCopyButtons(contentRef);
    }
  });

  const saveFeedback = (
    rating: "good" | "needs_work" | "sounded_funny" | "fix_words"
  ) => {
    props.onFeedback?.(props.message, rating, {
      reasons: selectedReasons(),
      note: note().trim(),
      selected_text: selectedText().trim(),
      correction_text: correctionText().trim(),
    });
    setSaved("Saved");
    window.setTimeout(() => setSaved(""), 1400);
  };

  const toggleReason = (reason: string) => {
    setSelectedReasons((current) =>
      current.includes(reason)
        ? current.filter((entry) => entry !== reason)
        : [...current, reason]
    );
  };

  return (
    <div class={`chat-message ${isUser() ? "chat-message--user" : "chat-message--assistant"}`}>
      <div class="chat-message__inner">
        <div
          class={`chat-message__avatar ${
            isUser()
              ? "chat-message__avatar--user"
              : "chat-message__avatar--assistant"
          }`}
          aria-hidden="true"
        >
          {isUser() ? "You" : "FI"}
        </div>

        <div class="chat-message__content">
          <div class="chat-message__label">
            {isUser() ? "You" : "Fenua AI"}
          </div>
          <Show
            when={!isUser()}
            fallback={
              <p class="chat-message__text">
                {props.message.content}
              </p>
            }
          >
            <div
              ref={contentRef}
              class="markdown-content chat-message__markdown"
              innerHTML={renderMarkdown(props.message.content)}
            />
          </Show>

          <Show when={!isUser() && props.onFeedback}>
            <div class="chat-feedback-actions" aria-label="Message feedback">
              <button type="button" onClick={() => saveFeedback("good")}>
                Good
              </button>
              <button
                type="button"
                class={activePanel() === "needs_work" ? "is-active" : ""}
                onClick={() => {
                  setActivePanel(activePanel() === "needs_work" ? null : "needs_work");
                  saveFeedback("needs_work");
                }}
              >
                Needs work
              </button>
              <button
                type="button"
                class={activePanel() === "fix_words" ? "is-active" : ""}
                onClick={() => {
                  setActivePanel(activePanel() === "fix_words" ? null : "fix_words");
                }}
              >
                Fix words
              </button>
              <button
                type="button"
                class={activePanel() === "sounded_funny" ? "is-active" : ""}
                onClick={() => {
                  setActivePanel(
                    activePanel() === "sounded_funny" ? null : "sounded_funny"
                  );
                  saveFeedback("sounded_funny");
                }}
              >
                Sounded funny
              </button>
            </div>

            <Show when={saved()}>
              <div class="chat-feedback-saved" role="status" aria-live="polite">
                {saved()}
              </div>
            </Show>

            <Show when={activePanel()}>
              <section class="chat-feedback-panel">
                <div class="chat-feedback-reasons">
                  <Show
                    when={activePanel() !== "fix_words"}
                    fallback={
                      <div class="chat-feedback-fix-grid">
                        <label>
                          <span>Words to fix</span>
                          <input
                            value={selectedText()}
                            onInput={(event) =>
                              setSelectedText(event.currentTarget.value)
                            }
                            placeholder="Copy the word or phrase"
                          />
                        </label>
                        <label>
                          <span>Better wording</span>
                          <input
                            value={correctionText()}
                            onInput={(event) =>
                              setCorrectionText(event.currentTarget.value)
                            }
                            placeholder="Write the correction"
                          />
                        </label>
                      </div>
                    }
                  >
                    <For each={FEEDBACK_REASONS}>
                      {(reason) => (
                        <button
                          type="button"
                          class={selectedReasons().includes(reason) ? "is-active" : ""}
                          aria-pressed={selectedReasons().includes(reason)}
                          onClick={() => toggleReason(reason)}
                        >
                          {reason}
                        </button>
                      )}
                    </For>
                  </Show>
                </div>
                <label>
                  <span>Optional note</span>
                  <textarea
                    value={note()}
                    onInput={(event) => setNote(event.currentTarget.value)}
                    onBlur={() =>
                      saveFeedback(activePanel() || "needs_work")
                    }
                    placeholder={
                      activePanel() === "fix_words"
                        ? "Type the better wording or explain the word to fix."
                        : "A short note is enough."
                    }
                    rows={2}
                  />
                </label>
                <button
                  type="button"
                  class="chat-feedback-save"
                  onClick={() => saveFeedback(activePanel() || "needs_work")}
                >
                  Save feedback
                </button>
              </section>
            </Show>
          </Show>
        </div>
      </div>
    </div>
  );
}
