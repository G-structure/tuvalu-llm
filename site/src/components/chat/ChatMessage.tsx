import { createEffect, createMemo, createSignal, For, onMount, Show, untrack } from "solid-js";
import type { JSX } from "solid-js";
import type { Message } from "~/lib/types";
import {
  applyCorrectionsToText,
  buildSelectionFromWordRange,
  CHAT_FEEDBACK_REASON_OPTIONS,
  clampSelectionRange,
  createCorrectionFromSelection,
  createFeedbackEvent,
  hasChatMessageFeedback,
  normalizeChatMessageFeedback,
  normalizeMessageForFeedback,
  tokenizeSelectableText,
  type ChatFeedbackEvent,
  type ChatFeedbackReason,
  type ChatFeedbackType,
  type ChatMessageFeedback,
  type ChatWordCorrection,
  type FeedbackSelection,
  type SelectableTextToken,
} from "~/lib/chat-feedback";
import { renderMarkdown, initCopyButtons } from "~/lib/markdown";

type ActivePanel = "needs_work" | "sounded_funny" | "fix_words" | "say_more" | null;

export default function ChatMessage(props: {
  message: Message;
  canRegenerate?: boolean;
  showTimestamp?: boolean;
  wide?: boolean;
  onEdit?: (message: Message) => void;
  onRegenerate?: () => void;
  onFeedbackChange?: (message: Message, feedback: ChatMessageFeedback) => void;
  onFeedbackEvent?: (
    message: Message,
    event: ChatFeedbackEvent,
    feedback: ChatMessageFeedback
  ) => void;
}) {
  const isUser = () => props.message.role === "user";
  const feedback = createMemo(() =>
    normalizeChatMessageFeedback(props.message.feedback)
  );
  const normalizedText = createMemo(() =>
    normalizeMessageForFeedback(props.message.content).text
  );
  const tokens = createMemo(() => tokenizeSelectableText(normalizedText()));
  const [activePanel, setActivePanel] = createSignal<ActivePanel>(null);
  const [selectedReasons, setSelectedReasons] = createSignal<ChatFeedbackReason[]>([]);
  const [noteDraft, setNoteDraft] = createSignal("");
  const [sayMoreDraft, setSayMoreDraft] = createSignal("");
  const [fixSelection, setFixSelection] = createSignal<FeedbackSelection | null>(null);
  const [replacementText, setReplacementText] = createSignal("");
  const [showOriginal, setShowOriginal] = createSignal(false);
  const [saved, setSaved] = createSignal("");
  const [copied, setCopied] = createSignal(false);
  let contentRef: HTMLDivElement | undefined;
  let editInputRef: HTMLInputElement | undefined;
  let lastMessageKey = "";

  onMount(() => {
    if (contentRef && !isUser()) initCopyButtons(contentRef);
  });

  createEffect(() => {
    const nextMessageKey = props.message.id || `${props.message.role}:${props.message.content}`;
    if (nextMessageKey === lastMessageKey) return;
    lastMessageKey = nextMessageKey;
    const nextFeedback = untrack(feedback);
    setSelectedReasons(nextFeedback.reasons);
    setNoteDraft(nextFeedback.note);
    setSayMoreDraft(nextFeedback.say_more);
    setFixSelection(null);
    setReplacementText("");
    setActivePanel(null);
  });

  createEffect(() => {
    const selection = fixSelection();
    if (!selection || activePanel() !== "fix_words") return;
    window.requestAnimationFrame(() => {
      editInputRef?.focus();
      editInputRef?.select();
    });
  });

  const showStatus = (text: string) => {
    setSaved(text);
    window.setTimeout(() => setSaved(""), 1500);
  };

  const timestamp = () => {
    if (!props.showTimestamp || !props.message.created_at) return "";
    const parsed = Date.parse(props.message.created_at);
    if (!Number.isFinite(parsed)) return "";
    return new Date(parsed).toLocaleTimeString(undefined, {
      hour: "numeric",
      minute: "2-digit",
    });
  };

  const pushFeedback = (
    patch: Partial<ChatMessageFeedback>,
    eventInput?: Parameters<typeof createFeedbackEvent>[0]
  ) => {
    const event = eventInput ? createFeedbackEvent(eventInput) : undefined;
    const next = normalizeChatMessageFeedback({
      ...feedback(),
      ...patch,
      events: event ? [...feedback().events, event] : feedback().events,
    });
    props.onFeedbackChange?.(props.message, next);
    if (event) props.onFeedbackEvent?.(props.message, event, next);
    return next;
  };

  const recordFeedbackEvent = (
    eventInput: Parameters<typeof createFeedbackEvent>[0]
  ) => {
    const event = createFeedbackEvent(eventInput);
    const next = normalizeChatMessageFeedback({
      ...feedback(),
      events: [...feedback().events, event],
    });
    props.onFeedbackEvent?.(props.message, event, next);
    return event;
  };

  const copyMessage = async () => {
    try {
      await navigator.clipboard.writeText(props.message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      showStatus("Copy failed");
    }
  };

  const toggleHelpful = () => {
    const nextRating = feedback().rating === "good" ? null : "good";
    pushFeedback(
      {
        rating: nextRating,
        reasons: nextRating ? [] : feedback().reasons,
        note: nextRating ? "" : feedback().note,
      },
      {
        action: nextRating ? "saved" : "dismissed",
        type: "helpful",
      }
    );
    setActivePanel(null);
    showStatus(nextRating ? "Saved" : "Cleared");
  };

  const openPanel = (panel: Exclude<ActivePanel, null>) => {
    if (activePanel() === panel) {
      setActivePanel(null);
      recordFeedbackEvent({ action: "dismissed", type: panelToType(panel) });
      return;
    }

    setActivePanel(panel);
    setSelectedReasons(feedback().reasons);
    setNoteDraft(feedback().note);
    setSayMoreDraft(feedback().say_more);
    setFixSelection(null);
    setReplacementText("");

    recordFeedbackEvent({ action: "opened", type: panelToType(panel) });
  };

  const saveReasonFeedback = () => {
    const panel = activePanel();
    if (panel !== "needs_work" && panel !== "sounded_funny") return;
    pushFeedback(
      {
        rating: panel,
        reasons: selectedReasons(),
        note: noteDraft().trim(),
      },
      {
        action: "saved",
        freeformComment: noteDraft().trim() || undefined,
        selectedReasons: selectedReasons(),
        type: "not_right",
      }
    );
    showStatus("Saved");
  };

  const saveSayMore = () => {
    const comment = sayMoreDraft().trim();
    pushFeedback(
      { say_more: comment },
      {
        action: comment ? "saved" : "dismissed",
        freeformComment: comment || undefined,
        type: "say_more",
      }
    );
    showStatus(comment ? "Saved" : "Cleared");
  };

  const toggleReason = (reason: ChatFeedbackReason) => {
    setSelectedReasons((current) =>
      current.includes(reason)
        ? current.filter((entry) => entry !== reason)
        : [...current, reason]
    );
  };

  const commitSelection = (
    startWordIndex: number,
    endWordIndex: number,
    selectionExpanded = false
  ) => {
    const range = clampSelectionRange(startWordIndex, endWordIndex);
    const selection = buildSelectionFromWordRange(
      normalizedText(),
      tokens(),
      range.startWordIndex,
      range.endWordIndex
    );
    setFixSelection(selection);
    setReplacementText(selection?.originalText ?? "");

    if (selection) {
      recordFeedbackEvent({
        action: "selected",
        correctionKind: selection.correctionKind,
        originalText: selection.originalText,
        selectionEnd: selection.selectionEnd,
        selectionExpanded,
        selectionStart: selection.selectionStart,
        type: "fix_words",
      });
    }
  };

  const handleWordClick = (wordIndex: number) => {
    const selection = fixSelection();
    if (!selection) {
      commitSelection(wordIndex, wordIndex);
      return;
    }

    if (
      wordIndex >= selection.wordStartIndex &&
      wordIndex <= selection.wordEndIndex
    ) {
      if (selection.wordStartIndex === selection.wordEndIndex) {
        setFixSelection(null);
        setReplacementText("");
        return;
      }
      commitSelection(wordIndex, wordIndex);
      return;
    }

    if (wordIndex === selection.wordStartIndex - 1) {
      commitSelection(wordIndex, selection.wordEndIndex, true);
      return;
    }

    if (wordIndex === selection.wordEndIndex + 1) {
      commitSelection(selection.wordStartIndex, wordIndex, true);
      return;
    }

    commitSelection(wordIndex, wordIndex);
  };

  const saveFix = () => {
    const selection = fixSelection();
    const replacement = replacementText().replace(/\s+/g, " ").trim();
    if (!selection || !replacement) return;
    const correction = createCorrectionFromSelection(
      normalizedText(),
      selection,
      replacement
    );
    pushFeedback(
      {
        rating: feedback().rating || "needs_work",
        corrections: [...feedback().corrections, correction],
      },
      {
        action: "saved",
        correctionKind: selection.correctionKind,
        originalText: selection.originalText,
        replacementText: replacement,
        selectionEnd: selection.selectionEnd,
        selectionStart: selection.selectionStart,
        type: "fix_words",
      }
    );
    setFixSelection(null);
    setReplacementText("");
    setShowOriginal(false);
    showStatus("Fix saved");
  };

  const removeFix = (fix: ChatWordCorrection) => {
    pushFeedback(
      {
        corrections: feedback().corrections.filter((entry) => entry.id !== fix.id),
      },
      {
        action: "reverted",
        correctionKind: fix.correctionKind,
        originalText: fix.original,
        replacementText: fix.replacement,
        selectionEnd: fix.selectionEnd,
        selectionStart: fix.selectionStart,
        type: "fix_words",
      }
    );
    showStatus("Fix removed");
  };

  const isSelectedToken = (token: SelectableTextToken) => {
    const selection = fixSelection();
    return (
      selection &&
      token.kind === "word" &&
      token.wordIndex >= selection.wordStartIndex &&
      token.wordIndex <= selection.wordEndIndex
    );
  };

  const inSelectionRange = (token: SelectableTextToken) => {
    const selection = fixSelection();
    return (
      !!selection &&
      token.start < selection.selectionEnd &&
      token.end > selection.selectionStart
    );
  };

  const renderSelectableTokens = (): JSX.Element[] => {
    const selection = fixSelection();
    let editorInserted = false;

    return tokens().map((token, index) => {
      if (token.kind === "newline") {
        return <br />;
      }

      const tokenInsideSelection =
        selection &&
        token.start >= selection.selectionStart &&
        token.end <= selection.selectionEnd;

      if (tokenInsideSelection) {
        if (!editorInserted) {
          editorInserted = true;
          return (
            <span class="chat-word-editor">
              <input
                ref={(element) => {
                  editInputRef = element;
                }}
                value={replacementText()}
                onInput={(event) => setReplacementText(event.currentTarget.value)}
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    saveFix();
                  }
                  if (event.key === "Escape") {
                    event.preventDefault();
                    setFixSelection(null);
                    setReplacementText("");
                  }
                }}
                aria-label="Edit selected wording"
                style={{
                  width: `${Math.max(
                    4,
                    replacementText().length || selection.originalText.length
                  ) + 1}ch`,
                }}
              />
            </span>
          );
        }
        return <></>;
      }

      if (token.kind === "space" || token.kind === "punctuation") {
        return (
          <span class={inSelectionRange(token) ? "is-in-selection-range" : ""}>
            {token.value}
          </span>
        );
      }

      return (
        <button
          type="button"
          aria-pressed={!!isSelectedToken(token)}
          class={`chat-word-token ${isSelectedToken(token) ? "is-selected" : ""}`}
          onClick={() => handleWordClick(token.wordIndex)}
        >
          {token.value}
        </button>
      );
    });
  };

  return (
    <div class={`chat-message ${isUser() ? "chat-message--user" : "chat-message--assistant"} ${props.wide ? "chat-message--wide" : ""}`}>
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
            <span>{isUser() ? "You" : "Fenua AI"}</span>
            <Show when={timestamp()}>
              <time>{timestamp()}</time>
            </Show>
          </div>

          <Show
            when={!isUser()}
            fallback={<p class="chat-message__text">{props.message.content}</p>}
          >
            <AssistantMessageBody
              content={props.message.content}
              corrections={feedback().corrections}
              fixMode={activePanel() === "fix_words"}
              hasFeedback={hasChatMessageFeedback(feedback())}
              normalizedText={normalizedText()}
              originalRef={(element) => {
                contentRef = element;
              }}
              renderSelectableTokens={renderSelectableTokens}
              showOriginal={showOriginal()}
              onShowOriginalChange={setShowOriginal}
            />
          </Show>

          <div class="chat-message-actions chat-message-actions--refined" aria-label="Message actions">
            <Show when={isUser() && props.onEdit}>
              <button
                type="button"
                onClick={() => props.onEdit?.(props.message)}
              >
                Edit
              </button>
            </Show>
            <Show when={!isUser()}>
              <button
                type="button"
                class={feedback().rating === "good" ? "is-active" : ""}
                onClick={toggleHelpful}
              >
                👍 Helpful
              </button>
              <button
                type="button"
                class={activePanel() === "needs_work" || feedback().rating === "needs_work" ? "is-active" : ""}
                onClick={() => openPanel("needs_work")}
              >
                👎 Not right
              </button>
              <button
                type="button"
                class={activePanel() === "sounded_funny" || feedback().rating === "sounded_funny" ? "is-active" : ""}
                onClick={() => openPanel("sounded_funny")}
              >
                🌀 Sounds odd
              </button>
              <button
                type="button"
                class={activePanel() === "fix_words" ? "is-active" : ""}
                onClick={() => openPanel("fix_words")}
              >
                ✍️ Fix words
              </button>
              <button
                type="button"
                class={activePanel() === "say_more" || !!feedback().say_more ? "is-active" : ""}
                onClick={() => openPanel("say_more")}
              >
                💬 Say more
              </button>
              <button type="button" onClick={copyMessage}>
                {copied() ? "Copied" : "Copy"}
              </button>
              <Show when={props.canRegenerate}>
                <button type="button" onClick={() => props.onRegenerate?.()}>
                  Try again
                </button>
              </Show>
            </Show>
          </div>

          <Show when={saved()}>
            <div class="chat-feedback-saved" role="status" aria-live="polite">
              {saved()}
            </div>
          </Show>

          <Show when={!isUser() && activePanel()}>
            <FeedbackPanel
              activePanel={activePanel()!}
              corrections={feedback().corrections}
              noteDraft={noteDraft()}
              replacementText={replacementText()}
              sayMoreDraft={sayMoreDraft()}
              selectedReasons={selectedReasons()}
              selection={fixSelection()}
              onCancel={() => {
                recordFeedbackEvent({
                  action: fixSelection() ? "abandoned" : "dismissed",
                  correctionKind: fixSelection()?.correctionKind,
                  originalText: fixSelection()?.originalText,
                  selectionEnd: fixSelection()?.selectionEnd,
                  selectionStart: fixSelection()?.selectionStart,
                  type: panelToType(activePanel()),
                });
                setActivePanel(null);
                setFixSelection(null);
                setReplacementText("");
              }}
              onNoteChange={setNoteDraft}
              onReasonToggle={toggleReason}
              onRemoveFix={removeFix}
              onSaveFix={saveFix}
              onSaveReason={saveReasonFeedback}
              onSaveSayMore={saveSayMore}
              onSayMoreChange={setSayMoreDraft}
            />
          </Show>
        </div>
      </div>
    </div>
  );
}

function panelToType(panel: ActivePanel): ChatFeedbackType {
  if (panel === "fix_words") return "fix_words";
  if (panel === "say_more") return "say_more";
  return "not_right";
}

function AssistantMessageBody(props: {
  content: string;
  corrections: ChatWordCorrection[];
  fixMode: boolean;
  hasFeedback: boolean;
  normalizedText: string;
  originalRef: (element: HTMLDivElement) => void;
  renderSelectableTokens: () => JSX.Element[];
  showOriginal: boolean;
  onShowOriginalChange: (value: boolean) => void;
}) {
  return (
    <Show
      when={props.fixMode}
      fallback={
        <Show
          when={props.corrections.length > 0 && !props.showOriginal}
          fallback={
            <div
              ref={props.originalRef}
              class="markdown-content chat-message__markdown"
              innerHTML={renderMarkdown(props.content)}
            />
          }
        >
          <div class="chat-correction-toggle" aria-label="Correction view">
            <button
              type="button"
              class={!props.showOriginal ? "is-active" : ""}
              onClick={() => props.onShowOriginalChange(false)}
            >
              Edited
            </button>
            <button
              type="button"
              class={props.showOriginal ? "is-active" : ""}
              onClick={() => props.onShowOriginalChange(true)}
            >
              Original
            </button>
          </div>
          <CorrectedText text={props.normalizedText} corrections={props.corrections} />
        </Show>
      }
    >
      <div class="chat-fix-mode">
        <div class="chat-fix-mode__head">
          <strong>Tap a word. Tap neighbors to expand the phrase.</strong>
          <span>Press Enter to save the replacement.</span>
        </div>
        <div class="chat-word-selectable">{props.renderSelectableTokens()}</div>
      </div>
    </Show>
  );
}

function CorrectedText(props: {
  text: string;
  corrections: ChatWordCorrection[];
}) {
  const segments = createMemo(() =>
    applyCorrectionsToText(props.text, props.corrections)
  );
  return (
    <div class="chat-corrected-text">
      <For each={segments()}>
        {(segment) => (
          segment.kind === "replacement" ? (
            <span class="chat-corrected-text__replacement">
              {segment.replacement}
              <span>{segment.original}</span>
            </span>
          ) : (
            <span>{segment.text}</span>
          )
        )}
      </For>
    </div>
  );
}

function FeedbackPanel(props: {
  activePanel: Exclude<ActivePanel, null>;
  corrections: ChatWordCorrection[];
  noteDraft: string;
  replacementText: string;
  sayMoreDraft: string;
  selectedReasons: ChatFeedbackReason[];
  selection: FeedbackSelection | null;
  onCancel: () => void;
  onNoteChange: (value: string) => void;
  onReasonToggle: (reason: ChatFeedbackReason) => void;
  onRemoveFix: (fix: ChatWordCorrection) => void;
  onSaveFix: () => void;
  onSaveReason: () => void;
  onSaveSayMore: () => void;
  onSayMoreChange: (value: string) => void;
}) {
  const isNegative = () =>
    props.activePanel === "needs_work" || props.activePanel === "sounded_funny";
  return (
    <section class="chat-feedback-panel chat-feedback-panel--reference">
      <Show when={isNegative()}>
        <div class="chat-feedback-panel__head">
          <h3>Tell us what can be better</h3>
          <p>A short note is enough. Reasons help us group training data.</p>
        </div>
        <div class="chat-feedback-reasons">
          <For each={CHAT_FEEDBACK_REASON_OPTIONS}>
            {(reason) => (
              <button
                type="button"
                class={props.selectedReasons.includes(reason) ? "is-active" : ""}
                aria-pressed={props.selectedReasons.includes(reason)}
                onClick={() => props.onReasonToggle(reason)}
              >
                {reason}
              </button>
            )}
          </For>
        </div>
        <label class="chat-feedback-note">
          <span>Optional note</span>
          <textarea
            value={props.noteDraft}
            onInput={(event) => props.onNoteChange(event.currentTarget.value)}
            placeholder={
              props.activePanel === "sounded_funny"
                ? "What sounded odd?"
                : "What felt wrong?"
            }
            rows={3}
          />
        </label>
        <div class="chat-feedback-panel__actions">
          <button type="button" onClick={props.onSaveReason}>
            Save feedback
          </button>
          <button type="button" onClick={props.onCancel}>
            Done
          </button>
        </div>
      </Show>

      <Show when={props.activePanel === "say_more"}>
        <div class="chat-feedback-panel__head">
          <h3>What should Fenua say more about?</h3>
          <p>This captures missing context, explanation gaps, or dialect notes.</p>
        </div>
        <label class="chat-feedback-note">
          <span>Comment</span>
          <textarea
            value={props.sayMoreDraft}
            onInput={(event) => props.onSayMoreChange(event.currentTarget.value)}
            placeholder="Tell us what extra detail would have helped."
            rows={3}
          />
        </label>
        <div class="chat-feedback-panel__actions">
          <button type="button" onClick={props.onSaveSayMore}>
            Save note
          </button>
          <button type="button" onClick={props.onCancel}>
            Done
          </button>
        </div>
      </Show>

      <Show when={props.activePanel === "fix_words"}>
        <div class="chat-feedback-panel__head">
          <h3>Fix exact wording</h3>
          <p>
            Select a word or phrase above, type the better wording inline, then save.
          </p>
        </div>
        <Show
          when={props.selection}
          fallback={<p class="chat-feedback-help">No word selected yet.</p>}
        >
          {(selection) => (
            <div class="chat-fix-selection-summary">
              <span>Selected</span>
              <strong>{selection().originalText}</strong>
              <em>{selection().correctionKind === "phrase" ? "Phrase" : "Single word"}</em>
            </div>
          )}
        </Show>
        <div class="chat-feedback-panel__actions">
          <button
            type="button"
            onClick={props.onSaveFix}
            disabled={!props.selection || !props.replacementText.trim()}
          >
            Save wording fix
          </button>
          <button type="button" onClick={props.onCancel}>
            Done
          </button>
        </div>
      </Show>

      <Show when={props.corrections.length > 0}>
        <div class="chat-fix-list" aria-label="Saved wording fixes">
          <strong>Saved word fixes</strong>
          <For each={props.corrections}>
            {(fix) => (
              <div class="chat-fix-list__item">
                <span>{fix.original}</span>
                <em>→</em>
                <span>{fix.replacement}</span>
                <button type="button" onClick={() => props.onRemoveFix(fix)}>
                  Remove
                </button>
              </div>
            )}
          </For>
        </div>
      </Show>
    </section>
  );
}
