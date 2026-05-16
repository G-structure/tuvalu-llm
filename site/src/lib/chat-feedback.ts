export const CHAT_FEEDBACK_REASON_OPTIONS = [
  "Wrong word",
  "Wrong phrase",
  "Wrong meaning",
  "Sounds strange",
  "Too formal",
  "Mixed languages",
  "Missing something",
  "Different dialect",
  "Not what I meant",
  "Other",
] as const;

export type ChatFeedbackReason = (typeof CHAT_FEEDBACK_REASON_OPTIONS)[number];
export type ChatFeedbackRating = "good" | "needs_work" | "sounded_funny" | null;
export type ChatFeedbackType =
  | "helpful"
  | "not_right"
  | "fix_words"
  | "say_more"
  | "phrase_comment";
export type ChatFeedbackAction =
  | "opened"
  | "selected"
  | "saved"
  | "dismissed"
  | "abandoned"
  | "reverted";
export type CorrectionKind = "single_word" | "phrase";

export interface FeedbackSelection {
  correctionKind: CorrectionKind;
  originalText: string;
  selectedWords: string[];
  selectionEnd: number;
  selectionStart: number;
  wordEndIndex: number;
  wordStartIndex: number;
}

export interface ChatWordCorrection {
  id: string;
  original: string;
  replacement: string;
  selectionStart: number;
  selectionEnd: number;
  correctionKind: CorrectionKind;
  selectedWords: string[];
  contextBefore: string;
  contextAfter: string;
  created_at: string;
}

export interface ChatFeedbackEvent {
  id: string;
  type: ChatFeedbackType;
  action: ChatFeedbackAction;
  selectedReasons: ChatFeedbackReason[];
  freeformComment?: string;
  originalText?: string;
  selectionStart?: number;
  selectionEnd?: number;
  replacementText?: string;
  correctionKind?: CorrectionKind;
  selectionExpanded?: boolean;
  created_at: string;
}

export interface ChatMessageFeedback {
  rating: ChatFeedbackRating;
  reasons: ChatFeedbackReason[];
  note: string;
  say_more: string;
  corrections: ChatWordCorrection[];
  events: ChatFeedbackEvent[];
}

export interface NormalizedFeedbackText {
  didSimplify: boolean;
  omittedReadOnlyContent: boolean;
  text: string;
}

interface BaseSelectableToken {
  end: number;
  start: number;
  value: string;
}

export type SelectableTextToken =
  | (BaseSelectableToken & { kind: "newline" })
  | (BaseSelectableToken & { kind: "punctuation" })
  | (BaseSelectableToken & { kind: "space" })
  | (BaseSelectableToken & { kind: "word"; wordIndex: number });

const CHAT_FEEDBACK_REASON_SET = new Set<string>(CHAT_FEEDBACK_REASON_OPTIONS);

function cleanString(value: unknown, max: number): string {
  return typeof value === "string" ? value.trim().slice(0, max) : "";
}

function cleanOptionalString(value: unknown, max: number): string | undefined {
  const cleaned = cleanString(value, max);
  return cleaned || undefined;
}

export function newFeedbackId(prefix = "feedback"): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}_${crypto.randomUUID()}`;
  }
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

export function createEmptyChatMessageFeedback(): ChatMessageFeedback {
  return {
    rating: null,
    reasons: [],
    note: "",
    say_more: "",
    corrections: [],
    events: [],
  };
}

export function normalizeChatMessageFeedback(
  feedback?: Partial<ChatMessageFeedback> | null
): ChatMessageFeedback {
  return {
    rating:
      feedback?.rating === "good" ||
      feedback?.rating === "needs_work" ||
      feedback?.rating === "sounded_funny"
        ? feedback.rating
        : null,
    reasons: Array.isArray(feedback?.reasons)
      ? feedback.reasons.filter(
          (reason): reason is ChatFeedbackReason =>
            typeof reason === "string" && CHAT_FEEDBACK_REASON_SET.has(reason)
        )
      : [],
    note: cleanString(feedback?.note, 1200),
    say_more: cleanString(feedback?.say_more, 1200),
    corrections: Array.isArray(feedback?.corrections)
      ? feedback.corrections
          .filter(
            (correction): correction is ChatWordCorrection =>
              correction &&
              typeof correction.id === "string" &&
              typeof correction.original === "string" &&
              typeof correction.replacement === "string" &&
              typeof correction.selectionStart === "number" &&
              typeof correction.selectionEnd === "number" &&
              (correction.correctionKind === "single_word" ||
                correction.correctionKind === "phrase")
          )
          .slice(-40)
          .map((correction) => ({
            id: cleanString(correction.id, 120),
            original: cleanString(correction.original, 600),
            replacement: cleanString(correction.replacement, 600),
            selectionStart: correction.selectionStart,
            selectionEnd: correction.selectionEnd,
            correctionKind: correction.correctionKind,
            selectedWords: Array.isArray(correction.selectedWords)
              ? correction.selectedWords
                  .filter((word) => typeof word === "string")
                  .map((word) => word.slice(0, 120))
                  .slice(0, 40)
              : [],
            contextBefore: cleanString(correction.contextBefore, 240),
            contextAfter: cleanString(correction.contextAfter, 240),
            created_at: cleanString(correction.created_at, 80),
          }))
      : [],
    events: Array.isArray(feedback?.events)
      ? feedback.events
          .filter(
            (event): event is ChatFeedbackEvent =>
              event &&
              typeof event.id === "string" &&
              typeof event.type === "string" &&
              typeof event.action === "string"
          )
          .slice(-120)
          .map((event) => ({
            id: cleanString(event.id, 120),
            type: event.type,
            action: event.action,
            selectedReasons: Array.isArray(event.selectedReasons)
              ? event.selectedReasons.filter(
                  (reason): reason is ChatFeedbackReason =>
                    typeof reason === "string" &&
                    CHAT_FEEDBACK_REASON_SET.has(reason)
                )
              : [],
            freeformComment: cleanOptionalString(event.freeformComment, 1200),
            originalText: cleanOptionalString(event.originalText, 600),
            selectionStart:
              typeof event.selectionStart === "number"
                ? event.selectionStart
                : undefined,
            selectionEnd:
              typeof event.selectionEnd === "number"
                ? event.selectionEnd
                : undefined,
            replacementText: cleanOptionalString(event.replacementText, 600),
            correctionKind:
              event.correctionKind === "single_word" ||
              event.correctionKind === "phrase"
                ? event.correctionKind
                : undefined,
            selectionExpanded: event.selectionExpanded === true,
            created_at: cleanString(event.created_at, 80),
          }))
      : [],
  };
}

export function hasChatMessageFeedback(
  feedback?: Partial<ChatMessageFeedback> | null
): boolean {
  const normalized = normalizeChatMessageFeedback(feedback);
  return Boolean(
    normalized.rating ||
      normalized.reasons.length ||
      normalized.note ||
      normalized.say_more ||
      normalized.corrections.length ||
      normalized.events.length
  );
}

export function createFeedbackEvent(input: {
  action: ChatFeedbackAction;
  correctionKind?: CorrectionKind;
  freeformComment?: string;
  originalText?: string;
  replacementText?: string;
  selectedReasons?: ChatFeedbackReason[];
  selectionEnd?: number;
  selectionExpanded?: boolean;
  selectionStart?: number;
  type: ChatFeedbackType;
}): ChatFeedbackEvent {
  return {
    action: input.action,
    correctionKind: input.correctionKind,
    created_at: new Date().toISOString(),
    freeformComment: cleanOptionalString(input.freeformComment, 1200),
    id: newFeedbackId("event"),
    originalText: cleanOptionalString(input.originalText, 600),
    replacementText: cleanOptionalString(input.replacementText, 600),
    selectedReasons: input.selectedReasons ?? [],
    selectionEnd: input.selectionEnd,
    selectionExpanded: input.selectionExpanded,
    selectionStart: input.selectionStart,
    type: input.type,
  };
}

export function normalizeMessageForFeedback(content: string): NormalizedFeedbackText {
  let text = content.replace(/\r\n/g, "\n");
  let didSimplify = false;
  let omittedReadOnlyContent = false;

  text = text.replace(/```[\s\S]*?```/g, () => {
    didSimplify = true;
    omittedReadOnlyContent = true;
    return "\n";
  });

  const replacements: Array<[RegExp, string]> = [
    [/`([^`]+)`/g, "$1"],
    [/!\[([^\]]*)\]\([^)]+\)/g, "$1"],
    [/\[([^\]]+)\]\([^)]+\)/g, "$1"],
    [/^#{1,6}\s+/gm, ""],
    [/^>\s?/gm, ""],
    [/^\s*[-*+]\s+/gm, ""],
    [/^\s*\d+\.\s+/gm, ""],
    [/^\|?(?:\s*:?-{3,}:?\s*\|)+\s*$/gm, ""],
    [/\|/g, "  "],
    [/\*\*|__|~~/g, ""],
    [/(^|[^\*])\*([^\*]+)\*(?!\*)/g, "$1$2"],
    [/(^|[^_])_([^_]+)_(?!_)/g, "$1$2"],
  ];

  for (const [pattern, replacement] of replacements) {
    const nextText = text.replace(pattern, replacement);
    if (nextText !== text) {
      didSimplify = true;
      text = nextText;
    }
  }

  text = text.replace(/\n{3,}/g, "\n\n").trim();

  if (!text) {
    return {
      didSimplify,
      omittedReadOnlyContent,
      text: content.replace(/\r\n/g, "\n").trim(),
    };
  }

  return { didSimplify, omittedReadOnlyContent, text };
}

export function tokenizeSelectableText(text: string): SelectableTextToken[] {
  const tokens: SelectableTextToken[] = [];
  const pattern =
    /(\r?\n)|([^\S\r\n]+)|([\p{L}\p{M}\p{N}]+(?:['_-][\p{L}\p{M}\p{N}]+)*)|([^\s])/gu;
  let wordIndex = 0;

  for (const match of text.matchAll(pattern)) {
    const value = match[0];
    const start = match.index ?? 0;
    const end = start + value.length;

    if (match[1]) {
      tokens.push({ end, kind: "newline", start, value });
    } else if (match[2]) {
      tokens.push({ end, kind: "space", start, value });
    } else if (match[3]) {
      tokens.push({ end, kind: "word", start, value, wordIndex });
      wordIndex += 1;
    } else {
      tokens.push({ end, kind: "punctuation", start, value });
    }
  }

  return tokens;
}

export function buildSelectionFromWordRange(
  text: string,
  tokens: SelectableTextToken[],
  startWordIndex: number,
  endWordIndex: number
): FeedbackSelection | null {
  const wordTokens = tokens.filter(
    (token): token is SelectableTextToken & { kind: "word"; wordIndex: number } =>
      token.kind === "word" && typeof token.wordIndex === "number"
  );
  const startToken = wordTokens[startWordIndex];
  const endToken = wordTokens[endWordIndex];

  if (!startToken || !endToken) return null;

  const orderedStart = Math.min(startWordIndex, endWordIndex);
  const orderedEnd = Math.max(startWordIndex, endWordIndex);
  const selectionStart = Math.min(startToken.start, endToken.start);
  const selectionEnd = Math.max(startToken.end, endToken.end);

  return {
    correctionKind: orderedStart === orderedEnd ? "single_word" : "phrase",
    originalText: text.slice(selectionStart, selectionEnd),
    selectedWords: wordTokens
      .slice(orderedStart, orderedEnd + 1)
      .map((token) => token.value),
    selectionEnd,
    selectionStart,
    wordEndIndex: orderedEnd,
    wordStartIndex: orderedStart,
  };
}

export function clampSelectionRange(
  anchorIndex: number,
  nextIndex: number,
  maxWords = 24
) {
  if (anchorIndex === nextIndex) {
    return { endWordIndex: nextIndex, startWordIndex: anchorIndex };
  }

  const direction = nextIndex > anchorIndex ? 1 : -1;
  const distance = Math.abs(nextIndex - anchorIndex);

  if (distance + 1 <= maxWords) {
    return {
      endWordIndex: Math.max(anchorIndex, nextIndex),
      startWordIndex: Math.min(anchorIndex, nextIndex),
    };
  }

  const limitedIndex = anchorIndex + direction * (maxWords - 1);
  return {
    endWordIndex: Math.max(anchorIndex, limitedIndex),
    startWordIndex: Math.min(anchorIndex, limitedIndex),
  };
}

export function createCorrectionFromSelection(
  text: string,
  selection: FeedbackSelection,
  replacement: string
): ChatWordCorrection {
  return {
    id: newFeedbackId("fix"),
    original: selection.originalText,
    replacement: replacement.replace(/\s+/g, " ").trim(),
    selectionStart: selection.selectionStart,
    selectionEnd: selection.selectionEnd,
    correctionKind: selection.correctionKind,
    selectedWords: selection.selectedWords,
    contextBefore: text.slice(Math.max(0, selection.selectionStart - 90), selection.selectionStart),
    contextAfter: text.slice(selection.selectionEnd, selection.selectionEnd + 90),
    created_at: new Date().toISOString(),
  };
}

export function applyCorrectionsToText(
  text: string,
  corrections: ChatWordCorrection[]
) {
  const ordered = [...corrections]
    .filter(
      (correction) =>
        correction.selectionStart >= 0 &&
        correction.selectionEnd > correction.selectionStart &&
        correction.selectionEnd <= text.length
    )
    .sort((a, b) => a.selectionStart - b.selectionStart);
  const segments: Array<
    | { kind: "text"; text: string }
    | { kind: "replacement"; original: string; replacement: string }
  > = [];
  let cursor = 0;

  for (const correction of ordered) {
    if (correction.selectionStart < cursor) continue;
    if (correction.selectionStart > cursor) {
      segments.push({ kind: "text", text: text.slice(cursor, correction.selectionStart) });
    }
    segments.push({
      kind: "replacement",
      original: text.slice(correction.selectionStart, correction.selectionEnd),
      replacement: correction.replacement,
    });
    cursor = correction.selectionEnd;
  }

  if (cursor < text.length) {
    segments.push({ kind: "text", text: text.slice(cursor) });
  }

  return segments;
}
