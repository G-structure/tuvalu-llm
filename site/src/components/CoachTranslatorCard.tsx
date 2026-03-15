import { createMemo, createSignal, For, Show } from "solid-js";
import type { LanguageMode } from "./LanguageToggle";

interface CoachTranslatorCardProps {
  articleId: string;
  paragraphCount: number;
  initialMode: LanguageMode;
}

function ensureSessionId(): string {
  const existing = localStorage.getItem("talafutipolo_session");
  if (existing) return existing;
  const next = crypto.randomUUID();
  localStorage.setItem("talafutipolo_session", next);
  return next;
}

export default function CoachTranslatorCard(props: CoachTranslatorCardProps) {
  const [helpfulScore, setHelpfulScore] = createSignal<0 | 1 | null>(null);
  const [modePreference, setModePreference] = createSignal<LanguageMode>(
    props.initialMode === "tv" || props.initialMode === "tv+en" || props.initialMode === "en"
      ? props.initialMode
      : "tv"
  );
  const [correctionParagraphIdx, setCorrectionParagraphIdx] = createSignal<string>("");
  const [correctionText, setCorrectionText] = createSignal("");
  const [submitting, setSubmitting] = createSignal(false);
  const [submitted, setSubmitted] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  const paragraphs = createMemo(() =>
    Array.from({ length: props.paragraphCount }, (_, i) => ({
      idx: i,
      label: `Paragraph ${i + 1}`,
    }))
  );

  const submit = async () => {
    if (submitted() || submitting()) return;
    if (helpfulScore() === null) {
      setError("Pick whether the translation helped first.");
      return;
    }

    const trimmedCorrection = correctionText().trim();
    if (trimmedCorrection && correctionParagraphIdx() === "") {
      setError("Pick the paragraph you want to improve.");
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const island = localStorage.getItem("talafutipolo_island");
      const sessionId = ensureSessionId();
      const response = await fetch("/api/article-feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          article_id: props.articleId,
          helpful_score: helpfulScore(),
          mode_preference: modePreference(),
          correction_paragraph_idx:
            trimmedCorrection && correctionParagraphIdx() !== ""
              ? parseInt(correctionParagraphIdx(), 10)
              : undefined,
          correction_text: trimmedCorrection || undefined,
          island,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.error || "Could not save your coaching note.");
      }

      setSubmitted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save your coaching note.");
    } finally {
      setSubmitting(false);
    }
  };

  const islandLabel = () => localStorage.getItem("talafutipolo_island") || "your island";

  return (
    <section class="mt-8 rounded-2xl border border-[var(--gold)]/40 bg-[var(--ocean-deep)] text-white p-5">
      <div class="flex items-start justify-between gap-4">
        <div>
          <p class="text-xs uppercase tracking-[0.2em] text-[var(--gold)]">
            Te Fatele
          </p>
          <h2 class="mt-1 text-xl font-bold">Coach the Translator</h2>
          <p class="mt-2 text-sm leading-relaxed text-[var(--sky-dark)]">
            Add one real community signal from this football story. Vote on the
            translation, tell us which reading mode worked best, and leave a
            better Tuvaluan phrasing if you spot one.
          </p>
        </div>
        <div class="rounded-full bg-white/10 px-3 py-1 text-xs text-[var(--gold)]">
          +1 signal
        </div>
      </div>

      <Show
        when={!submitted()}
        fallback={
          <div class="mt-4 rounded-xl bg-white/10 p-4 text-sm text-[var(--sky-dark)]">
            <p class="font-medium text-white">Malo!</p>
            <p class="mt-1">
              Your coaching note was saved for {islandLabel()}. This article now
              contributes structured feedback for future preference tuning and
              correction review.
            </p>
          </div>
        }
      >
        <>
          <div class="mt-5">
            <p class="text-sm font-medium text-white">
              Was this translation helpful?
            </p>
            <div class="mt-2 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setHelpfulScore(1)}
                class={`rounded-full px-4 py-2 text-sm font-medium transition-colors cursor-pointer border ${
                  helpfulScore() === 1
                    ? "border-[var(--gold)] bg-[var(--gold)] text-[var(--ocean-deep)]"
                    : "border-white/20 bg-white/10 text-white hover:bg-white/15"
                }`}
              >
                Yes, keep this style
              </button>
              <button
                type="button"
                onClick={() => setHelpfulScore(0)}
                class={`rounded-full px-4 py-2 text-sm font-medium transition-colors cursor-pointer border ${
                  helpfulScore() === 0
                    ? "border-[var(--gold)] bg-[var(--gold)] text-[var(--ocean-deep)]"
                    : "border-white/20 bg-white/10 text-white hover:bg-white/15"
                }`}
              >
                Needs work
              </button>
            </div>
          </div>

          <div class="mt-5">
            <p class="text-sm font-medium text-white">
              Which reading mode helped most?
            </p>
            <div class="mt-2 flex flex-wrap gap-2">
              <For
                each={[
                  { value: "tv", label: "TV" },
                  { value: "tv+en", label: "TV + EN" },
                  { value: "en", label: "EN" },
                ] as const}
              >
                {(option) => (
                  <button
                    type="button"
                    onClick={() => setModePreference(option.value)}
                    class={`rounded-full px-4 py-2 text-sm font-medium transition-colors cursor-pointer border ${
                      modePreference() === option.value
                        ? "border-[var(--gold)] bg-[var(--gold)] text-[var(--ocean-deep)]"
                        : "border-white/20 bg-white/10 text-white hover:bg-white/15"
                    }`}
                  >
                    {option.label}
                  </button>
                )}
              </For>
            </div>
          </div>

          <div class="mt-5 grid gap-3 sm:grid-cols-[200px_1fr]">
            <div>
              <label class="block text-sm font-medium text-white" for="coach-paragraph">
                Paragraph to improve
              </label>
              <select
                id="coach-paragraph"
                value={correctionParagraphIdx()}
                onInput={(e) => setCorrectionParagraphIdx(e.currentTarget.value)}
                class="mt-2 w-full rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-sm text-white"
              >
                <option value="">Optional</option>
                <For each={paragraphs()}>
                  {(item) => (
                    <option value={String(item.idx)} class="text-black">
                      {item.label}
                    </option>
                  )}
                </For>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium text-white" for="coach-correction">
                Better Tuvaluan phrasing
              </label>
              <textarea
                id="coach-correction"
                value={correctionText()}
                onInput={(e) => setCorrectionText(e.currentTarget.value)}
                rows={4}
                maxLength={1000}
                placeholder="Optional. Paste a better translation, wording, or name fix."
                class="mt-2 w-full rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-sm text-white placeholder:text-[var(--sky-dark)]"
              />
            </div>
          </div>

          <Show when={error()}>
            {(message) => (
              <p class="mt-3 text-sm text-[#ffd6d6]">{message()}</p>
            )}
          </Show>

          <div class="mt-5 flex flex-wrap items-center justify-between gap-3">
            <p class="text-xs text-[var(--sky-dark)]">
              Anonymous session only. We store your island, feedback choices,
              and optional correction note so they can be exported later for
              preference tuning.
            </p>
            <button
              type="button"
              disabled={submitting()}
              onClick={submit}
              class="rounded-xl bg-[var(--gold)] px-4 py-2 text-sm font-semibold text-[var(--ocean-deep)] transition-colors hover:bg-[#f7d55e] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {submitting() ? "Saving..." : "Save coaching note"}
            </button>
          </div>
        </>
      </Show>
    </section>
  );
}
