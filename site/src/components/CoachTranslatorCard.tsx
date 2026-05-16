import { createMemo, createSignal, For, Show } from "solid-js";
import type { LanguageMode } from "./LanguageToggle";
import { promptForIslandIfUnknown } from "./IslandSelector";
import { ensureCommunitySessionId, getKnownIsland } from "~/lib/community";

interface CoachTranslatorCardProps {
  articleId: string;
  paragraphCount: number;
  initialMode: LanguageMode;
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
      const island = getKnownIsland();
      const sessionId = ensureCommunitySessionId();
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
      if (!island) {
        void promptForIslandIfUnknown();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save your coaching note.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section class="coach-card">
      <div class="coach-card__head">
        <div>
          <p class="site-kicker">
            Kominiti
          </p>
          <h2>Coach the Translator</h2>
          <p class="coach-card__intro">
            Add one real community signal from this football story. Vote on the
            translation, tell us which reading mode worked best, and leave a
            better Tuvaluan phrasing if you spot one.
          </p>
        </div>
        <div class="coach-card__badge">
          +1 signal
        </div>
      </div>

      <Show
        when={!submitted()}
        fallback={
          <div class="coach-card__success">
            <p>Malo!</p>
            <p class="mt-1">
              Your coaching note was saved. This article now contributes
              structured feedback for translation review and future tuning.
            </p>
          </div>
        }
      >
        <>
          <div class="coach-card__group">
            <p class="coach-card__label">
              Was this translation helpful?
            </p>
            <div class="coach-card__options">
              <button
                type="button"
                onClick={() => setHelpfulScore(1)}
                class={`coach-option ${
                  helpfulScore() === 1
                    ? "coach-option--active"
                    : ""
                }`}
              >
                Yes, keep this style
              </button>
              <button
                type="button"
                onClick={() => setHelpfulScore(0)}
                class={`coach-option ${
                  helpfulScore() === 0
                    ? "coach-option--active"
                    : ""
                }`}
              >
                Needs work
              </button>
            </div>
          </div>

          <div class="coach-card__group">
            <p class="coach-card__label">
              Which reading mode helped most?
            </p>
            <div class="coach-card__options">
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
                    class={`coach-option ${
                      modePreference() === option.value
                        ? "coach-option--active"
                        : ""
                    }`}
                  >
                    {option.label}
                  </button>
                )}
              </For>
            </div>
          </div>

          <div class="coach-card__correction-grid">
            <div>
              <label class="coach-card__label" for="coach-paragraph">
                Paragraph to improve
              </label>
              <select
                id="coach-paragraph"
                value={correctionParagraphIdx()}
                onInput={(e) => setCorrectionParagraphIdx(e.currentTarget.value)}
                class="coach-field"
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
              <label class="coach-card__label" for="coach-correction">
                Better Tuvaluan phrasing
              </label>
              <textarea
                id="coach-correction"
                value={correctionText()}
                onInput={(e) => setCorrectionText(e.currentTarget.value)}
                rows={4}
                maxLength={1000}
                placeholder="Optional. Paste a better translation, wording, or name fix."
                class="coach-field coach-field--textarea"
              />
            </div>
          </div>

          <Show when={error()}>
            {(message) => (
              <p class="coach-card__error">{message()}</p>
            )}
          </Show>

          <div class="coach-card__footer">
            <p class="coach-card__note">
              Anonymous browser session only. We group your feedback, island,
              and optional correction note under one session so they can be
              exported later for preference tuning.
            </p>
            <button
              type="button"
              disabled={submitting()}
              onClick={submit}
              class="site-button site-button--gold disabled:cursor-not-allowed disabled:opacity-60"
            >
              {submitting() ? "Saving..." : "Save coaching note"}
            </button>
          </div>
        </>
      </Show>
    </section>
  );
}
