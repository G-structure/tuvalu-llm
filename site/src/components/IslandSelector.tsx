import { Show, createSignal, onCleanup, onMount } from "solid-js";
import { ensureCommunitySessionId, getKnownIsland, setKnownIsland } from "~/lib/community";
import { ISLANDS } from "~/lib/types";

const [show, setShow] = createSignal(false);
const [savingIsland, setSavingIsland] = createSignal<string | null>(null);
let resolvePromise: (() => void) | null = null;
let pendingPrompt: Promise<void> | null = null;

/**
 * Call after a feedback write succeeds. If we still do not know the submitter's
 * island, show the dialog and resolve when the user picks or dismisses it.
 */
export function promptForIslandIfUnknown(): Promise<void> {
  if (getKnownIsland()) {
    return Promise.resolve();
  }
  ensureCommunitySessionId();
  if (pendingPrompt) {
    return pendingPrompt;
  }
  pendingPrompt = new Promise<void>((resolve) => {
    resolvePromise = resolve;
    setShow(true);
  });
  return pendingPrompt;
}

export default function IslandSelector() {
  const finishPrompt = () => {
    setShow(false);
    resolvePromise?.();
    resolvePromise = null;
    pendingPrompt = null;
  };

  const syncIslandToSession = async (island: string, sessionId: string) => {
    setSavingIsland(island);
    try {
      const response = await fetch("/api/island", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          island,
          session_id: sessionId,
        }),
      });
      if (!response.ok) {
        throw new Error("Could not sync island");
      }
    } catch {
      // Keep the island locally even if the network write misses; the next
      // feedback submission will still carry the selected origin.
    } finally {
      setSavingIsland(null);
    }
  };

  const selectIsland = async (island: string) => {
    const sessionId = ensureCommunitySessionId();
    setKnownIsland(island);
    await syncIslandToSession(island, sessionId);
    finishPrompt();
  };

  const skip = () => {
    ensureCommunitySessionId();
    finishPrompt();
  };

  onMount(() => {
    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && show() && !savingIsland()) {
        event.preventDefault();
        skip();
      }
    };
    window.addEventListener("keydown", handleKeydown);
    onCleanup(() => window.removeEventListener("keydown", handleKeydown));
  });

  return (
    <Show when={show()}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="island-dialog-title"
        class="fixed inset-0 z-50 bg-[var(--ocean-deep)] flex items-center justify-center p-6"
      >
        <div class="w-full max-w-sm text-center mx-auto">
          <h2 id="island-dialog-title" class="text-2xl font-bold text-[var(--gold)] mb-2">Talofa!</h2>
          <p class="text-[var(--sky-dark)] mb-6 text-sm">
            Ko koe mai fea? Where are you from?
          </p>

          <div class="grid grid-cols-2 gap-3 mb-4">
            {ISLANDS.slice(0, 9).map((island) => (
              <button
                type="button"
                onClick={() => selectIsland(island)}
                disabled={savingIsland() !== null}
                class="py-3 px-4 bg-white/10 text-white rounded-lg text-sm font-medium hover:bg-[var(--gold)]/20 transition-colors cursor-pointer border border-[var(--gold)]/30"
              >
                {island}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => selectIsland("I fafo")}
            disabled={savingIsland() !== null}
            class="w-full py-3 px-4 bg-white/10 text-white rounded-lg text-sm font-medium hover:bg-[var(--gold)]/20 transition-colors cursor-pointer border border-[var(--gold)]/30 mb-4"
          >
            I fafo (Diaspora)
          </button>

          <button
            type="button"
            onClick={skip}
            disabled={savingIsland() !== null}
            class="text-[var(--sky-dark)] text-sm hover:text-[var(--gold)] cursor-pointer bg-transparent border-none"
          >
            {savingIsland() ? "Saving..." : "Fano →"}
          </button>
        </div>
      </div>
    </Show>
  );
}
