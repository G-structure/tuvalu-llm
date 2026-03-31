import { createSignal, Show } from "solid-js";
import { ISLANDS } from "~/lib/types";

function generateUUID(): string {
  return crypto.randomUUID();
}

const [show, setShow] = createSignal(false);
let resolvePromise: (() => void) | null = null;

/**
 * Call before submitting any feedback. If the user hasn't chosen an island yet,
 * shows the modal and waits for them to pick/skip. Resolves immediately if
 * already chosen. Always ensures a session ID exists.
 */
export function ensureIslandChosen(): Promise<void> {
  if (!localStorage.getItem("talafutipolo_session")) {
    localStorage.setItem("talafutipolo_session", generateUUID());
  }
  if (localStorage.getItem("talafutipolo_island_chosen")) {
    return Promise.resolve();
  }
  return new Promise<void>((resolve) => {
    resolvePromise = resolve;
    setShow(true);
  });
}

export default function IslandSelector() {
  const selectIsland = (island: string) => {
    localStorage.setItem("talafutipolo_island", island);
    if (!localStorage.getItem("talafutipolo_session")) {
      localStorage.setItem("talafutipolo_session", generateUUID());
    }
    localStorage.setItem("talafutipolo_island_chosen", "true");
    setShow(false);
    resolvePromise?.();
    resolvePromise = null;
  };

  const skip = () => {
    if (!localStorage.getItem("talafutipolo_session")) {
      localStorage.setItem("talafutipolo_session", generateUUID());
    }
    localStorage.setItem("talafutipolo_island_chosen", "true");
    setShow(false);
    resolvePromise?.();
    resolvePromise = null;
  };

  return (
    <Show when={show()}>
      <div class="fixed inset-0 z-50 bg-[var(--ocean-deep)] flex items-center justify-center p-6">
        <div class="w-full max-w-sm text-center">
          <h2 class="text-2xl font-bold text-[var(--gold)] mb-2">Talofa!</h2>
          <p class="text-[var(--sky-dark)] mb-6 text-sm">
            Ko koe mai fea? Where are you from?
          </p>

          <div class="grid grid-cols-2 gap-3 mb-4">
            {ISLANDS.slice(0, 9).map((island) => (
              <button
                onClick={() => selectIsland(island)}
                class="py-3 px-4 bg-white/10 text-white rounded-lg text-sm font-medium hover:bg-[var(--gold)]/20 transition-colors cursor-pointer border border-[var(--gold)]/30"
              >
                {island}
              </button>
            ))}
          </div>

          {/* I fafo (Diaspora) full width */}
          <button
            onClick={() => selectIsland("I fafo")}
            class="w-full py-3 px-4 bg-white/10 text-white rounded-lg text-sm font-medium hover:bg-[var(--gold)]/20 transition-colors cursor-pointer border border-[var(--gold)]/30 mb-4"
          >
            I fafo (Diaspora)
          </button>

          <button
            onClick={skip}
            class="text-[var(--sky-dark)] text-sm hover:text-[var(--gold)] cursor-pointer bg-transparent border-none"
          >
            Fano &rarr;
          </button>
        </div>
      </div>
    </Show>
  );
}
