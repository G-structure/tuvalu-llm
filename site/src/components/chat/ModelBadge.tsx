import { createSignal, onCleanup, onMount, Show } from "solid-js";
import { isServer } from "solid-js/web";

interface ModelInfo {
  sampler_path: string;
  step: string;
  run: string;
  status: string;
  latest_train_step: number;
  latest_train_nll: number | null;
  latest_val_nll: number | null;
}

const OFFLINE_MODEL_INFO: ModelInfo = {
  sampler_path: "",
  step: "0",
  run: "stage_b_llama8b",
  status: "offline",
  latest_train_step: 0,
  latest_train_nll: null,
  latest_val_nll: null,
};

async function fetchModelInfo(): Promise<ModelInfo | undefined> {
  if (isServer) return undefined;
  try {
    const resp = await fetch("/api/model-info");
    if (!resp.ok) return OFFLINE_MODEL_INFO;
    return resp.json();
  } catch {
    return OFFLINE_MODEL_INFO;
  }
}

export default function ModelBadge() {
  const [info, setInfo] = createSignal<ModelInfo | undefined>();
  const [loading, setLoading] = createSignal(true);
  const [expanded, setExpanded] = createSignal(false);

  let timer: ReturnType<typeof setInterval> | undefined;
  const refresh = async () => {
    const next = await fetchModelInfo();
    setInfo(next || OFFLINE_MODEL_INFO);
    setLoading(false);
  };

  const shouldPoll = () => {
    const connection = (navigator as any).connection;
    return (
      !connection?.saveData &&
      !["slow-2g", "2g"].includes(connection?.effectiveType)
    );
  };

  onMount(() => {
    void refresh();
    if (shouldPoll()) {
      timer = setInterval(() => void refresh(), 60000);
    }
  });
  onCleanup(() => {
    if (timer) clearInterval(timer);
  });

  const isOffline = () => info()?.status === "offline";

  return (
    <div class="relative">
      <button
        type="button"
        onClick={() => setExpanded(!expanded())}
        onKeyDown={(e) => { if (e.key === "Escape" && expanded()) { setExpanded(false); e.stopPropagation(); } }}
        aria-expanded={expanded()}
        aria-label="Model training status"
        class="flex items-center gap-2 text-[11px] text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
      >
        {loading() && <span>Connecting...</span>}
        {info() && (
          <>
            <span
              class={`inline-block w-1.5 h-1.5 rounded-full ${
                isOffline() ? "bg-[#f87171]" : "bg-[var(--color-accent)]"
              }`}
            />
            <span class="tabular">
              {isOffline()
                ? "Model offline"
                : `Step ${info()!.latest_train_step.toLocaleString()} · ${info()!.status}`}
            </span>
            <svg
              class={`w-2.5 h-2.5 transition-transform ${expanded() ? "rotate-180" : ""}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </>
        )}
      </button>

      <Show when={expanded() && info()}>
        <div class="absolute top-7 left-0 z-50 bg-[var(--color-surface-2)] border border-[var(--color-border)] rounded-lg p-4 shadow-xl shadow-black/40 min-w-[260px]">
          <div class="text-[12px] font-medium text-[var(--color-text)] mb-3">Training</div>
          <div class="space-y-2 text-[11px]">
            <Row label="Run" value={info()!.run || "—"} />
            <Row label="Status" value={info()!.status || "—"} />
            <Show when={!isOffline()}>
              <Row label="Step" value={info()!.latest_train_step.toLocaleString()} />
            </Show>
            <Show when={info()!.latest_train_nll !== null}>
              <Row label="Train NLL" value={info()!.latest_train_nll!.toFixed(4)} />
            </Show>
            <Show when={info()!.latest_val_nll !== null}>
              <Row label="Val NLL" value={info()!.latest_val_nll!.toFixed(4)} />
            </Show>
            <div class="pt-2 border-t border-[var(--color-border)]">
              <a
                href="/chat/training"
                class="text-[var(--color-accent)] hover:underline text-[11px]"
              >
                View dashboard →
              </a>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}

function Row(props: { label: string; value: string }) {
  return (
    <div class="flex justify-between">
      <span class="text-[var(--color-text-muted)]">{props.label}</span>
      <span class="text-[var(--color-text-secondary)] tabular">{props.value}</span>
    </div>
  );
}
