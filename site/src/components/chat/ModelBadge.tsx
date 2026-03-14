import { createResource, createSignal, Show } from "solid-js";
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

async function fetchModelInfo(): Promise<ModelInfo | undefined> {
  if (isServer) return undefined;
  const resp = await fetch("/api/model-info");
  if (!resp.ok) throw new Error("Backend unavailable");
  return resp.json();
}

export default function ModelBadge() {
  const [info, { refetch }] = createResource(fetchModelInfo);
  const [expanded, setExpanded] = createSignal(false);

  setInterval(() => refetch(), 30000);

  return (
    <div class="relative">
      <button
        onClick={() => setExpanded(!expanded())}
        class="flex items-center gap-2 text-[11px] text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
      >
        {info.loading && <span>Connecting...</span>}
        {info.error && <span class="text-[#f87171]/70">Offline</span>}
        {info() && (
          <>
            <span class="inline-block w-1.5 h-1.5 rounded-full bg-[var(--color-accent)]" />
            <span class="tabular">
              Step {info()!.latest_train_step.toLocaleString()} · {info()!.status}
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
            <Row label="Model" value="Qwen3-30B-A3B" />
            <Row label="Step" value={`${info()!.latest_train_step.toLocaleString()} / 45,561`} />
            <Row
              label="Progress"
              value={`${((info()!.latest_train_step / 45561) * 100).toFixed(1)}%`}
            />
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
