import { createResource, For, Show, createMemo } from "solid-js";
import { Title } from "@solidjs/meta";

interface TrainingStats {
  metrics: Array<Record<string, any>>;
  mix_stats: Record<string, any>;
  checkpoints: Array<Record<string, any>>;
  current_step: number;
  total_steps: number;
  progress_pct: number;
  model_name: string;
  sampler_path: string;
  sampler_step: string;
}

async function fetchStats(): Promise<TrainingStats> {
  const resp = await fetch("/api/training-stats");
  if (!resp.ok) throw new Error("Failed to fetch");
  return resp.json();
}

export default function Training() {
  const [stats, { refetch }] = createResource(fetchStats);
  setInterval(() => refetch(), 15000);

  const latestRunMetrics = createMemo(() => {
    if (!stats()) return [];
    const all = stats()!.metrics;
    let lastRestart = 0;
    for (let i = all.length - 1; i >= 0; i--) {
      if (all[i].step === 0) { lastRestart = i; break; }
    }
    return all.slice(lastRestart);
  });

  const trainMetrics = createMemo(() => {
    return latestRunMetrics().filter(
      (m: any) => "train_nll" in m || "train_mean_nll" in m
    ).map((m: any) => ({ ...m, nll: m.train_nll ?? m.train_mean_nll }));
  });

  const valMetrics = createMemo(() => {
    return latestRunMetrics().filter((m: any) => "validation_mean_nll" in m);
  });

  const genEvalMetrics = createMemo(() => {
    return latestRunMetrics().filter((m: any) => "gen_eval_chrf_pp" in m);
  });

  const latest = createMemo(() => {
    const t = trainMetrics();
    const v = valMetrics();
    const g = genEvalMetrics();
    return {
      train: t.length > 0 ? t[t.length - 1] : null,
      val: v.length > 0 ? v[v.length - 1] : null,
      gen: g.length > 0 ? g[g.length - 1] : null,
    };
  });

  const etaHours = createMemo(() => {
    if (!stats()) return null;
    const s = stats()!;
    const remaining = s.total_steps - s.current_step;
    const secPerStep = 5;
    return (remaining * secPerStep / 3600).toFixed(1);
  });

  const trainTrend = createMemo(() => {
    const t = trainMetrics();
    if (t.length < 20) return null;
    const recent = t[t.length - 1].nll;
    const prev = t[t.length - 20].nll;
    const delta = ((recent - prev) / prev * 100).toFixed(1);
    return { direction: recent < prev ? "down" : "up", delta };
  });

  return (
    <>
      <Title>Training — TVL</Title>
      <div class="min-h-screen bg-[var(--color-bg)]">

        {/* Nav */}
        <nav class="border-b border-[var(--color-border)] h-12 flex items-center px-6">
          <div class="max-w-5xl w-full mx-auto flex items-center justify-between">
            <div class="flex items-center gap-6">
              <a href="/" class="text-[13px] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors flex items-center gap-1.5">
                <span class="text-[var(--color-accent)] text-[10px]">&#10038;</span>
                TVL Chat
              </a>
              <span class="text-[var(--color-border-subtle)]">/</span>
              <span class="text-[13px] text-[var(--color-text)] font-medium">Training</span>
            </div>
            <button
              onClick={() => refetch()}
              class="text-[12px] text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
            >
              Refresh
            </button>
          </div>
        </nav>

        <Show
          when={stats()}
          fallback={
            <div class="flex items-center justify-center h-64">
              <div class="flex gap-1.5">
                <span class="typing-dot w-2 h-2 bg-[var(--color-accent)] rounded-full" />
                <span class="typing-dot w-2 h-2 bg-[var(--color-accent)] rounded-full" />
                <span class="typing-dot w-2 h-2 bg-[var(--color-accent)] rounded-full" />
              </div>
            </div>
          }
        >
          <div class="max-w-5xl mx-auto px-6 py-8 space-y-8">

            {/* Header section */}
            <div class="flex items-end justify-between">
              <div>
                <h1 class="text-[20px] font-semibold tracking-tight text-[var(--color-text)]">
                  Stage B Training
                </h1>
                <p class="text-[13px] text-[var(--color-text-muted)] mt-1">
                  {stats()!.model_name} · Bilingual capability adapter
                </p>
              </div>
              <div class="text-right">
                <div class="flex items-center gap-2 justify-end">
                  <span class="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)]" />
                  <span class="text-[12px] text-[var(--color-text-secondary)]">Training</span>
                </div>
              </div>
            </div>

            {/* Progress */}
            <div>
              <div class="flex items-baseline justify-between mb-3">
                <div class="flex items-baseline gap-3">
                  <span class="text-[32px] font-semibold tracking-tight tabular">
                    {stats()!.progress_pct}%
                  </span>
                  <span class="text-[13px] text-[var(--color-text-muted)] tabular">
                    Step {stats()!.current_step.toLocaleString()} of {stats()!.total_steps.toLocaleString()}
                  </span>
                </div>
                <Show when={etaHours()}>
                  <span class="text-[12px] text-[var(--color-text-muted)] tabular">
                    ~{etaHours()}h remaining
                  </span>
                </Show>
              </div>
              <div class="w-full bg-white/[0.04] rounded-full h-1.5">
                <div
                  class="progress-bar-animated h-1.5 rounded-full transition-all duration-1000 ease-out"
                  style={{ width: `${Math.max(0.5, stats()!.progress_pct)}%` }}
                />
              </div>
            </div>

            {/* Metrics row */}
            <div class="grid grid-cols-4 gap-px bg-[var(--color-border)] rounded-lg overflow-hidden">
              <Metric
                label="Train NLL"
                value={latest().train?.nll?.toFixed(4) ?? "—"}
                sub={latest().train ? `Step ${latest().train.step.toLocaleString()}` : ""}
                trend={trainTrend()}
              />
              <Metric
                label="Val NLL"
                value={latest().val?.validation_mean_nll?.toFixed(4) ?? "—"}
                sub={latest().val ? `Step ${latest().val.step.toLocaleString()}` : ""}
              />
              <Metric
                label="chrF++"
                value={latest().gen?.gen_eval_chrf_pp?.toFixed(1) ?? "—"}
                sub={latest().gen ? `Step ${latest().gen.step.toLocaleString()}` : ""}
              />
              <Metric
                label="BLEU"
                value={latest().gen?.gen_eval_bleu?.toFixed(1) ?? "—"}
                sub={latest().gen ? `Exact ${(latest().gen.gen_eval_exact_match * 100).toFixed(1)}%` : ""}
              />
            </div>

            {/* Loss chart */}
            <Show when={trainMetrics().length > 5}>
              <Card>
                <div class="flex items-center justify-between mb-6">
                  <span class="text-[13px] font-medium text-[var(--color-text)]">Loss</span>
                  <div class="flex items-center gap-5 text-[11px] text-[var(--color-text-muted)]">
                    <span class="flex items-center gap-1.5">
                      <span class="w-5 h-[1.5px] bg-[var(--color-accent)] rounded" /> Train
                    </span>
                    <Show when={valMetrics().length > 0}>
                      <span class="flex items-center gap-1.5">
                        <span class="w-5 h-[1.5px] rounded" style="background: #f0c674; opacity: 0.7" /> Val
                      </span>
                    </Show>
                  </div>
                </div>
                <LossChart data={trainMetrics()} valData={valMetrics()} />
              </Card>
            </Show>

            {/* Two columns */}
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Dataset composition */}
              <Show when={stats()!.mix_stats?.train}>
                <Card>
                  <div class="flex items-center justify-between mb-5">
                    <span class="text-[13px] font-medium text-[var(--color-text)]">Dataset composition</span>
                    <span class="text-[11px] text-[var(--color-text-muted)] tabular">
                      {stats()!.mix_stats.train.count?.toLocaleString()} examples · {stats()!.mix_stats.train.total_tokens_human}
                    </span>
                  </div>
                  <div class="space-y-4">
                    <For each={Object.entries(stats()!.mix_stats.train.by_source || {}).sort((a, b) => (b[1] as number) - (a[1] as number))}>
                      {([source, count]) => {
                        const total = stats()!.mix_stats.train.count || 1;
                        const pct = ((count as number) / total * 100);
                        return (
                          <div>
                            <div class="flex items-center justify-between mb-1.5">
                              <span class="text-[12px] text-[var(--color-text-secondary)] capitalize">{source.replace(/_/g, " ")}</span>
                              <span class="text-[12px] text-[var(--color-text-muted)] tabular">
                                {pct.toFixed(1)}%
                              </span>
                            </div>
                            <div class="w-full bg-white/[0.04] rounded-full h-1">
                              <div
                                class="h-1 rounded-full transition-all duration-500"
                                style={{
                                  width: `${pct}%`,
                                  background: sourceColor(source),
                                  opacity: "0.7",
                                }}
                              />
                            </div>
                          </div>
                        );
                      }}
                    </For>
                  </div>
                </Card>
              </Show>

              {/* Gen eval history */}
              <Card>
                <span class="text-[13px] font-medium text-[var(--color-text)] block mb-5">Evaluations</span>
                <Show
                  when={genEvalMetrics().length > 0}
                  fallback={
                    <p class="text-[12px] text-[var(--color-text-muted)]">
                      First generation eval at step 500.
                    </p>
                  }
                >
                  <table class="w-full text-[12px]">
                    <thead>
                      <tr class="text-[var(--color-text-muted)]">
                        <th class="text-left py-2 font-normal">Step</th>
                        <th class="text-right py-2 font-normal">chrF++</th>
                        <th class="text-right py-2 font-normal">BLEU</th>
                        <th class="text-right py-2 font-normal">Exact</th>
                      </tr>
                    </thead>
                    <tbody>
                      <For each={genEvalMetrics().slice().reverse()}>
                        {(m: any, idx) => (
                          <tr class={`border-t border-[var(--color-border)] ${idx() === 0 ? "text-[var(--color-text)]" : "text-[var(--color-text-secondary)]"}`}>
                            <td class="py-2.5 tabular">{m.step.toLocaleString()}</td>
                            <td class="py-2.5 text-right tabular">{m.gen_eval_chrf_pp?.toFixed(1)}</td>
                            <td class="py-2.5 text-right tabular">{m.gen_eval_bleu?.toFixed(1)}</td>
                            <td class="py-2.5 text-right tabular">{(m.gen_eval_exact_match * 100).toFixed(1)}%</td>
                          </tr>
                        )}
                      </For>
                    </tbody>
                  </table>
                </Show>
              </Card>
            </div>

            {/* Task families */}
            <Show when={stats()!.mix_stats?.train?.by_task_family}>
              <Card>
                <span class="text-[13px] font-medium text-[var(--color-text)] block mb-5">Task families</span>
                <div class="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 gap-4">
                  <For each={Object.entries(stats()!.mix_stats.train.by_task_family || {}).sort((a, b) => (b[1] as number) - (a[1] as number))}>
                    {([family, count]) => (
                      <div>
                        <div class="text-[16px] font-semibold tabular text-[var(--color-text)]">{formatCount(count as number)}</div>
                        <div class="text-[11px] text-[var(--color-text-muted)] capitalize mt-0.5">{family}</div>
                      </div>
                    )}
                  </For>
                </div>
              </Card>
            </Show>

            {/* Footer */}
            <div class="text-center py-4 text-[11px] text-[var(--color-text-muted)]">
              Tuvalu mo te Atua — Te gagana o Tuvalu
            </div>

          </div>
        </Show>
      </div>
    </>
  );
}


function Card(props: { children: any }) {
  return (
    <div class="bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg p-5">
      {props.children}
    </div>
  );
}


function Metric(props: {
  label: string;
  value: string;
  sub?: string;
  trend?: { direction: string; delta: string } | null;
}) {
  return (
    <div class="bg-[var(--color-surface)] p-4">
      <div class="text-[11px] text-[var(--color-text-muted)] mb-2">{props.label}</div>
      <div class="flex items-baseline gap-2">
        <span class="text-[22px] font-semibold tabular tracking-tight">{props.value}</span>
        <Show when={props.trend}>
          <span class={`text-[11px] tabular ${props.trend?.direction === "down" ? "text-[var(--color-accent)]" : "text-[#f87171]"}`}>
            {props.trend?.direction === "down" ? "↓" : "↑"}{Math.abs(parseFloat(props.trend?.delta || "0"))}%
          </span>
        </Show>
      </div>
      <Show when={props.sub}>
        <div class="text-[11px] text-[var(--color-text-muted)] mt-1 tabular">{props.sub}</div>
      </Show>
    </div>
  );
}


function LossChart(props: { data: any[]; valData: any[] }) {
  const width = 880;
  const height = 200;
  const pad = { t: 12, r: 12, b: 28, l: 48 };
  const chartW = width - pad.l - pad.r;
  const chartH = height - pad.t - pad.b;

  const paths = createMemo(() => {
    const d = props.data;
    if (d.length < 2) return { train: "", val: "", area: "", xLabels: [], yLabels: [] };

    const maxStep = d[d.length - 1].step;
    const minStep = d[0].step;
    const allNll = d.map((m: any) => m.nll);
    const maxNll = Math.max(...allNll);
    const minNll = Math.min(...allNll) * 0.95;
    const range = maxNll - minNll || 1;

    const sx = (s: number) => pad.l + ((s - minStep) / (maxStep - minStep || 1)) * chartW;
    const sy = (v: number) => pad.t + (1 - (v - minNll) / range) * chartH;

    const step = Math.max(1, Math.floor(d.length / 300));
    const sampled = d.filter((_: any, i: number) => i % step === 0 || i === d.length - 1);

    const train = sampled
      .map((m: any, i: number) => `${i === 0 ? "M" : "L"}${sx(m.step).toFixed(1)},${sy(m.nll).toFixed(1)}`)
      .join(" ");

    const area = train + ` L${sx(sampled[sampled.length - 1].step).toFixed(1)},${(pad.t + chartH).toFixed(1)} L${sx(sampled[0].step).toFixed(1)},${(pad.t + chartH).toFixed(1)} Z`;

    const val = props.valData.length > 1
      ? props.valData
          .map((m: any, i: number) => `${i === 0 ? "M" : "L"}${sx(m.step).toFixed(1)},${sy(m.validation_mean_nll).toFixed(1)}`)
          .join(" ")
      : "";

    const xCount = 5;
    const xLabels = Array.from({ length: xCount }, (_, i) => {
      const s = minStep + ((maxStep - minStep) * i) / (xCount - 1);
      return { x: sx(s), label: Math.round(s).toLocaleString() };
    });

    const yCount = 4;
    const yLabels = Array.from({ length: yCount }, (_, i) => {
      const v = minNll + (range * i) / (yCount - 1);
      return { y: sy(v), label: v.toFixed(2) };
    });

    return { train, val, area, xLabels, yLabels };
  });

  return (
    <svg viewBox={`0 0 ${width} ${height}`} class="w-full" style={{ "max-height": "240px" }}>
      <defs>
        <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#2ec4b6" stop-opacity="0.12" />
          <stop offset="100%" stop-color="#2ec4b6" stop-opacity="0" />
        </linearGradient>
      </defs>

      {/* Grid lines */}
      <For each={paths().yLabels}>
        {(yl) => (
          <>
            <line x1={pad.l} x2={width - pad.r} y1={yl.y} y2={yl.y} stroke="rgba(255,255,255,0.04)" stroke-width="1" />
            <text x={pad.l - 8} y={yl.y + 3.5} text-anchor="end" fill="rgba(255,255,255,0.3)" font-size="10" font-family="system-ui, sans-serif">{yl.label}</text>
          </>
        )}
      </For>
      <For each={paths().xLabels}>
        {(xl) => (
          <text x={xl.x} y={height - 6} text-anchor="middle" fill="rgba(255,255,255,0.3)" font-size="10" font-family="system-ui, sans-serif">{xl.label}</text>
        )}
      </For>

      {/* Area */}
      <Show when={paths().area}>
        <path d={paths().area} fill="url(#areaFill)" />
      </Show>

      {/* Train */}
      <Show when={paths().train}>
        <path d={paths().train} fill="none" stroke="#2ec4b6" stroke-width="1.5" />
      </Show>

      {/* Val */}
      <Show when={paths().val}>
        <path d={paths().val} fill="none" stroke="#f0c674" stroke-width="1.5" stroke-opacity="0.6" stroke-dasharray="4,3" />
      </Show>
    </svg>
  );
}


function sourceColor(source: string): string {
  const colors: Record<string, string> = {
    english: "#60a5fa",
    synthetic_tvl: "#34d399",
    crosslingual: "#2ec4b6",
    anchor: "#f0c674",
    real_tvl_chat: "#f87171",
  };
  return colors[source] || "rgba(255,255,255,0.3)";
}


function formatCount(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(0)}K`;
  return n.toString();
}
