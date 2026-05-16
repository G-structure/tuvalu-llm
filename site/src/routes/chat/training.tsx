import { createResource, For, Show, createMemo, onCleanup, onMount } from "solid-js";
import { isServer } from "solid-js/web";
import OGMeta from "~/components/OGMeta";
import StructuredData from "~/components/StructuredData";
import { breadcrumbList } from "~/lib/seo";
import { absoluteChatUrl, CHAT_META, SITE_ORIGINS } from "~/lib/site";
import { offlineTrainingStats, type TrainingStats } from "~/lib/training-snapshot";

const REFRESH_MS = 15000;

async function fetchStats(): Promise<TrainingStats | undefined> {
  if (isServer) return offlineTrainingStats("Training backend unavailable");
  try {
    const resp = await fetch("/api/training-stats");
    if (!resp.ok) return offlineTrainingStats("Training backend unavailable");
    const data = (await resp.json()) as TrainingStats;
    return data?.metrics ? data : offlineTrainingStats("Training data unavailable");
  } catch {
    return offlineTrainingStats("Training backend unavailable");
  }
}

export default function Training() {
  const [stats, { refetch, mutate }] = createResource(fetchStats);

  onMount(() => {
    void (async () => {
      const next = await fetchStats();
      if (next) mutate(next);
    })();

    const timer = setInterval(async () => {
      const next = await fetchStats();
      if (next) mutate(next);
    }, REFRESH_MS);
    onCleanup(() => clearInterval(timer));
  });

  const latestRunMetrics = createMemo(() => {
    if (!stats()) return [];
    const all = stats()!.metrics;
    let lastRestart = 0;
    for (let i = all.length - 1; i >= 0; i--) {
      if (all[i].step === 0) {
        lastRestart = i;
        break;
      }
    }
    return all.slice(lastRestart);
  });

  const trainMetrics = createMemo(() =>
    latestRunMetrics()
      .filter((m: any) => "train_nll" in m || "train_mean_nll" in m)
      .map((m: any) => ({ ...m, nll: m.train_nll ?? m.train_mean_nll }))
  );

  const valMetrics = createMemo(() =>
    latestRunMetrics().filter((m: any) => "validation_mean_nll" in m)
  );

  const genEvalMetrics = createMemo(() =>
    latestRunMetrics().filter((m: any) => "gen_eval_chrf_pp" in m)
  );

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
    if (!stats() || stats()!.status === "offline") return null;
    const remaining = Math.max(0, stats()!.total_steps - stats()!.current_step);
    return ((remaining * 5) / 3600).toFixed(1);
  });

  const trainTrend = createMemo(() => {
    const t = trainMetrics();
    if (t.length < 4) return null;
    const recent = t[t.length - 1].nll;
    const prev = t[Math.max(0, t.length - 4)].nll;
    if (!prev) return null;
    const delta = (((recent - prev) / prev) * 100).toFixed(1);
    return { direction: recent < prev ? "down" : "up", delta };
  });

  const isOffline = createMemo(() => stats()?.status === "offline");

  const sourceMix = createMemo(() => {
    const train = stats()?.mix_stats?.train;
    const bySource = train?.by_source || {};
    const total =
      Number(train?.count) ||
      Object.values(bySource).reduce((sum, value) => sum + Number(value || 0), 0) ||
      1;

    return Object.entries(bySource)
      .map(([source, count]) => ({
        source,
        count: Number(count || 0),
        pct: (Number(count || 0) / total) * 100,
      }))
      .sort((a, b) => b.count - a.count);
  });

  const taskFamilies = createMemo(() => {
    const byFamily = stats()?.mix_stats?.train?.by_task_family || {};
    return Object.entries(byFamily)
      .map(([family, count]) => ({ family, count: Number(count || 0) }))
      .sort((a, b) => b.count - a.count);
  });

  const latestCheckpoint = createMemo(() => {
    const checkpoints = stats()?.checkpoints || [];
    return checkpoints.length ? checkpoints[checkpoints.length - 1] : null;
  });

  const progressWidth = createMemo(() =>
    `${Math.min(100, Math.max(1.5, stats()?.progress_pct || 0))}%`
  );

  const exactMatchLabel = createMemo(() => {
    const exact = latest().gen?.gen_eval_exact_match;
    return exact != null ? `${(exact * 100).toFixed(1)}% exact` : "";
  });

  return (
    <>
      <OGMeta
        title="TVL Training Dashboard"
        description="Live training metrics for the bilingual Tuvaluan-English language model."
        url={absoluteChatUrl("/chat/training")}
        image={CHAT_META.trainingOgImage}
        imageOrigin={SITE_ORIGINS.chat}
        imageWidth={CHAT_META.defaultOgImageWidth}
        imageHeight={CHAT_META.defaultOgImageHeight}
        imageAlt="TVL Chat social card for live model training metrics."
        siteName={CHAT_META.productName}
        titleSuffix={CHAT_META.productName}
      />
      <StructuredData
        data={[
          {
            "@context": "https://schema.org",
            "@type": "WebPage",
            name: "TVL Training Dashboard",
            description:
              "Live training metrics for the bilingual Tuvaluan-English language model.",
            url: absoluteChatUrl("/chat/training"),
            isPartOf: {
              "@id": `${SITE_ORIGINS.organization}/#website`,
            },
            inLanguage: ["tvl", "en"],
          },
          breadcrumbList([
            { name: CHAT_META.productName, url: absoluteChatUrl("/chat") },
            { name: "Training", url: absoluteChatUrl("/chat/training") },
          ]),
        ]}
      />

      <div class="chat-theme training-shell">
        <nav aria-label="Training navigation" class="chat-nav training-nav">
          <div class="chat-nav__left">
            <a href="/" class="chat-nav__brand">
              <span class="chat-nav__brand-mark" aria-hidden="true" />
              <span>
                <strong>Fenua</strong>
                <em>Intelligence</em>
              </span>
            </a>
            <h1 class="chat-nav__title">Training Lab</h1>
          </div>
          <div class="chat-nav__actions training-nav__actions">
            <a href="/chat" class="chat-nav__link">
              Chat
            </a>
            <a href="/chat/eval" class="chat-nav__link">
              Eval
            </a>
            <button type="button" onClick={() => refetch()} class="chat-nav__button">
              Refresh
            </button>
          </div>
        </nav>

        <Show when={stats()} fallback={<TrainingSkeleton />}>
          {(s) => (
            <main class="training-main">
              <div class="training-main__inner">
                <section class="training-hero" aria-label="Training status">
                  <div class="training-hero__copy">
                    <p class="training-kicker">Fenua model training</p>
                    <h2>Stage B bilingual adapter</h2>
                    <p>
                      A clearer view of the model run: loss, generation evals,
                      dataset mix, and the community signals feeding the next
                      Tuvaluan-English iteration.
                    </p>
                    <div class="training-hero__chips">
                      <span>{s().model_name || "TVL model"}</span>
                      <span>{s().sampler_path || "Sampler pending"}</span>
                    </div>
                  </div>

                  <aside class="training-progress-panel">
                    <div class="training-status-row">
                      <span
                        class={`training-status-dot ${isOffline() ? "is-paused" : ""}`}
                      />
                      <strong>{isOffline() ? "Offline snapshot" : "Live training"}</strong>
                    </div>
                    <div
                      class="training-progress-ring"
                      style={{ "--progress": progressWidth() }}
                    >
                      <span>{s().progress_pct}%</span>
                    </div>
                    <div class="training-progress-copy">
                      <strong>
                        {s().current_step.toLocaleString()} /{" "}
                        {s().total_steps.toLocaleString()} steps
                      </strong>
                      <span>
                        {etaHours() ? `About ${etaHours()}h remaining` : "Latest packaged run"}
                      </span>
                    </div>
                    <div class="training-progress-track">
                      <span style={{ width: progressWidth() }} />
                    </div>
                  </aside>
                </section>

                <Show when={isOffline()}>
                  <div class="training-offline-banner">
                    Showing a packaged local snapshot because the live training
                    backend is not reachable in this environment.
                  </div>
                </Show>

                <section class="training-metric-grid" aria-label="Current metrics">
                  <MetricCard
                    label="Train NLL"
                    value={formatMetric(latest().train?.nll, 4)}
                    sub={latest().train ? `Step ${latest().train.step.toLocaleString()}` : "No train metric"}
                    trend={trainTrend()}
                  />
                  <MetricCard
                    label="Validation NLL"
                    value={formatMetric(latest().val?.validation_mean_nll, 4)}
                    sub={latest().val ? `Step ${latest().val.step.toLocaleString()}` : "No validation metric"}
                    tone="gold"
                  />
                  <MetricCard
                    label="chrF++"
                    value={formatMetric(latest().gen?.gen_eval_chrf_pp, 1)}
                    sub={latest().gen ? `Step ${latest().gen.step.toLocaleString()}` : "First eval pending"}
                  />
                  <MetricCard
                    label="BLEU"
                    value={formatMetric(latest().gen?.gen_eval_bleu, 1)}
                    sub={exactMatchLabel() || "Exact match pending"}
                    tone="red"
                  />
                </section>

                <section class="training-status-grid">
                  <InfoTile label="Last updated" value={formatUpdated(s().updated_at)} />
                  <InfoTile
                    label="Checkpoint"
                    value={
                      latestCheckpoint()
                        ? `${latestCheckpoint()!.label || "Checkpoint"} at ${Number(latestCheckpoint()!.step || 0).toLocaleString()}`
                        : "No checkpoint yet"
                    }
                  />
                  <InfoTile label="Sampler step" value={s().sampler_step || "Not set"} />
                </section>

                <TrainingPanel
                  eyebrow="Loss curve"
                  title="Training is moving in the right direction"
                  action={
                    <div class="training-legend" aria-label="Chart legend">
                      <span><i class="is-train" /> Train</span>
                      <span><i class="is-val" /> Validation</span>
                    </div>
                  }
                >
                  <Show
                    when={trainMetrics().length > 5}
                    fallback={<EmptyPanel text="Loss history will appear after the first few train metrics arrive." />}
                  >
                    <LossChart data={trainMetrics()} valData={valMetrics()} />
                  </Show>
                </TrainingPanel>

                <div class="training-two-column">
                  <TrainingPanel
                    eyebrow="Dataset"
                    title="What the model is learning from"
                    meta={
                      s().mix_stats?.train?.count
                        ? `${s().mix_stats.train.count.toLocaleString()} examples`
                        : undefined
                    }
                  >
                    <Show
                      when={sourceMix().length > 0}
                      fallback={<EmptyPanel text="Dataset composition is not available for this run yet." />}
                    >
                      <div class="training-bars">
                        <For each={sourceMix()}>
                          {(item) => (
                            <div class="training-bar-row">
                              <div>
                                <strong>{humanize(item.source)}</strong>
                                <span>{formatCount(item.count)}</span>
                              </div>
                              <div class="training-bar-track">
                                <span
                                  style={{
                                    width: `${Math.max(2, item.pct)}%`,
                                    background: sourceColor(item.source),
                                  }}
                                />
                              </div>
                            </div>
                          )}
                        </For>
                      </div>
                    </Show>
                  </TrainingPanel>

                  <TrainingPanel eyebrow="Evaluations" title="Generation quality">
                    <Show
                      when={genEvalMetrics().length > 0}
                      fallback={<EmptyPanel text="Generation evals start after the first scheduled eval step." />}
                    >
                      <div class="training-table-wrap">
                        <table class="training-table">
                          <thead>
                            <tr>
                              <th>Step</th>
                              <th>chrF++</th>
                              <th>BLEU</th>
                              <th>Exact</th>
                            </tr>
                          </thead>
                          <tbody>
                            <For each={genEvalMetrics().slice().reverse().slice(0, 6)}>
                              {(metric: any, idx) => (
                                <tr class={idx() === 0 ? "is-latest" : ""}>
                                  <td>{metric.step.toLocaleString()}</td>
                                  <td>{formatMetric(metric.gen_eval_chrf_pp, 1)}</td>
                                  <td>{formatMetric(metric.gen_eval_bleu, 1)}</td>
                                  <td>{formatPercent(metric.gen_eval_exact_match)}</td>
                                </tr>
                              )}
                            </For>
                          </tbody>
                        </table>
                      </div>
                    </Show>
                  </TrainingPanel>
                </div>

                <Show when={taskFamilies().length > 0}>
                  <TrainingPanel eyebrow="Coverage" title="Task families">
                    <div class="training-task-grid">
                      <For each={taskFamilies()}>
                        {(item) => (
                          <div class="training-task-tile">
                            <strong>{formatCount(item.count)}</strong>
                            <span>{humanize(item.family)}</span>
                          </div>
                        )}
                      </For>
                    </div>
                  </TrainingPanel>
                </Show>

                <footer class="training-footer">
                  Tuvalu mo te Atua - Te gagana o Tuvalu
                </footer>
              </div>
            </main>
          )}
        </Show>
      </div>
    </>
  );
}

function TrainingSkeleton() {
  return (
    <main class="training-main">
      <div class="training-main__inner">
        <div class="training-skeleton">
          <span />
          <span />
          <span />
        </div>
      </div>
    </main>
  );
}

function TrainingPanel(props: {
  eyebrow: string;
  title: string;
  meta?: string;
  action?: any;
  children: any;
}) {
  return (
    <section class="training-panel">
      <div class="training-panel__head">
        <div>
          <p>{props.eyebrow}</p>
          <h3>{props.title}</h3>
          <Show when={props.meta}>
            <span>{props.meta}</span>
          </Show>
        </div>
        <div class="training-panel__action">{props.action}</div>
      </div>
      {props.children}
    </section>
  );
}

function MetricCard(props: {
  label: string;
  value: string;
  sub?: string;
  tone?: "blue" | "gold" | "red";
  trend?: { direction: string; delta: string } | null;
}) {
  return (
    <article class={`training-metric-card training-metric-card--${props.tone || "blue"}`}>
      <span>{props.label}</span>
      <div>
        <strong>{props.value}</strong>
        <Show when={props.trend}>
          <em class={props.trend?.direction === "down" ? "is-good" : "is-up"}>
            {props.trend?.direction === "down" ? "down" : "up"}{" "}
            {Math.abs(parseFloat(props.trend?.delta || "0"))}%
          </em>
        </Show>
      </div>
      <p>{props.sub}</p>
    </article>
  );
}

function InfoTile(props: { label: string; value: string }) {
  return (
    <div class="training-info-tile">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}

function EmptyPanel(props: { text: string }) {
  return <div class="training-empty">{props.text}</div>;
}

function LossChart(props: { data: any[]; valData: any[] }) {
  const width = 880;
  const height = 240;
  const pad = { t: 18, r: 16, b: 34, l: 48 };
  const chartW = width - pad.l - pad.r;
  const chartH = height - pad.t - pad.b;

  const paths = createMemo(() => {
    const trainData = props.data;
    if (trainData.length < 2) return { train: "", val: "", area: "", xLabels: [], yLabels: [] };

    const allSteps = [
      ...trainData.map((m: any) => m.step),
      ...props.valData.map((m: any) => m.step),
    ];
    const minStep = Math.min(...allSteps);
    const maxStep = Math.max(...allSteps);
    const allNll = [
      ...trainData.map((m: any) => m.nll),
      ...props.valData.map((m: any) => m.validation_mean_nll),
    ].filter((value) => Number.isFinite(value));
    const maxNll = Math.max(...allNll);
    const minNll = Math.min(...allNll) * 0.94;
    const range = maxNll - minNll || 1;

    const sx = (step: number) => pad.l + ((step - minStep) / (maxStep - minStep || 1)) * chartW;
    const sy = (value: number) => pad.t + (1 - (value - minNll) / range) * chartH;

    const sampleStep = Math.max(1, Math.floor(trainData.length / 260));
    const sampled = trainData.filter(
      (_: any, index: number) => index % sampleStep === 0 || index === trainData.length - 1
    );

    const train = sampled
      .map((m: any, index: number) => `${index === 0 ? "M" : "L"}${sx(m.step).toFixed(1)},${sy(m.nll).toFixed(1)}`)
      .join(" ");

    const area =
      `${train} L${sx(sampled[sampled.length - 1].step).toFixed(1)},${(pad.t + chartH).toFixed(1)}` +
      ` L${sx(sampled[0].step).toFixed(1)},${(pad.t + chartH).toFixed(1)} Z`;

    const val = props.valData.length > 1
      ? props.valData
          .map((m: any, index: number) =>
            `${index === 0 ? "M" : "L"}${sx(m.step).toFixed(1)},${sy(m.validation_mean_nll).toFixed(1)}`
          )
          .join(" ")
      : "";

    const xLabels = Array.from({ length: 5 }, (_, index) => {
      const step = minStep + ((maxStep - minStep) * index) / 4;
      return { x: sx(step), label: Math.round(step).toLocaleString() };
    });

    const yLabels = Array.from({ length: 4 }, (_, index) => {
      const value = minNll + (range * index) / 3;
      return { y: sy(value), label: value.toFixed(2) };
    });

    return { train, val, area, xLabels, yLabels };
  });

  return (
    <svg viewBox={`0 0 ${width} ${height}`} class="training-chart" role="img" aria-label="Training and validation loss chart">
      <defs>
        <linearGradient id="trainingLossArea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#75d6f4" stop-opacity="0.2" />
          <stop offset="100%" stop-color="#75d6f4" stop-opacity="0" />
        </linearGradient>
      </defs>

      <For each={paths().yLabels}>
        {(yl) => (
          <>
            <line x1={pad.l} x2={width - pad.r} y1={yl.y} y2={yl.y} class="training-chart__grid" />
            <text x={pad.l - 9} y={yl.y + 4} text-anchor="end" class="training-chart__label">
              {yl.label}
            </text>
          </>
        )}
      </For>
      <For each={paths().xLabels}>
        {(xl) => (
          <text x={xl.x} y={height - 8} text-anchor="middle" class="training-chart__label">
            {xl.label}
          </text>
        )}
      </For>

      <Show when={paths().area}>
        <path d={paths().area} fill="url(#trainingLossArea)" />
      </Show>
      <Show when={paths().train}>
        <path d={paths().train} class="training-chart__train" />
      </Show>
      <Show when={paths().val}>
        <path d={paths().val} class="training-chart__val" />
      </Show>
    </svg>
  );
}

function sourceColor(source: string): string {
  const colors: Record<string, string> = {
    anchor: "#061b35",
    crosslingual: "#00a7d8",
    english: "#ffc400",
    real_tvl_chat: "#c83b32",
    synthetic_tvl: "#75d6f4",
  };
  return colors[source] || "#607585";
}

function formatMetric(value: number | undefined | null, digits: number): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function formatPercent(value: number | undefined | null): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-";
}

function formatCount(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}K`;
  return n.toString();
}

function humanize(value: string): string {
  return value.replace(/_/g, " ");
}

function formatUpdated(value?: string): string {
  if (!value) return "Recent snapshot";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Recent snapshot";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}
