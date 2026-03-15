import type { APIEvent } from "@solidjs/start/server";
import { hasDb, getTrainingMetrics, getTrainingConfig, getLatestMetric } from "~/lib/db";

const BACKEND_URL = process.env.CHAT_BACKEND_URL || "http://localhost:8787";
const STAGE_B_RUN_ID = "stage_b_llama8b";
const STAGE_A_RUN_ID = "stage_a_3ep";

function parseMetrics(rawMetrics: Array<{ step: number; metric_type: string; value_json: string }>) {
  return rawMetrics.map((m) => {
    const parsed = JSON.parse(m.value_json);
    return { step: m.step, metric_type: m.metric_type, ...parsed };
  });
}

export async function GET(_event: APIEvent) {
  // If D1 is not available (local dev), proxy to Python backend
  if (!hasDb()) {
    try {
      const resp = await fetch(`${BACKEND_URL}/api/training-stats`);
      return new Response(resp.body, {
        headers: { "Content-Type": "application/json" },
      });
    } catch {
      return new Response(JSON.stringify({ error: "Backend unavailable" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }
  }

  try {
    // Read both Stage A and Stage B from D1
    const [
      rawMetricsB, mixStats, runInfoB, latestTrainB,
      rawMetricsA, runInfoA,
    ] = await Promise.all([
      getTrainingMetrics(STAGE_B_RUN_ID),
      getTrainingConfig("mix_stats"),
      getTrainingConfig(`run_info_${STAGE_B_RUN_ID}`).then(r => r ?? getTrainingConfig("run_info")),
      getLatestMetric(STAGE_B_RUN_ID, "train_nll"),
      getTrainingMetrics(STAGE_A_RUN_ID),
      getTrainingConfig(`run_info_${STAGE_A_RUN_ID}`),
    ]);

    const metricsB = parseMetrics(rawMetricsB);
    const metricsA = parseMetrics(rawMetricsA);

    const currentStep = latestTrainB ? latestTrainB.step : 0;
    const totalSteps = runInfoB?.total_steps ?? 0;
    const progressPct =
      totalSteps > 0 ? Math.round((currentStep / totalSteps) * 1000) / 10 : 0;

    const result = {
      metrics: metricsB,
      mix_stats: mixStats ?? {},
      checkpoints: [],
      current_step: currentStep,
      total_steps: totalSteps,
      progress_pct: progressPct,
      model_name: runInfoB?.model_name ?? "",
      sampler_path: runInfoB?.sampler_path ?? "",
      sampler_step: runInfoB?.sampler_step ?? "",
      // Stage A
      stage_a: {
        metrics: metricsA,
        model_name: runInfoA?.model_name ?? "",
        total_steps: runInfoA?.total_steps ?? 0,
        current_step: runInfoA?.current_step ?? 0,
      },
    };

    return new Response(JSON.stringify(result), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (e: any) {
    return new Response(
      JSON.stringify({ error: e.message || "D1 query failed" }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
