import type { APIEvent } from "@solidjs/start/server";
import { hasDb, getTrainingMetrics, getTrainingConfig, getLatestMetric } from "~/lib/db";

const BACKEND_URL = process.env.CHAT_BACKEND_URL || "http://localhost:8787";
const DEFAULT_RUN_ID = "stage_b_llama8b";

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
    // Read from D1
    const [rawMetrics, mixStats, runInfo, latestTrain] = await Promise.all([
      getTrainingMetrics(DEFAULT_RUN_ID),
      getTrainingConfig("mix_stats"),
      getTrainingConfig("run_info"),
      getLatestMetric(DEFAULT_RUN_ID, "train_nll"),
    ]);

    // Parse value_json for each metric and flatten into the shape the frontend expects
    const metrics = rawMetrics.map((m) => {
      const parsed = JSON.parse(m.value_json);
      return {
        step: m.step,
        metric_type: m.metric_type,
        ...parsed,
      };
    });

    const currentStep = latestTrain ? latestTrain.step : 0;
    const totalSteps = runInfo?.total_steps ?? 0;
    const progressPct =
      totalSteps > 0 ? Math.round((currentStep / totalSteps) * 1000) / 10 : 0;

    const result = {
      metrics,
      mix_stats: mixStats ?? {},
      checkpoints: [],
      current_step: currentStep,
      total_steps: totalSteps,
      progress_pct: progressPct,
      model_name: runInfo?.model_name ?? "",
      sampler_path: runInfo?.sampler_path ?? "",
      sampler_step: runInfo?.sampler_step ?? "",
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
