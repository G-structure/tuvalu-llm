import type { APIEvent } from "@solidjs/start/server";
import { hasDb, getLatestMetric, getTrainingConfig } from "~/lib/chat-db";

const BACKEND_URL = process.env.CHAT_BACKEND_URL || "http://localhost:8787";
const DEFAULT_RUN_ID = "stage_b_llama8b";

function offlineModelInfo() {
  return {
    sampler_path: "",
    step: "0",
    run: DEFAULT_RUN_ID,
    status: "offline",
    latest_train_step: 0,
    latest_train_nll: null,
    latest_val_nll: null,
  };
}

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function GET(_event: APIEvent) {
  // If D1 is available, read from it directly
  if (hasDb()) {
    try {
      const [latestTrain, latestVal, runInfo] = await Promise.all([
        getLatestMetric(DEFAULT_RUN_ID, "train_nll"),
        getLatestMetric(DEFAULT_RUN_ID, "val_nll"),
        getTrainingConfig("run_info"),
      ]);

      const trainParsed = latestTrain
        ? JSON.parse(latestTrain.value_json)
        : null;
      const valParsed = latestVal ? JSON.parse(latestVal.value_json) : null;

      const result = {
        sampler_path: runInfo?.sampler_path ?? "",
        step: latestTrain ? String(latestTrain.step) : "0",
        run: DEFAULT_RUN_ID,
        status: "training",
        latest_train_step: latestTrain?.step ?? 0,
        latest_train_nll:
          trainParsed?.train_nll ?? trainParsed?.train_mean_nll ?? null,
        latest_val_nll: valParsed?.validation_mean_nll ?? null,
      };

      return json(result);
    } catch (e: any) {
      return json({
        ...offlineModelInfo(),
        error: e.message || "D1 query failed",
      });
    }
  }

  // Fallback: proxy to Python backend
  try {
    const resp = await fetch(`${BACKEND_URL}/api/model-info`);
    if (!resp.ok) {
      return json(offlineModelInfo());
    }
    return new Response(resp.body, {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return json(offlineModelInfo());
  }
}
