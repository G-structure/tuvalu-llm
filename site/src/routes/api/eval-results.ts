import type { APIEvent } from "@solidjs/start/server";
import { hasDb, getEvalRuns, getEvalPredictions } from "~/lib/chat-db";

export async function GET(event: APIEvent) {
  if (!hasDb()) {
    return new Response(JSON.stringify({ error: "D1 not available" }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }

  try {
    const url = new URL(event.request.url);
    const runId = url.searchParams.get("run_id");
    const modelKey = url.searchParams.get("model");
    const task = url.searchParams.get("task");

    // If specific predictions requested
    if (runId && modelKey && task) {
      const preds = await getEvalPredictions(runId, modelKey, task);
      return new Response(JSON.stringify({ predictions: preds }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    // Otherwise return all runs with parsed results
    const runs = await getEvalRuns();
    const parsed = runs.map((r) => ({
      ...r,
      results: JSON.parse(r.results_json),
    }));

    // Group by run_id
    const grouped: Record<string, any> = {};
    for (const run of parsed) {
      if (!grouped[run.run_id]) {
        grouped[run.run_id] = {
          run_id: run.run_id,
          budget: run.budget,
          created_at: run.created_at,
          models: {},
        };
      }
      grouped[run.run_id].models[run.model_key] = run.results;
    }

    return new Response(
      JSON.stringify({ runs: Object.values(grouped) }),
      { headers: { "Content-Type": "application/json" } }
    );
  } catch (e: any) {
    return new Response(
      JSON.stringify({ error: e.message || "D1 query failed" }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
