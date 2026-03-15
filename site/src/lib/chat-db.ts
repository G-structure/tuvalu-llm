// D1 database helper for the chat/training dashboard

function getDb(): D1Database | null {
  // Production: Cloudflare injects DB binding
  const env = (process.env as any) || (globalThis as any).__env__ || {};
  if (env.DB) return env.DB;
  // Local dev: return null (will use local backend instead)
  return null;
}

export interface TrainingMetricRow {
  id: number;
  run_id: string;
  step: number;
  metric_type: string;
  value_json: string;
  created_at: string;
}

export async function getTrainingMetrics(runId: string): Promise<TrainingMetricRow[]> {
  const db = getDb();
  if (!db) return [];
  const { results } = await db
    .prepare(
      `SELECT id, run_id, step, metric_type, value_json, created_at
       FROM training_metrics
       WHERE run_id = ?
       ORDER BY step ASC, id ASC`
    )
    .bind(runId)
    .all();
  return results as unknown as TrainingMetricRow[];
}

export async function getTrainingConfig(key: string): Promise<any | null> {
  const db = getDb();
  if (!db) return null;
  const row = await db
    .prepare(`SELECT value_json FROM training_config WHERE key = ?`)
    .bind(key)
    .first();
  if (!row) return null;
  return JSON.parse((row as any).value_json);
}

export async function getLatestMetric(
  runId: string,
  metricType: string
): Promise<TrainingMetricRow | null> {
  const db = getDb();
  if (!db) return null;
  const row = await db
    .prepare(
      `SELECT id, run_id, step, metric_type, value_json, created_at
       FROM training_metrics
       WHERE run_id = ? AND metric_type = ?
       ORDER BY step DESC
       LIMIT 1`
    )
    .bind(runId, metricType)
    .first();
  return (row as unknown as TrainingMetricRow) || null;
}

export interface EvalRunRow {
  id: number;
  run_id: string;
  model_key: string;
  budget: string;
  results_json: string;
  created_at: string;
}

export interface EvalPredictionRow {
  id: number;
  run_id: string;
  model_key: string;
  task: string;
  example_id: string;
  prediction: string;
  reference: string;
  metadata_json: string;
  created_at: string;
}

export async function getEvalRuns(): Promise<EvalRunRow[]> {
  const db = getDb();
  if (!db) return [];
  const { results } = await db
    .prepare(
      `SELECT id, run_id, model_key, budget, results_json, created_at
       FROM eval_runs
       ORDER BY created_at DESC`
    )
    .all();
  return results as unknown as EvalRunRow[];
}

export async function getEvalPredictions(
  runId: string,
  modelKey: string,
  task: string,
  limit: number = 20,
): Promise<EvalPredictionRow[]> {
  const db = getDb();
  if (!db) return [];
  const { results } = await db
    .prepare(
      `SELECT id, run_id, model_key, task, example_id, prediction, reference, metadata_json
       FROM eval_predictions
       WHERE run_id = ? AND model_key = ? AND task = ?
       LIMIT ?`
    )
    .bind(runId, modelKey, task, limit)
    .all();
  return results as unknown as EvalPredictionRow[];
}

export function hasDb(): boolean {
  return getDb() !== null;
}
