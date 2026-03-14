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

export function hasDb(): boolean {
  return getDb() !== null;
}
