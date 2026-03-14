-- Training metrics (one row per logged metric point)
CREATE TABLE IF NOT EXISTS training_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL DEFAULT 'stage_b_llama8b',
  step INTEGER NOT NULL,
  metric_type TEXT NOT NULL,  -- 'train_nll', 'val_nll', 'gen_eval'
  value_json TEXT NOT NULL,   -- JSON blob of all metrics at this step
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON training_metrics(run_id, step);
CREATE INDEX IF NOT EXISTS idx_metrics_type ON training_metrics(run_id, metric_type);

-- Training config / mix stats (key-value store for static config)
CREATE TABLE IF NOT EXISTS training_config (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
