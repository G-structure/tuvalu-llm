export interface TrainingStats {
  metrics: Array<Record<string, any>>;
  mix_stats: Record<string, any>;
  checkpoints: Array<Record<string, any>>;
  current_step: number;
  total_steps: number;
  progress_pct: number;
  model_name: string;
  sampler_path: string;
  sampler_step: string;
  status?: "training" | "offline";
  error?: string;
  updated_at?: string;
}

const trainSeries = [
  [0, 3.84],
  [200, 3.42],
  [400, 3.18],
  [700, 2.92],
  [1000, 2.71],
  [1400, 2.49],
  [1800, 2.33],
  [2200, 2.19],
  [2600, 2.08],
  [3000, 1.98],
  [3600, 1.86],
  [4200, 1.76],
  [4800, 1.68],
  [5400, 1.61],
  [6000, 1.55],
  [6600, 1.49],
  [7200, 1.45],
  [7800, 1.41],
  [8400, 1.38],
] as const;

const valSeries = [
  [1000, 2.86],
  [2000, 2.38],
  [3000, 2.12],
  [4000, 1.93],
  [5000, 1.82],
  [6000, 1.73],
  [7000, 1.67],
  [8000, 1.63],
] as const;

const evalSeries = [
  [2000, 24.6, 6.8, 0.08],
  [4000, 31.2, 10.9, 0.14],
  [6000, 37.8, 15.6, 0.21],
  [8000, 41.8, 18.4, 0.26],
] as const;

export function offlineTrainingStats(reason = "Backend unavailable"): TrainingStats {
  const metrics = [
    ...trainSeries.map(([step, trainMeanNll]) => ({
      step,
      train_mean_nll: trainMeanNll,
    })),
    ...valSeries.map(([step, validationMeanNll]) => ({
      step,
      validation_mean_nll: validationMeanNll,
    })),
    ...evalSeries.map(([step, chrf, bleu, exact]) => ({
      step,
      gen_eval_chrf_pp: chrf,
      gen_eval_bleu: bleu,
      gen_eval_exact_match: exact,
    })),
  ].sort((a, b) => a.step - b.step);

  return {
    metrics,
    mix_stats: {
      train: {
        count: 342000,
        total_tokens_human: "48.7M tokens",
        by_source: {
          synthetic_tvl: 128000,
          crosslingual: 84000,
          english: 62000,
          anchor: 42000,
          real_tvl_chat: 26000,
        },
        by_task_family: {
          translation: 134000,
          chat: 72000,
          qa: 54000,
          summarization: 36000,
          grammar: 26000,
          safety: 20000,
        },
      },
    },
    checkpoints: [
      { step: 4000, label: "Stage B warmup" },
      { step: 8000, label: "Best eval snapshot" },
    ],
    current_step: 8400,
    total_steps: 12000,
    progress_pct: 70,
    model_name: "TVL Stage B local snapshot",
    sampler_path: "FriezaForce/tvl-en-llm-translation-stage-b",
    sampler_step: "8400",
    status: "offline",
    error: reason,
    updated_at: "2026-05-15T00:00:00.000Z",
  };
}
