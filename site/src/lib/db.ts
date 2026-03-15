import type {
  Article,
  ArticleFeedbackFormSubmission,
  Category,
  FeedbackSubmission,
  FateleStats,
  SignalSubmission,
} from "./types";

// D1 binding is injected by Cloudflare Workers runtime
// In dev mode, lazily initialize via wrangler's getPlatformProxy()
let _devProxy: any = null;
let _communitySchemaReady: Promise<void> | null = null;

async function getDb(): Promise<D1Database> {
  const db = (process.env as any).DB || (globalThis as any).__env__?.DB;
  if (db) return db;

  // Dev fallback: use wrangler's local D1 emulation
  if (!_devProxy) {
    const { getPlatformProxy } = await import("wrangler");
    _devProxy = await getPlatformProxy({ persist: { path: ".wrangler/state/v3" } });
    (globalThis as any).__env__ = _devProxy.env;
  }
  return _devProxy.env.DB;
}

async function ensureCommunitySchema(db: D1Database): Promise<void> {
  if (_communitySchemaReady) {
    await _communitySchemaReady;
    return;
  }

  _communitySchemaReady = (async () => {
    await db
      .prepare(
        `CREATE TABLE IF NOT EXISTS article_feedback_forms (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          article_id TEXT NOT NULL REFERENCES articles(id),
          session_id TEXT,
          island TEXT,
          helpful_score INTEGER NOT NULL,
          mode_preference TEXT NOT NULL,
          correction_paragraph_idx INTEGER,
          correction_text TEXT,
          created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )`
      )
      .run();

    await db
      .prepare(
        `CREATE INDEX IF NOT EXISTS idx_article_feedback_forms_article
         ON article_feedback_forms(article_id)`
      )
      .run();

    await db
      .prepare(
        `CREATE INDEX IF NOT EXISTS idx_article_feedback_forms_created
         ON article_feedback_forms(created_at)`
      )
      .run();

    await db
      .prepare(
        `CREATE INDEX IF NOT EXISTS idx_article_feedback_forms_session
         ON article_feedback_forms(session_id, article_id)`
      )
      .run();
  })();

  await _communitySchemaReady;
}

// Full select (with bodies) — for single article detail page
const ARTICLE_SELECT = `
  SELECT
    a.id, a.source_id, a.url,
    a.title_en, a.body_en, a.author,
    a.published_at, a.category, a.tags,
    a.image_url, a.image_alt, a.image_width, a.image_height,
    a.og_description_en, a.word_count,
    CASE WHEN t.is_collapsed = 1 THEN NULL ELSE t.title_tvl END AS title_tvl,
    CASE WHEN t.is_collapsed = 1 THEN NULL ELSE t.body_tvl END AS body_tvl,
    CASE WHEN t.is_collapsed = 1 THEN NULL ELSE t.og_description_tvl END AS og_description_tvl
  FROM articles a
  LEFT JOIN translations t ON t.article_id = a.id
`;

// Lightweight select (no bodies) — for list/search pages
const ARTICLE_LIST_SELECT = `
  SELECT
    a.id, a.source_id, a.url,
    a.title_en, a.author,
    a.published_at, a.category,
    a.image_url, a.image_alt,
    CASE WHEN t.is_collapsed = 1 THEN NULL ELSE t.title_tvl END AS title_tvl
  FROM articles a
  LEFT JOIN translations t ON t.article_id = a.id
`;

export async function getArticles(
  limit = 20,
  offset = 0,
  category?: string
): Promise<Article[]> {
  const db = await getDb();
  if (category) {
    const { results } = await db
      .prepare(
        `${ARTICLE_LIST_SELECT}
         WHERE a.category = ?
         ORDER BY a.published_at DESC
         LIMIT ? OFFSET ?`
      )
      .bind(category, limit, offset)
      .all();
    return results as unknown as Article[];
  }
  const { results } = await db
    .prepare(
      `${ARTICLE_LIST_SELECT}
       ORDER BY a.published_at DESC
       LIMIT ? OFFSET ?`
    )
    .bind(limit, offset)
    .all();
  return results as unknown as Article[];
}

export async function getArticle(id: string): Promise<Article | undefined> {
  const db = await getDb();
  const result = await db
    .prepare(`${ARTICLE_SELECT} WHERE a.id = ?`)
    .bind(id)
    .first();
  return (result as unknown as Article) || undefined;
}

export async function getCategories(): Promise<Category[]> {
  const db = await getDb();
  const { results } = await db
    .prepare(
      `SELECT category AS slug, COUNT(*) AS count
       FROM articles
       WHERE category IS NOT NULL AND category != ''
       GROUP BY category
       ORDER BY count DESC`
    )
    .all();
  return results as unknown as Category[];
}

export async function getArticleCount(category?: string): Promise<number> {
  const db = await getDb();
  if (category) {
    const row = await db
      .prepare("SELECT COUNT(*) AS cnt FROM articles WHERE category = ?")
      .bind(category)
      .first();
    return (row as any)?.cnt ?? 0;
  }
  const row = await db
    .prepare("SELECT COUNT(*) AS cnt FROM articles")
    .first();
  return (row as any)?.cnt ?? 0;
}

export async function searchArticles(query: string, limit = 20): Promise<Article[]> {
  const db = await getDb();
  const pattern = `%${query}%`;
  const { results } = await db
    .prepare(
      `${ARTICLE_LIST_SELECT}
       WHERE a.title_en LIKE ?1 OR t.title_tvl LIKE ?1
          OR a.body_en LIKE ?1 OR t.body_tvl LIKE ?1
       ORDER BY a.published_at DESC
       LIMIT ?2`
    )
    .bind(pattern, limit)
    .all();
  return results as unknown as Article[];
}

export async function insertFeedback(fb: FeedbackSubmission): Promise<void> {
  const db = await getDb();
  await ensureCommunitySchema(db);
  await db
    .prepare(
      `INSERT INTO feedback (article_id, paragraph_idx, feedback_type, island, session_id)
       VALUES (?, ?, ?, ?, ?)`
    )
    .bind(fb.article_id, fb.paragraph_idx, fb.feedback_type, fb.island ?? null, fb.session_id ?? null)
    .run();
}

export async function insertSignal(sig: SignalSubmission): Promise<void> {
  const db = await getDb();
  await ensureCommunitySchema(db);
  await db
    .prepare(
      `INSERT INTO implicit_signals (article_id, signal_type, paragraph_index, session_id, island)
       VALUES (?, ?, ?, ?, ?)`
    )
    .bind(sig.article_id, sig.signal_type, sig.paragraph_index ?? null, sig.session_id ?? null, sig.island ?? null)
    .run();
}

export async function insertArticleFeedbackForm(
  feedback: ArticleFeedbackFormSubmission
): Promise<void> {
  const db = await getDb();
  await ensureCommunitySchema(db);
  await db
    .prepare(
      `INSERT INTO article_feedback_forms (
         article_id,
         session_id,
         island,
         helpful_score,
         mode_preference,
         correction_paragraph_idx,
         correction_text
       ) VALUES (?, ?, ?, ?, ?, ?, ?)`
    )
    .bind(
      feedback.article_id,
      feedback.session_id ?? null,
      feedback.island ?? null,
      feedback.helpful_score,
      feedback.mode_preference,
      feedback.correction_paragraph_idx ?? null,
      feedback.correction_text?.trim() || null
    )
    .run();
}

export async function getFateleStats(): Promise<FateleStats> {
  const db = await getDb();
  await ensureCommunitySchema(db);
  const [
    total,
    { results: islands },
    articleFeedback,
    corrections,
    helpful,
    { results: modePreferences },
  ] = await Promise.all([
    db
      .prepare(
        `SELECT COUNT(*) AS cnt FROM (
           SELECT created_at FROM implicit_signals
           UNION ALL
           SELECT created_at FROM feedback
           UNION ALL
           SELECT created_at FROM article_feedback_forms
         )
         WHERE created_at >= date('now', 'start of month')`
      )
      .first(),
    db
      .prepare(
        `SELECT island, COUNT(*) AS count FROM (
           SELECT island FROM implicit_signals
           UNION ALL
           SELECT island FROM feedback
           UNION ALL
           SELECT island FROM article_feedback_forms
         )
         WHERE island IS NOT NULL
         GROUP BY island
         ORDER BY count DESC`
      )
      .all(),
    db
      .prepare(
        `SELECT COUNT(*) AS cnt
         FROM article_feedback_forms
         WHERE created_at >= date('now', 'start of month')`
      )
      .first(),
    db
      .prepare(
        `SELECT COUNT(*) AS cnt
         FROM article_feedback_forms
         WHERE correction_text IS NOT NULL
           AND TRIM(correction_text) != ''
           AND created_at >= date('now', 'start of month')`
      )
      .first(),
    db
      .prepare(
        `SELECT
           SUM(CASE WHEN helpful_score = 1 THEN 1 ELSE 0 END) AS helpful_yes,
           SUM(CASE WHEN helpful_score = 0 THEN 1 ELSE 0 END) AS helpful_no
         FROM article_feedback_forms
         WHERE created_at >= date('now', 'start of month')`
      )
      .first(),
    db
      .prepare(
        `SELECT mode_preference AS mode, COUNT(*) AS count
         FROM article_feedback_forms
         WHERE created_at >= date('now', 'start of month')
         GROUP BY mode_preference
         ORDER BY count DESC`
      )
      .all(),
  ]);

  return {
    total_this_month: (total as any)?.cnt ?? 0,
    islands: islands as unknown as { island: string; count: number }[],
    article_feedback_count: (articleFeedback as any)?.cnt ?? 0,
    corrections_count: (corrections as any)?.cnt ?? 0,
    helpful_yes: (helpful as any)?.helpful_yes ?? 0,
    helpful_no: (helpful as any)?.helpful_no ?? 0,
    mode_preferences: modePreferences as unknown as {
      mode: "tv" | "tv+en" | "en";
      count: number;
    }[],
  };
}

// Lightweight version for the teaser bar — single query, no islands breakdown
export async function getFateleTeaserCount(): Promise<number> {
  const db = await getDb();
  await ensureCommunitySchema(db);
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS cnt FROM (
         SELECT created_at FROM implicit_signals
         UNION ALL
         SELECT created_at FROM feedback
         UNION ALL
         SELECT created_at FROM article_feedback_forms
       )
       WHERE created_at >= date('now', 'start of month')`
    )
    .first();
  return (row as any)?.cnt ?? 0;
}
