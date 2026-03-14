import type { Article, Category, FeedbackSubmission, SignalSubmission, FateleStats } from "./types";

// D1 binding is injected by Cloudflare Workers runtime
// In dev mode, lazily initialize via wrangler's getPlatformProxy()
let _devProxy: any = null;

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
  await db
    .prepare(
      `INSERT INTO implicit_signals (article_id, signal_type, paragraph_index, session_id, island)
       VALUES (?, ?, ?, ?, ?)`
    )
    .bind(sig.article_id, sig.signal_type, sig.paragraph_index ?? null, sig.session_id ?? null, sig.island ?? null)
    .run();
}

export async function getFateleStats(): Promise<FateleStats> {
  const db = await getDb();
  const [total, { results: islands }] = await Promise.all([
    db
      .prepare(
        `SELECT COUNT(*) AS cnt FROM implicit_signals
         WHERE created_at >= date('now', 'start of month')`
      )
      .first(),
    db
      .prepare(
        `SELECT island, COUNT(*) AS count FROM implicit_signals
         WHERE island IS NOT NULL
         GROUP BY island
         ORDER BY count DESC`
      )
      .all(),
  ]);

  return {
    total_this_month: (total as any)?.cnt ?? 0,
    islands: islands as unknown as { island: string; count: number }[],
  };
}

// Lightweight version for the teaser bar — single query, no islands breakdown
export async function getFateleTeaserCount(): Promise<number> {
  const db = await getDb();
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS cnt FROM implicit_signals
       WHERE created_at >= date('now', 'start of month')`
    )
    .first();
  return (row as any)?.cnt ?? 0;
}
