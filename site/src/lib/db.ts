import Database from "better-sqlite3";
import { join, resolve } from "path";
import type { Article, Category, FeedbackSubmission, SignalSubmission, FateleStats } from "./types";

// process.cwd() is site/ in both dev and production
const DB_PATH = resolve(join(process.cwd(), "..", "data", "football", "football.db"));

let db: Database.Database | null = null;
let dbWrite: Database.Database | null = null;

function getDb(): Database.Database {
  if (!db) {
    db = new Database(DB_PATH, { readonly: true });
    db.pragma("journal_mode = WAL");
  }
  return db;
}

function getWriteDb(): Database.Database {
  if (!dbWrite) {
    dbWrite = new Database(DB_PATH, { readonly: false });
    dbWrite.pragma("journal_mode = WAL");
  }
  return dbWrite;
}

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

export function getArticles(
  limit = 20,
  offset = 0,
  category?: string
): Article[] {
  const conn = getDb();
  if (category) {
    const stmt = conn.prepare(
      `${ARTICLE_SELECT}
       WHERE a.category = ?
       ORDER BY a.published_at DESC
       LIMIT ? OFFSET ?`
    );
    return stmt.all(category, limit, offset) as Article[];
  }
  const stmt = conn.prepare(
    `${ARTICLE_SELECT}
     ORDER BY a.published_at DESC
     LIMIT ? OFFSET ?`
  );
  return stmt.all(limit, offset) as Article[];
}

export function getArticle(id: string): Article | undefined {
  const conn = getDb();
  const stmt = conn.prepare(`${ARTICLE_SELECT} WHERE a.id = ?`);
  return stmt.get(id) as Article | undefined;
}

export function getCategories(): Category[] {
  const conn = getDb();
  const stmt = conn.prepare(`
    SELECT category AS slug, COUNT(*) AS count
    FROM articles
    WHERE category IS NOT NULL AND category != ''
    GROUP BY category
    ORDER BY count DESC
  `);
  return stmt.all() as Category[];
}

export function getArticleCount(category?: string): number {
  const conn = getDb();
  if (category) {
    const stmt = conn.prepare(
      "SELECT COUNT(*) AS cnt FROM articles WHERE category = ?"
    );
    return (stmt.get(category) as { cnt: number }).cnt;
  }
  const stmt = conn.prepare("SELECT COUNT(*) AS cnt FROM articles");
  return (stmt.get() as { cnt: number }).cnt;
}

export function insertFeedback(fb: FeedbackSubmission): void {
  const conn = getWriteDb();
  conn.prepare(
    `INSERT INTO feedback (article_id, paragraph_idx, feedback_type, island, session_id)
     VALUES (?, ?, ?, ?, ?)`
  ).run(fb.article_id, fb.paragraph_idx, fb.feedback_type, fb.island ?? null, fb.session_id ?? null);
}

export function insertSignal(sig: SignalSubmission): void {
  const conn = getWriteDb();
  conn.prepare(
    `INSERT INTO implicit_signals (article_id, signal_type, paragraph_index, session_id, island)
     VALUES (?, ?, ?, ?, ?)`
  ).run(sig.article_id, sig.signal_type, sig.paragraph_index ?? null, sig.session_id ?? null, sig.island ?? null);
}

export function getFateleStats(): FateleStats {
  const conn = getDb();
  const total = conn.prepare(
    `SELECT COUNT(*) AS cnt FROM implicit_signals
     WHERE created_at >= date('now', 'start of month')`
  ).get() as { cnt: number };

  const islands = conn.prepare(
    `SELECT island, COUNT(*) AS count FROM implicit_signals
     WHERE island IS NOT NULL
     GROUP BY island
     ORDER BY count DESC`
  ).all() as { island: string; count: number }[];

  return {
    total_this_month: total.cnt,
    islands,
  };
}
