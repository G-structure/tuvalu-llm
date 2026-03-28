import type { APIEvent } from "@solidjs/start/server";

async function getDb(): Promise<D1Database> {
  const db = (process.env as any).DB || (globalThis as any).__env__?.DB;
  if (db) return db;
  const { getPlatformProxy } = await import("wrangler");
  const proxy = await getPlatformProxy({ persist: { path: ".wrangler/state/v3" } });
  (globalThis as any).__env__ = proxy.env;
  return proxy.env.DB;
}

async function ensureTable(db: D1Database) {
  await db
    .prepare(
      `CREATE TABLE IF NOT EXISTS newsletter_signups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      )`
    )
    .run();
}

export async function POST(event: APIEvent) {
  try {
    const body = await event.request.json();
    const email = (body.email || "").trim().toLowerCase();

    if (!email || !email.includes("@") || email.length > 320) {
      return new Response(JSON.stringify({ error: "Invalid email" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const db = await getDb();
    await ensureTable(db);

    await db
      .prepare("INSERT OR IGNORE INTO newsletter_signups (email) VALUES (?)")
      .bind(email)
      .run();

    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return new Response(JSON.stringify({ error: "Server error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
