# Talafutipolo Latency Optimization

Complete latency analysis and optimization plan for talafutipolo.pages.dev,
targeting users in Tuvalu on low-speed connections and low-powered devices.

## Network topology

```
User in Funafuti
    │
    │  ~20-30ms RTT (Vaka submarine cable, Oct 2025)
    │  [Before cable: 600ms satellite / 50ms Starlink]
    ▼
Cloudflare Suva PoP (Fiji)
    │
    │  ~0ms cold start (V8 isolates, pre-warmed on TLS handshake)
    ▼
Cloudflare Pages Worker (SSR)
    │
    │  ~10-30ms if D1 read replica in Oceania
    │  ~150-200ms if routed to US/EU primary
    ▼
D1 SQLite Database
    │
    │  Query execution: <1ms (simple), 2-5ms (JOIN + LIKE)
    ▼
Worker renders HTML (SolidStart SSR)
    │
    │  ~20-30ms RTT back to user
    ▼
Browser receives HTML, fetches CSS/JS from edge cache
    │
    │  ~0ms (same PoP, cached after first visit)
    ▼
SolidJS hydration → interactive
```

## Full page load timeline (estimated, Funafuti)

### First visit (cold cache)

| Phase | Duration | Cumulative | Notes |
|-------|----------|------------|-------|
| DNS resolution | 50-100ms | 100ms | First visit to pages.dev domain |
| TLS handshake | 40-60ms | 160ms | 1.5 RTT (TLS 1.3) |
| HTTP request → Worker | 20-30ms | 190ms | One-way to Suva PoP |
| Worker boots | ~0ms | 190ms | Pre-warmed during TLS handshake |
| D1 queries (parallel) | 10-30ms | 220ms | With Oceania read replica |
| SSR render | 5-10ms | 230ms | SolidJS compiles to string fast |
| Response transit | 20-30ms | 260ms | HTML back to client |
| **TTFB** | | **~260ms** | |
| Parse HTML + fetch CSS/JS | 20-30ms | 290ms | 103 Early Hints help here |
| CSS paint (FCP) | 10-20ms | 310ms | SSR HTML renders immediately |
| JS download | 50-100ms | 410ms | ~40KB gzipped on slow link |
| Hydration | 20-50ms | 460ms | SolidJS hydration is fast |
| **TTI** | | **~460ms** | |
| Hero image loads | 200-500ms | 960ms | External CDN, DNS prefetched |
| **LCP** | | **~960ms** | |

### Repeat visit (warm cache)

| Phase | Duration | Cumulative | Notes |
|-------|----------|------------|-------|
| Service worker intercept | 0ms | 0ms | Cache-first for assets |
| HTML (network-first) | 260ms | 260ms | Still hits worker for fresh data |
| CSS/JS from SW cache | 0ms | 260ms | Instant, no network |
| **TTI** | | **~280ms** | Near-instant hydration |
| Images from SW cache | 0ms | 280ms | Cached on first view |
| **LCP** | | **~280ms** | |

### Offline

| Phase | Duration | Notes |
|-------|----------|-------|
| Service worker serves cached HTML | <10ms | Previously visited pages |
| All assets from cache | <10ms | JS, CSS, images |
| **Full page** | **<20ms** | Completely offline-capable |

## D1 query analysis per route

### Before optimization

| Route | Queries | Pattern | Problem |
|-------|---------|---------|---------|
| `/` | 4 | articles, categories (serial), teaser count, teaser islands | Sequential, teaser wastes 1 query |
| `/articles/:id` | 3 | article, teaser count, teaser islands | Teaser wastes 1 query |
| `/category/:slug` | 4 | articles, categories (serial), teaser count, teaser islands | Sequential, teaser wastes 1 query |
| `/search` | 3 | search, teaser count, teaser islands | Teaser wastes 1 query |
| `/fatele` | 4 | count, islands, teaser count, teaser islands | Duplicate: teaser re-runs both |

### After optimization

| Route | Queries | Pattern | Improvement |
|-------|---------|---------|-------------|
| `/` | 3 | articles + categories (parallel), teaser count | -1 query, parallel |
| `/articles/:id` | 2 | article, teaser count | -1 query |
| `/category/:slug` | 3 | articles + categories (parallel), teaser count | -1 query, parallel |
| `/search` | 2 | search, teaser count | -1 query |
| `/fatele` | 3 | count + islands (parallel), teaser count | -1 query, parallel |

**Impact:** Every page load saves 1 D1 round-trip (~10-30ms). List pages save an
additional sequential wait by parallelizing (~10-30ms more).

## What we optimized (implemented)

### Bundle size (biggest impact)

| Asset | Before | After | Savings |
|-------|--------|-------|---------|
| Main JS bundle | 998 KB | 93 KB | 91% smaller |
| Main JS gzipped | 326 KB | 29 KB | 91% smaller |
| Homepage HTML | 174 KB | 52 KB | 70% smaller |
| Total dist/ | 3.8 MB | 1.6 MB | 58% smaller |

**How:**
- `highlight.js`: full 194-language bundle → `highlight.js/lib/core` + 5 languages
- SQL queries: `ARTICLE_LIST_SELECT` excludes `body_en`/`body_tvl` on list pages
  (bodies were being serialized into SSR hydration scripts even though only titles are shown)

### Network round-trips

| Optimization | Savings |
|-------------|---------|
| DNS prefetch + preconnect for 3 image CDNs | 2-3 RTT on first image load |
| `_headers` file with immutable cache-control | Browser skips revalidation |
| Service worker precaches `/chat` route | Instant on repeat visit |
| 103 Early Hints (automatic on Cloudflare Pages) | CSS/JS fetch starts 1 RTT earlier |

### Rendering performance

| Optimization | Impact |
|-------------|--------|
| Hero image `loading="eager"` + `fetchpriority="high"` | LCP improvement |
| Thumbnail images `decoding="async"` | Non-blocking image decode |
| System fonts only (no Google Fonts) | Zero font-related latency |

### Server-side

| Optimization | Impact |
|-------------|--------|
| `Promise.all()` for homepage articles + categories | Parallel D1 queries, saves ~10-30ms |
| `Promise.all()` for category page queries | Same parallel savings |
| `Promise.all()` for fatele stats queries | Same parallel savings |
| `getFateleTeaserCount()` — single COUNT query | Was running 2 queries (count + islands GROUP BY), only using count |
| Fix wrangler externalization (pre-existing build error) | Build works again |

## What to optimize next (not yet implemented)

### High impact

#### 1. Enable D1 Global Read Replication
D1's primary is in one region (likely US). Without read replication,
every query from Suva PoP must cross the Pacific (~150-200ms RTT).
With Oceania read replica, queries route to nearest replica (~10-30ms).

**How:** Cloudflare Dashboard → D1 → talafutipolo → Settings → Enable Read Replication.
No code changes needed. Reads automatically route to nearest replica.

**Estimated savings:** 120-170ms per page load.

#### 2. Worker-level Cache API for D1 results
Articles change infrequently. Cache D1 query results at the edge PoP
using the Workers Cache API with a 5-minute TTL.

```typescript
// Pseudocode for the pattern
async function cachedGetArticles(limit, offset, category) {
  const cacheKey = new Request(`https://cache.internal/articles?l=${limit}&o=${offset}&c=${category}`);
  const cache = caches.default;
  const cached = await cache.match(cacheKey);
  if (cached) return cached;

  const data = await getArticles(limit, offset, category);
  const response = new Response(JSON.stringify(data), {
    headers: { 'Cache-Control': 'public, max-age=300' } // 5 min
  });
  ctx.waitUntil(cache.put(cacheKey, response.clone()));
  return response;
}
```

**Estimated savings:** Eliminates D1 latency entirely on cache hit (~10-200ms).
Cache is per-PoP, so Suva users share a warm cache.

#### 3. Smart Placement
Cloudflare Smart Placement automatically runs the Worker close to the
D1 primary (instead of at the edge). This helps when read replication is
not enabled — the Worker and D1 are co-located, eliminating the
Worker→D1 RTT at the cost of a slightly longer User→Worker RTT.

**How:** `wrangler.toml` → `[placement] mode = "smart"`

**Trade-off:** Only useful if NOT using read replication. With read replicas,
edge placement (default) is better.

### Medium impact

#### 4. Streaming SSR
SolidStart supports `renderToStream`. The shell (header, nav, skeleton)
flushes immediately while D1 queries resolve, then data streams in via
inline `<script>` tags that fill Suspense boundaries.

**Impact:** TTFB drops to ~190ms (shell) instead of ~260ms (full render).
User sees the page skeleton while data loads.

#### 5. Stale-while-revalidate pattern
Serve cached D1 results immediately, revalidate in background via
`ctx.waitUntil()`. User always gets instant response; data is at most
5 minutes stale (acceptable for football news).

#### 6. Preload next page on hover/touch
When user hovers over an article link (or `touchstart` on mobile),
prefetch the article data. By the time they tap, the data is ready.

```typescript
// In ArticleCard.tsx
onMouseEnter={() => loadArticle(article.id)}
```

SolidStart's `cache()` will deduplicate the actual navigation request.

### Low impact (diminishing returns)

#### 7. Image optimization
External images from Goal.com/FIFA.com/Sky Sports are served as-is.
Consider proxying through Cloudflare Images or a Worker that serves
WebP/AVIF with `Accept` header negotiation and resize.

#### 8. Critical CSS inlining
Inline the above-the-fold CSS in a `<style>` tag to eliminate the
CSS render-blocking request. Difficult with Tailwind (utility classes
are unpredictable), but could inline the CSS custom properties and
basic layout rules (~1KB).

#### 9. HTTP/3 (QUIC)
Cloudflare Pages serves HTTP/3 automatically. Users on modern browsers
benefit from 0-RTT connection resumption (saves ~20-30ms on repeat visits).
No action needed — already enabled.

## Payload budget

Target: **<100KB total transfer** for homepage on first visit (gzipped).

| Asset | Gzipped | Status |
|-------|---------|--------|
| HTML (SSR) | ~15KB | ✅ (52KB raw, gzips to ~15KB) |
| CSS | 6.4KB | ✅ |
| JS (framework + route) | ~40KB | ✅ (client + routing + home route) |
| Icons + manifest | ~3KB | ✅ |
| **Total** | **~65KB** | **✅ Under budget** |

Chat route adds 29KB gzipped (marked + hljs core + 5 langs) — only loaded
when visiting `/chat`, never on the main site.

## Service worker strategy

```
┌─────────────────────────────────────────────────┐
│                  Fetch Event                     │
├─────────────────────────────────────────────────┤
│                                                  │
│  /_build/* or static asset?                      │
│     YES → Cache-first (immutable, forever)       │
│                                                  │
│  text/html?                                      │
│     YES → Network-first, cache fallback          │
│            (fresh data, offline capable)          │
│                                                  │
│  External image CDN?                             │
│     YES → Cache-first (cache on first view)      │
│                                                  │
│  /api/*?                                         │
│     YES → Skip SW (always network)               │
│                                                  │
│  Everything else → Network                       │
└─────────────────────────────────────────────────┘
```

Precached on install: `/`, `/fatele`, `/search`, `/chat`

## Device performance considerations

Tuvalu users likely have mid-range Android devices (2-4GB RAM).
SolidJS is well-suited here:
- No virtual DOM diffing (direct DOM updates)
- Tiny runtime (~7KB)
- Fine-grained reactivity (no unnecessary re-renders)
- SSR hydration is lightweight (marks existing DOM, doesn't recreate)

The 48px touch targets in CSS ensure usability on budget phones.
System font stack avoids font download and FOIT/FOUT entirely.
