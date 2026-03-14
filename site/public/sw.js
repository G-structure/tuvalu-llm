const CACHE_VERSION = "talafutipolo-v1";
const STATIC_CACHE = CACHE_VERSION + "-static";
const PAGES_CACHE = CACHE_VERSION + "-pages";

// Static assets to pre-cache on install
const PRECACHE_URLS = ["/", "/fatele", "/search", "/chat"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(STATIC_CACHE)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => !k.startsWith(CACHE_VERSION))
            .map((k) => caches.delete(k))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Skip non-GET and API/signal requests
  if (event.request.method !== "GET") return;
  if (url.pathname.startsWith("/api/")) return;

  // Static assets (JS, CSS, images): cache-first
  if (
    url.pathname.startsWith("/_build/") ||
    url.pathname.match(/\.(js|css|woff2?|png|jpg|jpeg|webp|svg|ico)$/)
  ) {
    event.respondWith(
      caches.match(event.request).then(
        (cached) =>
          cached ||
          fetch(event.request).then((response) => {
            if (response.ok) {
              const clone = response.clone();
              caches.open(STATIC_CACHE).then((cache) => cache.put(event.request, clone));
            }
            return response;
          })
      )
    );
    return;
  }

  // HTML pages (articles, homepage): network-first, cache fallback
  if (event.request.headers.get("accept")?.includes("text/html")) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(PAGES_CACHE).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() =>
          caches.match(event.request).then(
            (cached) =>
              cached ||
              new Response(
                "<html><body style='font-family:system-ui;text-align:center;padding:4rem'>" +
                  "<h1>Seki isi te initaneti</h1>" +
                  "<p>You are offline. Previously visited articles are available.</p>" +
                  "<p><a href='/'>Foki ki te laupepa muaki</a></p>" +
                  "</body></html>",
                { status: 503, headers: { "Content-Type": "text/html" } }
              )
          )
        )
    );
    return;
  }

  // Article images: cache on first view
  if (url.hostname.includes("365dm.com") ||
      url.hostname.includes("assets.goal.com") ||
      url.hostname.includes("digitalhub.fifa.com")) {
    event.respondWith(
      caches.match(event.request).then(
        (cached) =>
          cached ||
          fetch(event.request).then((response) => {
            if (response.ok) {
              const clone = response.clone();
              caches.open(STATIC_CACHE).then((cache) => cache.put(event.request, clone));
            }
            return response;
          })
      )
    );
    return;
  }
});
