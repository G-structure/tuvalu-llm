import { Link, Meta, MetaProvider } from "@solidjs/meta";
import { Router, useLocation } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { Suspense, Show, onMount } from "solid-js";
import { isServer } from "solid-js/web";
import Header from "~/components/Header";
import IslandSelector from "~/components/IslandSelector";
import SiteFooter from "~/components/SiteFooter";
import { registerServiceWorker } from "~/lib/register-sw";
import { SITE_META } from "~/lib/site";
import "./app.css";

function Shell(props: { children: any }) {
  const location = useLocation();
  const isChatRoute = () => location.pathname.startsWith("/chat");
  const isBlogRoute = () => location.pathname.startsWith("/blog");

  return (
    <MetaProvider>
      <Meta name="theme-color" content="#013A63" />
      <Meta name="color-scheme" content="light" />
      <Meta name="application-name" content="Fenua Intelligence" />
      <Meta name="apple-mobile-web-app-capable" content="yes" />
      <Meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
      <Meta name="apple-mobile-web-app-title" content="Fenua AI" />
      <Meta name="mobile-web-app-capable" content="yes" />
      <Show when={!isChatRoute()}>
        <>
          <Link rel="preconnect" href="https://fonts.googleapis.com" />
          <Link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="" />
          <Link
            rel="stylesheet"
            href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Space+Grotesk:wght@400;500;700&display=swap"
          />
        </>
      </Show>
      <Link rel="manifest" href="/manifest.json" />
      <Link rel="icon" type="image/png" sizes="192x192" href="/icons/icon-192.png" />
      <Link rel="icon" type="image/png" sizes="512x512" href="/icons/icon-512.png" />
      <Link rel="apple-touch-icon" href="/icons/icon-512.png" />
      <Show when={!isChatRoute()}>
        <>
          <Link rel="alternate" type="application/rss+xml" title="Talafutipolo RSS" href={SITE_META.feeds.articlesRss} />
          <Show when={isBlogRoute()}>
            <>
              <Link
                rel="alternate"
                type="application/rss+xml"
                title={`${SITE_META.publicationName} RSS`}
                href={SITE_META.feeds.blogRss}
              />
              <Link
                rel="alternate"
                type="application/feed+json"
                title={`${SITE_META.publicationName} JSON Feed`}
                href={SITE_META.feeds.blogJson}
              />
            </>
          </Show>
        </>
      </Show>
      <Show
        when={!isChatRoute()}
        fallback={
          <Suspense fallback={<div class="chat-theme min-h-screen" />}>{props.children}</Suspense>
        }
      >
        <div class="site-app-shell">
          <Header />
          <Suspense>
            {props.children}
          </Suspense>
          <SiteFooter />
          <IslandSelector />
        </div>
      </Show>
    </MetaProvider>
  );
}

export default function App() {
  onMount(() => {
    if (!isServer && import.meta.env.PROD) registerServiceWorker();
  });

  return (
    <Router root={(props) => <Shell>{props.children}</Shell>}>
      <FileRoutes />
    </Router>
  );
}
