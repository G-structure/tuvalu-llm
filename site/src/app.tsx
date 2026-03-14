import { MetaProvider, Link } from "@solidjs/meta";
import { Router, useLocation } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { Suspense, Show, onMount } from "solid-js";
import { isServer } from "solid-js/web";
import Header from "~/components/Header";
import IslandSelector from "~/components/IslandSelector";
import FateleTeaser from "~/components/FateleTeaser";
import { registerServiceWorker } from "~/lib/register-sw";
import "./app.css";

function Shell(props: { children: any }) {
  const location = useLocation();
  const isChatRoute = () => location.pathname.startsWith("/chat");

  return (
    <MetaProvider>
      <Show when={!isChatRoute()}>
        <Link rel="alternate" type="application/rss+xml" title="TALAFUTIPOLO RSS" href="/feed.xml" />
      </Show>
      <Show
        when={!isChatRoute()}
        fallback={
          <Suspense>{props.children}</Suspense>
        }
      >
        <div class="min-h-screen pb-12">
          <Header />
          <Suspense
            fallback={
              <div class="max-w-3xl mx-auto p-4 text-center text-gray-400">
                Loading...
              </div>
            }
          >
            {props.children}
          </Suspense>
          <FateleTeaser />
          <IslandSelector />
        </div>
      </Show>
    </MetaProvider>
  );
}

export default function App() {
  onMount(() => {
    if (!isServer) registerServiceWorker();
  });

  return (
    <Router root={(props) => <Shell>{props.children}</Shell>}>
      <FileRoutes />
    </Router>
  );
}
