import { A, useLocation } from "@solidjs/router";
import OGMeta from "~/components/OGMeta";
import { NOINDEX_ROBOTS } from "~/lib/seo";
import { absoluteFootballUrl, FOOTBALL_META, SITE_ORIGINS } from "~/lib/site";

export default function NotFound() {
  const location = useLocation();

  return (
    <main class="max-w-3xl mx-auto p-4 text-center">
      <OGMeta
        title="Page not found"
        description="The page you're looking for doesn't exist."
        url={absoluteFootballUrl(location.pathname)}
        image={FOOTBALL_META.notFoundOgImage}
        imageOrigin={SITE_ORIGINS.football}
        imageWidth={FOOTBALL_META.defaultOgImageWidth}
        imageHeight={FOOTBALL_META.defaultOgImageHeight}
        imageAlt="Talafutipolo social card for a page that was not found."
        siteName={FOOTBALL_META.productName}
        titleSuffix={FOOTBALL_META.productName}
        robots={NOINDEX_ROBOTS}
      />
      <div class="mt-16">
        <h1 class="text-4xl font-bold text-gray-900">Seki kitea</h1>
        <p class="mt-4 text-lg text-gray-500">
          Te peesi tenei e seki kitea. The page you're looking for doesn't exist.
        </p>
        <A
          href="/"
          class="inline-block mt-8 px-6 py-3 bg-[var(--ocean-deep)] text-white text-sm font-medium rounded-lg no-underline hover:bg-[var(--ocean)] transition-colors"
        >
          &larr; Foki ki te kamata
        </A>
      </div>
    </main>
  );
}
