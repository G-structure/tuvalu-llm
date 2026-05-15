import { A, createAsync, cache } from "@solidjs/router";
import { For } from "solid-js";
import OGMeta from "~/components/OGMeta";
import StructuredData from "~/components/StructuredData";
import { getArchiveGroups } from "~/lib/blog-data";
import { absoluteUrl, BLOG_SECTION_OG_IMAGES, formatLongDate, SITE_META } from "~/lib/site";

const loadArchive = cache(async () => {
  "use server";
  return getArchiveGroups();
}, "blog-archive");

export const route = {
  load: () => loadArchive(),
};

export default function BlogArchivePage() {
  const archive = createAsync(() => loadArchive());

  return (
    <main class="blog-page blog-page--taxonomy">
      <OGMeta
        title={`Archive — ${SITE_META.publicationName}`}
        description={`Publication archive for the ${SITE_META.publicationName}.`}
        url={absoluteUrl("/blog/archive")}
        image={BLOG_SECTION_OG_IMAGES.archive}
        imageWidth={SITE_META.defaultOgImageWidth}
        imageHeight={SITE_META.defaultOgImageHeight}
        imageAlt="Language Lab Journal social card for the publication archive."
        siteName={SITE_META.publicationShortName}
        titleSuffix={SITE_META.publicationShortName}
      />
      <StructuredData
        data={{
          "@context": "https://schema.org",
          "@type": "CollectionPage",
          name: `${SITE_META.publicationName} archive`,
          url: absoluteUrl("/blog/archive"),
        }}
      />

      <div class="blog-shell blog-shell--wide">
        <section class="blog-taxonomy-hero">
          <p class="blog-kicker">Archive</p>
          <h1 class="blog-section-title">Publication history</h1>
          <p class="blog-panel__text">
            Every post published in the Language Lab Journal.
          </p>
        </section>

        <div class="blog-archive-groups">
          <For each={archive() || []}>
            {(group) => (
              <section class="blog-archive-group">
                <div class="blog-archive-group__year">{group.year}</div>
                <div class="blog-archive-group__list">
                  <For each={group.posts}>
                    {(post) => (
                      <A href={`/blog/${post.slug}`} class="blog-archive-post">
                        <div>
                          <span class="blog-archive-post__kind">{post.kind}</span>
                          <h2>{post.title}</h2>
                        </div>
                        <div class="blog-archive-post__meta">
                          <span>{formatLongDate(post.publishedAt)}</span>
                          <span>{post.readingTimeMinutes} min read</span>
                        </div>
                      </A>
                    )}
                  </For>
                </div>
              </section>
            )}
          </For>
        </div>
      </div>
    </main>
  );
}
