import { A, createAsync, cache, useParams } from "@solidjs/router";
import { For, Show } from "solid-js";
import { HttpStatusCode } from "@solidjs/start";
import PostCard from "~/components/blog/PostCard";
import OGMeta from "~/components/OGMeta";
import StructuredData from "~/components/StructuredData";
import { getAllTags, getPostsByTag } from "~/lib/blog-data";
import { absoluteUrl, BLOG_SECTION_OG_IMAGES, SITE_META } from "~/lib/site";

const loadTagPage = cache(async (slug: string) => {
  "use server";
  const tags = getAllTags();
  const tag = tags.find((item) => item.slug === slug) || null;
  if (!tag) return null;
  return {
    tag,
    posts: getPostsByTag(slug),
    tags,
  };
}, "blog-tag-page");

export const route = {
  load: ({ params }: { params: { slug: string } }) => loadTagPage(params.slug),
};

export default function BlogTagPage() {
  const params = useParams<{ slug: string }>();
  const data = createAsync(() => loadTagPage(params.slug));

  return (
    <Show when={data()} fallback={<TagNotFound />}>
      {(page) => {
        const title = `${page().tag.name} — ${SITE_META.publicationName}`;
        const url = absoluteUrl(`/blog/tag/${page().tag.slug}`);
        const image =
          BLOG_SECTION_OG_IMAGES.tags[
            page().tag.slug as keyof typeof BLOG_SECTION_OG_IMAGES.tags
          ] || SITE_META.defaultOgImage;

        return (
          <main class="blog-page blog-page--taxonomy">
            <OGMeta
              title={title}
              description={`Posts filed under ${page().tag.name} in the ${SITE_META.publicationName}.`}
              url={url}
              image={image}
              imageWidth={SITE_META.defaultOgImageWidth}
              imageHeight={SITE_META.defaultOgImageHeight}
              imageAlt={`Language Lab Journal social card for ${page().tag.name} posts.`}
              siteName={SITE_META.publicationShortName}
              titleSuffix={SITE_META.publicationShortName}
              keywords={[page().tag.name]}
            />
            <StructuredData
              data={{
                "@context": "https://schema.org",
                "@type": "CollectionPage",
                name: title,
                url,
                description: `Posts filed under ${page().tag.name}.`,
              }}
            />

            <div class="blog-shell blog-shell--wide">
              <section class="blog-taxonomy-hero">
                <p class="blog-kicker">Topic archive</p>
                <h1 class="blog-section-title">{page().tag.name}</h1>
                <p class="blog-panel__text">
                  {page().posts.length} post{page().posts.length === 1 ? "" : "s"} in this beat.
                </p>
                <div class="blog-taxonomy-chips">
                  <For each={page().tags}>
                    {(tag) => (
                      <A
                        href={`/blog/tag/${tag.slug}`}
                        class={`blog-topic-chip ${tag.slug === page().tag.slug ? "blog-topic-chip--active" : ""}`}
                      >
                        <span>{tag.name}</span>
                        <span>{tag.count}</span>
                      </A>
                    )}
                  </For>
                </div>
              </section>

              <section class="blog-post-list blog-post-list--spacious">
                <For each={page().posts}>
                  {(post) => <PostCard post={post} variant="list" />}
                </For>
              </section>
            </div>
          </main>
        );
      }}
    </Show>
  );
}

function TagNotFound() {
  return (
    <main class="blog-page blog-page--taxonomy">
      <HttpStatusCode code={404} />
      <div class="blog-shell blog-shell--narrow blog-empty-state">
        <p class="blog-kicker">Language Lab Journal</p>
        <h1 class="blog-section-title">Topic not found.</h1>
        <A href="/blog" class="blog-button blog-button--primary">
          Back to the blog
        </A>
      </div>
    </main>
  );
}
