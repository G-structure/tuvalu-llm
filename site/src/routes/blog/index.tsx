import { A, createAsync, cache } from "@solidjs/router";
import { For, Show } from "solid-js";
import PostCard from "~/components/blog/PostCard";
import OGMeta from "~/components/OGMeta";
import StructuredData from "~/components/StructuredData";
import {
  getAllPosts,
  getAllTags,
  getArchiveGroups,
  getFeaturedPost,
  getRecentPosts,
  type BlogArchiveGroup,
  type BlogTagIndex,
} from "~/lib/blog-data";
import type { BlogPost } from "~/lib/blog";
import { absoluteUrl, SITE_META } from "~/lib/site";

const loadBlogHome = cache(async (): Promise<{
  posts: BlogPost[];
  featured: BlogPost | null;
  latest: BlogPost[];
  tags: BlogTagIndex[];
  archive: BlogArchiveGroup[];
}> => {
  "use server";
  const posts = getAllPosts();
  return {
    posts,
    featured: getFeaturedPost(),
    latest: getRecentPosts(6),
    tags: getAllTags(),
    archive: getArchiveGroups(),
  };
}, "blog-home");

export const route = {
  load: () => loadBlogHome(),
};

export default function BlogIndex() {
  const data = createAsync(() => loadBlogHome());

  return (
    <Show when={data()}>
      {(blog) => {
        const secondaryPosts = blog().latest.filter(
          (post) => post.slug !== blog().featured?.slug
        );

        const collectionStructuredData = {
          "@context": "https://schema.org",
          "@type": "CollectionPage",
          name: SITE_META.publicationName,
          description: SITE_META.publicationDescription,
          url: absoluteUrl("/blog"),
          isPartOf: {
            "@type": "WebSite",
            name: SITE_META.publicationShortName,
            url: absoluteUrl("/"),
          },
          mainEntity: {
            "@type": "Blog",
            name: SITE_META.publicationName,
            url: absoluteUrl("/blog"),
            blogPost: blog().posts.map((post) => ({
              "@type": "BlogPosting",
              headline: post.title,
              url: absoluteUrl(`/blog/${post.slug}`),
              datePublished: post.publishedAt,
              description: post.description,
            })),
          },
        };

        return (
          <main class="blog-page blog-page--index">
            <OGMeta
              title={SITE_META.publicationName}
              description={SITE_META.publicationDescription}
              url={absoluteUrl("/blog")}
              image={SITE_META.defaultOgImage}
              imageWidth={SITE_META.defaultOgImageWidth}
              imageHeight={SITE_META.defaultOgImageHeight}
              imageAlt={SITE_META.defaultOgImageAlt}
              siteName={SITE_META.publicationShortName}
              titleSuffix={SITE_META.publicationShortName}
              keywords={blog().tags.map((tag) => tag.name)}
              alternateLinks={[
                {
                  href: SITE_META.feeds.blogRss,
                  title: `${SITE_META.publicationName} RSS`,
                  type: "application/rss+xml",
                },
                {
                  href: SITE_META.feeds.blogJson,
                  title: `${SITE_META.publicationName} JSON Feed`,
                  type: "application/feed+json",
                },
              ]}
            />
            <StructuredData data={collectionStructuredData} />

            <section class="blog-hero">
              <div class="blog-shell blog-shell--wide blog-hero__grid">
                <div class="blog-hero__copy">
                  <p class="blog-kicker">Language Lab Journal</p>
                  <h1 class="blog-hero__title">
                    Tuvaluan AI, built to be used.
                  </h1>
                  <p class="blog-hero__lede">
                    Product launches, research updates, field reports, and open-source
                    work from the team building practical language tools for Tuvaluan
                    speakers, learners, and communities.
                  </p>
                  <div class="blog-hero__actions">
                    <A href="/chat" class="blog-button blog-button--primary">
                      Try the product
                    </A>
                    <A href="/demo" class="blog-button blog-button--ghost">
                      Project overview
                    </A>
                    <a href={SITE_META.feeds.blogRss} class="blog-button blog-button--ghost">
                      RSS
                    </a>
                  </div>
                </div>

                <div class="blog-hero__meta">
                  <div class="blog-stat-card">
                    <span class="blog-stat-card__label">Posts</span>
                    <strong class="blog-stat-card__value">{blog().posts.length}</strong>
                    <p class="blog-stat-card__detail">
                      Launches, deep dives, field reports, and product updates.
                    </p>
                  </div>
                  <div class="blog-stat-card">
                    <span class="blog-stat-card__label">Topics</span>
                    <strong class="blog-stat-card__value">{blog().tags.length}</strong>
                    <p class="blog-stat-card__detail">
                      Model quality, usability, deployment, and open-source work.
                    </p>
                  </div>
                  <div class="blog-stat-card">
                    <span class="blog-stat-card__label">Focus</span>
                    <strong class="blog-stat-card__value">Tuvaluan AI</strong>
                    <p class="blog-stat-card__detail">
                      Consumer-facing language tools backed by serious research.
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <div class="blog-shell blog-shell--wide blog-home">
              <Show when={blog().featured}>
                {(featured) => (
                  <section class="blog-home__feature">
                    <div class="blog-home__section-head">
                      <div>
                        <p class="blog-section-kicker">Featured</p>
                        <h2 class="blog-section-title">Flagship story</h2>
                      </div>
                      <A href="/blog/archive" class="blog-text-link">
                        Full archive
                      </A>
                    </div>
                    <PostCard post={featured()} variant="feature" />
                  </section>
                )}
              </Show>

              <div class="blog-home__layout">
                <section class="blog-home__primary">
                  <Show
                    when={secondaryPosts.length > 0}
                    fallback={
                      <section class="blog-panel">
                        <p class="blog-panel__eyebrow">Start here</p>
                        <h2 class="blog-panel__title">
                          Follow how Language Lab is turning research into a real product.
                        </h2>
                        <p class="blog-panel__text">
                          The journal tracks the work behind a Tuvaluan assistant people
                          can actually use, from model quality and evaluation to launches
                          and deployment.
                        </p>
                      </section>
                    }
                  >
                    <div class="blog-home__section-head">
                      <div>
                        <p class="blog-section-kicker">Latest</p>
                        <h2 class="blog-section-title">Latest from Language Lab</h2>
                      </div>
                    </div>

                    <div class="blog-post-list">
                      <For each={secondaryPosts}>
                        {(post) => <PostCard post={post} variant="list" />}
                      </For>
                    </div>
                  </Show>
                </section>

                <aside class="blog-home__sidebar">
                  <section class="blog-panel">
                    <p class="blog-panel__eyebrow">Inside Language Lab</p>
                    <h2 class="blog-panel__title">Product, research, and field reality.</h2>
                    <ul class="blog-panel__list">
                      <li>Product launches and updates for the Tuvaluan assistant.</li>
                      <li>Research notes that explain quality, evaluation, and tradeoffs clearly.</li>
                      <li>Technical deep dives for readers who want code, benchmarks, and failure cases.</li>
                      <li>Field notes on deployment, adoption, and real-world use.</li>
                    </ul>
                  </section>

                  <section class="blog-panel">
                    <div class="blog-home__section-head blog-home__section-head--tight">
                      <div>
                        <p class="blog-panel__eyebrow">Topics</p>
                        <h2 class="blog-panel__title">Browse by beat</h2>
                      </div>
                    </div>
                    <div class="blog-tag-cloud">
                      <For each={blog().tags}>
                        {(tag) => (
                          <A href={`/blog/tag/${tag.slug}`} class="blog-topic-chip">
                            <span>{tag.name}</span>
                            <span>{tag.count}</span>
                          </A>
                        )}
                      </For>
                    </div>
                  </section>

                  <section class="blog-panel">
                    <div class="blog-home__section-head blog-home__section-head--tight">
                      <div>
                        <p class="blog-panel__eyebrow">Archive</p>
                        <h2 class="blog-panel__title">Publication history</h2>
                      </div>
                    </div>
                    <div class="blog-archive-list">
                      <For each={blog().archive}>
                        {(group) => (
                          <div class="blog-archive-list__item">
                            <span>{group.label}</span>
                            <span>{group.posts.length}</span>
                          </div>
                        )}
                      </For>
                    </div>
                  </section>

                  <section class="blog-panel blog-panel--cta">
                    <p class="blog-panel__eyebrow">Go hands-on</p>
                    <h2 class="blog-panel__title">Read it, then try it.</h2>
                    <p class="blog-panel__text">
                      Open the live assistant, see how the product works, and follow the
                      work as it gets better.
                    </p>
                    <div class="blog-panel__actions">
                      <A href="/chat" class="blog-button blog-button--primary">
                        Try the product
                      </A>
                      <A href="/demo" class="blog-button blog-button--ghost">
                        Project overview
                      </A>
                    </div>
                  </section>
                </aside>
              </div>
            </div>
          </main>
        );
      }}
    </Show>
  );
}
