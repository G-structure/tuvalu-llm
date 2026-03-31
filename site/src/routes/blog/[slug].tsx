import { A, createAsync, cache, useParams } from "@solidjs/router";
import { For, onMount, Show } from "solid-js";
import { HttpStatusCode } from "@solidjs/start";
import AuthorCard from "~/components/blog/AuthorCard";
import PostCard from "~/components/blog/PostCard";
import ReadingProgress from "~/components/blog/ReadingProgress";
import ShareActions from "~/components/blog/ShareActions";
import TableOfContents from "~/components/blog/TableOfContents";
import OGMeta from "~/components/OGMeta";
import StructuredData from "~/components/StructuredData";
import type { BlogPost, BlogPostFull } from "~/lib/blog";
import { initBlogCopyButtons } from "~/lib/blog";
import { getAdjacentPosts, getPostBySlug, getRelatedPosts } from "~/lib/blog-data";
import { absoluteImageUrl, absoluteUrl, formatLongDate, SITE_META } from "~/lib/site";

const loadPostPage = cache(
  async (
    slug: string
  ): Promise<{
    post: BlogPostFull;
    related: BlogPost[];
    adjacent: { newer: BlogPost | null; older: BlogPost | null };
  } | null> => {
    "use server";
    const post = getPostBySlug(slug);
    if (!post) return null;
    return {
      post,
      related: getRelatedPosts(slug, 3),
      adjacent: getAdjacentPosts(slug),
    };
  },
  "blog-post-page"
);

export const route = {
  load: ({ params }: { params: { slug: string } }) => loadPostPage(params.slug),
};

export default function BlogPostPage() {
  const params = useParams<{ slug: string }>();
  const data = createAsync(() => loadPostPage(params.slug));

  return (
    <Show when={data()} fallback={<NotFound />}>
      {(page) => {
        const post = () => page().post;
        const url = () => post().canonicalUrl || absoluteUrl(`/blog/${post().slug}`);
        const image = () => post().socialImage || post().image || SITE_META.defaultOgImage;

        const structuredData = [
          {
            "@context": "https://schema.org",
            "@type": "BlogPosting",
            headline: post().title,
            description: post().description,
            url: url(),
            image: absoluteImageUrl(image()),
            datePublished: post().publishedAt,
            dateModified: post().updatedAt || post().publishedAt,
            articleSection: post().kind,
            keywords: post().tags,
            wordCount: post().wordCount,
            timeRequired: `PT${post().readingTimeMinutes}M`,
            isPartOf: {
              "@type": "Blog",
              name: SITE_META.publicationName,
              url: absoluteUrl("/blog"),
            },
            author: post().authors.map((author) => ({
              "@type": author.slug === "language-lab" ? "Organization" : "Person",
              name: author.name,
              url: author.href,
            })),
            publisher: {
              "@type": "Organization",
              name: SITE_META.publicationShortName,
              url: absoluteUrl("/demo"),
            },
          },
          {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            itemListElement: [
              {
                "@type": "ListItem",
                position: 1,
                name: "Blog",
                item: absoluteUrl("/blog"),
              },
              {
                "@type": "ListItem",
                position: 2,
                name: post().title,
                item: url(),
              },
            ],
          },
        ];

        return (
          <main class="blog-page blog-page--post">
            <ReadingProgress />
            <OGMeta
              title={post().title}
              description={post().description}
              image={image()}
              imageAlt={post().socialImageAlt || post().imageAlt}
              url={url()}
              type="article"
              publishedAt={post().publishedAt}
              modifiedAt={post().updatedAt || post().publishedAt}
              category={post().kind}
              siteName={SITE_META.publicationShortName}
              titleSuffix={SITE_META.publicationShortName}
              keywords={post().tags}
              authorNames={post().authors.map((author) => author.name)}
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
            <StructuredData data={structuredData} />

            <div class="blog-shell blog-shell--article">
              <div class="blog-post__backbar">
                <A href="/blog" class="blog-text-link">
                  All posts
                </A>
                <a href={SITE_META.feeds.blogRss} target="_self" class="blog-text-link">
                  RSS feed
                </a>
                <a href={SITE_META.feeds.blogJson} target="_self" class="blog-text-link">
                  JSON feed
                </a>
              </div>

              <div class="blog-post-layout">
                <article class="blog-post-main">
                  <header class="blog-post-header">
                    <div class="blog-post-header__eyebrow">
                      <span>{post().kind}</span>
                      <span class="blog-card__dot" aria-hidden="true" />
                      <span>{post().authors.map((author) => author.name).join(", ")}</span>
                    </div>
                    <h1 class="blog-post-title">{post().title}</h1>
                    <p class="blog-post-dek">{post().description}</p>

                    <div class="blog-post-meta">
                      <div class="blog-post-meta__group">
                        <span class="blog-post-meta__label">Published</span>
                        <span>{formatLongDate(post().publishedAt)}</span>
                      </div>
                      <Show when={post().updatedAt}>
                        <div class="blog-post-meta__group">
                          <span class="blog-post-meta__label">Updated</span>
                          <span>{formatLongDate(post().updatedAt)}</span>
                        </div>
                      </Show>
                      <div class="blog-post-meta__group">
                        <span class="blog-post-meta__label">Read time</span>
                        <span>{post().readingTimeMinutes} min</span>
                      </div>
                      <div class="blog-post-meta__group">
                        <span class="blog-post-meta__label">Words</span>
                        <span>{post().wordCount.toLocaleString()}</span>
                      </div>
                    </div>

                    <Show when={post().tags.length > 0}>
                      <div class="blog-post-tags">
                        <For each={post().tags}>
                          {(tag, index) => (
                            <A href={`/blog/tag/${post().tagSlugs[index()]}`} class="blog-topic-chip">
                              {tag}
                            </A>
                          )}
                        </For>
                      </div>
                    </Show>

                    <div class="blog-post-share blog-post-share--mobile">
                      <ShareActions
                        title={post().title}
                        description={post().description}
                        url={url()}
                        eyebrow="Share Language Lab"
                        heading="Know someone who cares about Tuvaluan AI?"
                        text="Send them this post from the lab."
                      />
                    </div>

                    <Show when={post().image}>
                      <figure class="blog-post-hero-image">
                        <img
                          src={post().image}
                          alt={post().imageAlt || post().title}
                          loading="eager"
                          fetchpriority="high"
                          decoding="async"
                        />
                      </figure>
                    </Show>
                  </header>

                  <div
                    class="blog-content"
                    innerHTML={post().html}
                    ref={(el: HTMLDivElement) => onMount(() => initBlogCopyButtons(el))}
                  />

                  <section class="blog-endcap">
                    <div class="blog-endcap__intro">
                      <p class="blog-section-kicker">What next</p>
                      <h2 class="blog-section-title">Try the assistant. Check the results.</h2>
                      <p class="blog-panel__text">
                        Open the live Tuvaluan assistant, inspect the evaluation results,
                        or review the code and data behind the system.
                      </p>
                    </div>
                    <div class="blog-endcap__actions">
                      <A href="/chat/eval" class="blog-button blog-button--primary">
                        See eval results
                      </A>
                      <A href="/chat" class="blog-button blog-button--ghost">
                        Try the assistant
                      </A>
                      <a
                        href="https://github.com/G-structure/tuvalu-llm"
                        target="_blank"
                        rel="noreferrer noopener"
                        class="blog-button blog-button--ghost"
                      >
                        Review the code
                      </a>
                    </div>
                  </section>

                  <AuthorCard author={post().authors[0]} />

                  <Show when={page().related.length > 0}>
                    <section class="blog-related">
                      <div class="blog-home__section-head">
                        <div>
                          <p class="blog-section-kicker">Related reading</p>
                          <h2 class="blog-section-title">More from Language Lab</h2>
                        </div>
                      </div>
                      <div class="blog-related__grid">
                        <For each={page().related}>
                          {(relatedPost) => <PostCard post={relatedPost} variant="compact" />}
                        </For>
                      </div>
                    </section>
                  </Show>

                  <Show when={page().adjacent.newer || page().adjacent.older}>
                    <nav class="blog-pagination" aria-label="Post navigation">
                      <Show when={page().adjacent.newer}>
                        {(newer) => (
                          <A href={`/blog/${newer().slug}`} class="blog-pagination__card">
                            <span class="blog-pagination__label">Newer</span>
                            <strong>{newer().title}</strong>
                          </A>
                        )}
                      </Show>
                      <Show when={page().adjacent.older}>
                        {(older) => (
                          <A href={`/blog/${older().slug}`} class="blog-pagination__card">
                            <span class="blog-pagination__label">Older</span>
                            <strong>{older().title}</strong>
                          </A>
                        )}
                      </Show>
                    </nav>
                  </Show>
                </article>

                <aside class="blog-post-sidebar">
                  <Show when={post().headings.length > 0}>
                    <TableOfContents headings={post().headings} />
                  </Show>

                  <section class="blog-panel blog-panel--sticky-share">
                    <ShareActions
                      title={post().title}
                      description={post().description}
                      url={url()}
                      eyebrow="Share Language Lab"
                      heading="Bring more people into the Tuvaluan AI story."
                      text="If this should reach builders, funders, teachers, or diaspora communities, send it on."
                    />
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

function NotFound() {
  return (
    <main class="blog-page blog-page--post">
      <HttpStatusCode code={404} />
      <OGMeta title="Post not found" description="The link may be wrong or the post may have moved." />
      <div class="blog-shell blog-shell--narrow blog-empty-state">
        <p class="blog-kicker">Language Lab Journal</p>
        <h1 class="blog-section-title">Post not found.</h1>
        <p class="blog-panel__text">
          The link may be wrong or the post may have moved.
        </p>
        <A href="/blog" class="blog-button blog-button--primary">
          Back to the blog
        </A>
      </div>
    </main>
  );
}
