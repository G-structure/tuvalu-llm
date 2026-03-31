import { createSignal, Show } from "solid-js";

function shareUrl(base: string, params: Record<string, string>) {
  const url = new URL(base);
  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.set(key, value);
  });
  return url.toString();
}

export default function ShareActions(props: {
  title: string;
  description: string;
  url: string;
  eyebrow?: string;
  heading?: string;
  text?: string;
}) {
  const [copied, setCopied] = createSignal(false);

  const xUrl = () =>
    shareUrl("https://twitter.com/intent/tweet", {
      text: props.title,
      url: props.url,
    });

  const linkedInUrl = () =>
    shareUrl("https://www.linkedin.com/sharing/share-offsite/", {
      url: props.url,
    });

  const hackerNewsUrl = () =>
    shareUrl("https://news.ycombinator.com/submitlink", {
      u: props.url,
      t: props.title,
    });

  async function handlePrimaryShare() {
    if (navigator.share) {
      try {
        await navigator.share({
          title: props.title,
          text: props.description,
          url: props.url,
        });
        return;
      } catch {
        // Fall back to copy flow below.
      }
    }

    await navigator.clipboard.writeText(props.url);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  return (
    <div class="blog-share">
      <Show when={props.heading || props.text}>
        <div class="blog-share__intro">
          <Show when={props.eyebrow}>
            <p class="blog-share__eyebrow">{props.eyebrow}</p>
          </Show>
          <Show when={props.heading}>
            <h3 class="blog-share__heading">{props.heading}</h3>
          </Show>
          <Show when={props.text}>
            <p class="blog-share__text">{props.text}</p>
          </Show>
        </div>
      </Show>
      <div class="blog-share__actions">
        <button type="button" class="blog-share__primary" onClick={handlePrimaryShare}>
          <Show when={copied()} fallback={"Copy or share link"}>
            Link copied
          </Show>
        </button>
        <a href={xUrl()} target="_blank" rel="noreferrer noopener" class="blog-share__link">
          Post on X
        </a>
        <a
          href={linkedInUrl()}
          target="_blank"
          rel="noreferrer noopener"
          class="blog-share__link"
        >
          Share on LinkedIn
        </a>
        <a
          href={hackerNewsUrl()}
          target="_blank"
          rel="noreferrer noopener"
          class="blog-share__link"
        >
          Submit to HN
        </a>
      </div>
    </div>
  );
}
