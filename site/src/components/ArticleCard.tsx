import { A } from "@solidjs/router";
import type { Article } from "~/lib/types";
import { timeAgo } from "~/lib/time";

interface ArticleCardProps {
  article: Article;
  hero?: boolean;
}

export default function ArticleCard(props: ArticleCardProps) {
  const titleTvl = () => props.article.title_tvl;
  const titleEn = () => props.article.title_en;
  const title = () => titleTvl() || titleEn();
  const hasTranslation = () => !!titleTvl();
  const ago = () => timeAgo(props.article.published_at);
  const sourceName = () => {
    const map: Record<string, string> = {
      goal: "Goal.com",
      fifa: "FIFA.com",
      sky: "Sky Sports",
    };
    return map[props.article.source_id] || props.article.source_id;
  };

  if (props.hero) {
    return (
      <A
        href={`/articles/${props.article.id}`}
        class="block rounded-xl overflow-hidden bg-white border border-[var(--sky-dark)] no-underline text-inherit hover:border-[var(--ocean-bright)] transition-colors"
      >
        {props.article.image_url && (
          <img
            src={props.article.image_url}
            alt={props.article.image_alt || title()}
            loading="eager"
            fetchpriority="high"
            decoding="async"
            class="w-full h-48 sm:h-64 object-cover"
          />
        )}
        <div class="p-4">
          <h2 class="text-lg sm:text-xl font-bold text-gray-900 leading-snug">
            {title()}
          </h2>
          {hasTranslation() && (
            <p class="mt-1 text-sm text-gray-400 italic leading-snug line-clamp-2">
              {titleEn()}
            </p>
          )}
          <div class="mt-2 flex items-center gap-2 text-sm text-gray-500">
            <span>{sourceName()}</span>
            <span>&middot;</span>
            <span>{ago()}</span>
          </div>
        </div>
      </A>
    );
  }

  return (
    <A
      href={`/articles/${props.article.id}`}
      class="flex gap-3 p-3 rounded-lg bg-white no-underline text-inherit hover:bg-[var(--sky)] transition-colors"
    >
      {props.article.image_url && (
        <img
          src={props.article.image_url}
          alt={props.article.image_alt || title()}
          loading="lazy"
          decoding="async"
          class="w-24 h-24 sm:w-28 sm:h-28 object-cover rounded-lg shrink-0"
        />
      )}
      <div class="flex flex-col justify-center min-w-0">
        <h3 class="text-base font-semibold text-gray-900 leading-snug line-clamp-3">
          {title()}
        </h3>
        {hasTranslation() && (
          <p class="mt-0.5 text-xs text-gray-400 italic line-clamp-1">
            {titleEn()}
          </p>
        )}
        <div class="mt-1 flex items-center gap-2 text-xs text-gray-500">
          <span>{sourceName()}</span>
          <span>&middot;</span>
          <span>{ago()}</span>
        </div>
      </div>
    </A>
  );
}
