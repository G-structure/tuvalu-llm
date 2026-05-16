import { A } from "@solidjs/router";
import type { Article } from "~/lib/types";
import { timeAgo } from "~/lib/time";
import { FOOTBALL_META } from "~/lib/site";

interface ArticleCardProps {
  article: Article;
  hero?: boolean;
  tile?: boolean;
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
  const categoryLabel = () => (props.article.category || "Talafuti").replace(/-/g, " ");
  const hasImage = () => !!props.article.image_url;
  const imageSrc = () => props.article.image_url || FOOTBALL_META.articleFallbackOgImage;
  const imageAlt = () =>
    props.article.image_alt ||
    (hasImage() ? title() : FOOTBALL_META.articleFallbackImageAlt);
  const imageWidth = () =>
    props.article.image_width ||
    (hasImage()
      ? FOOTBALL_META.defaultOgImageWidth
      : FOOTBALL_META.articleFallbackImageWidth);
  const imageHeight = () =>
    props.article.image_height ||
    (hasImage()
      ? FOOTBALL_META.defaultOgImageHeight
      : FOOTBALL_META.articleFallbackImageHeight);

  if (props.hero) {
    return (
      <A
        href={`/articles/${props.article.id}`}
        class="article-card article-card--hero"
      >
        <div class="article-card__media article-card__media--hero">
          <img
            src={imageSrc()}
            alt={imageAlt()}
            width={imageWidth()}
            height={imageHeight()}
            loading="eager"
            fetchpriority="high"
            decoding="async"
            sizes="(max-width: 900px) 100vw, 56vw"
            class="article-card__image"
          />
        </div>
        <div class="article-card__body">
          <div class="article-card__meta-row">
            <span class="article-card__source">{sourceName()}</span>
            <span>{ago()}</span>
          </div>
          <h2 class="article-card__title article-card__title--hero">
            {title()}
          </h2>
          {hasTranslation() && (
            <p class="article-card__subtitle line-clamp-2">
              {titleEn()}
            </p>
          )}
        </div>
      </A>
    );
  }

  if (props.tile) {
    return (
      <A
        href={`/articles/${props.article.id}`}
        class="article-card article-card--tile"
      >
        <div class="article-card__media article-card__media--tile">
          <img
            src={imageSrc()}
            alt={imageAlt()}
            width={imageWidth()}
            height={imageHeight()}
            loading="lazy"
            decoding="async"
            sizes="(max-width: 760px) 100vw, 25vw"
            class="article-card__image"
          />
          <span class="article-card__badge">{categoryLabel()}</span>
        </div>
        <div class="article-card__body article-card__body--tile">
          <h3 class="article-card__title line-clamp-2">
            {title()}
          </h3>
          <div class="article-card__tile-foot">
            <span>{ago()}</span>
            <span class="article-card__bookmark" aria-hidden="true" />
          </div>
        </div>
      </A>
    );
  }

  return (
    <A
      href={`/articles/${props.article.id}`}
      class="article-card article-card--row"
    >
      <div class="article-card__media article-card__media--thumb">
        <img
          src={imageSrc()}
          alt={imageAlt()}
          width={imageWidth()}
          height={imageHeight()}
          loading="lazy"
          decoding="async"
          sizes="(max-width: 520px) 100vw, 8.5rem"
          class="article-card__image"
        />
      </div>
      <div class="article-card__body article-card__body--row">
        <div class="article-card__meta-row">
          <span class="article-card__source">{sourceName()}</span>
          <span>{ago()}</span>
        </div>
        <h3 class="article-card__title line-clamp-3">
          {title()}
        </h3>
        {hasTranslation() && (
          <p class="article-card__subtitle article-card__subtitle--row line-clamp-1">
            {titleEn()}
          </p>
        )}
      </div>
    </A>
  );
}
