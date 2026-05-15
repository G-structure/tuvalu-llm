import { Link, Meta, Title } from "@solidjs/meta";
import { absoluteImageUrl, SITE_META } from "~/lib/site";

interface AlternateLink {
  href: string;
  title: string;
  type: string;
}

interface OGMetaProps {
  title: string;
  description?: string;
  image?: string | null;
  imageOrigin?: string;
  imageWidth?: number | null;
  imageHeight?: number | null;
  imageAlt?: string | null;
  url?: string;
  type?: string;
  publishedAt?: string;
  modifiedAt?: string;
  category?: string;
  siteName?: string;
  titleSuffix?: string;
  keywords?: string[];
  authorNames?: string[];
  locale?: string;
  alternateLinks?: AlternateLink[];
}

export default function OGMeta(props: OGMetaProps) {
  const canonicalUrl = () => props.url;
  const hasImage = () => props.image !== null;
  const imageUrl = () => hasImage() ? absoluteImageUrl(props.image || SITE_META.defaultOgImage, props.imageOrigin) : "";
  const imageWidth = () => props.imageWidth || SITE_META.defaultOgImageWidth;
  const imageHeight = () => props.imageHeight || SITE_META.defaultOgImageHeight;
  const imageAlt = () => props.imageAlt || SITE_META.defaultOgImageAlt;
  const description = () => props.description || props.title;
  const siteName = () => props.siteName || SITE_META.productName;
  const titleSuffix = () => props.titleSuffix || SITE_META.productName;

  return (
    <>
      <Title>{`${props.title} — ${titleSuffix()}`}</Title>
      <Meta name="description" content={description()} />
      <Meta
        name="robots"
        content="index,follow,max-image-preview:large,max-snippet:-1,max-video-preview:-1"
      />

      {props.keywords?.length ? (
        <Meta name="keywords" content={props.keywords.join(", ")} />
      ) : null}
      {props.authorNames?.length ? (
        <Meta name="author" content={props.authorNames.join(", ")} />
      ) : null}

      {canonicalUrl() ? <Link rel="canonical" href={canonicalUrl()!} /> : null}

      {props.alternateLinks?.map((link) => (
        <Link rel="alternate" type={link.type} title={link.title} href={link.href} />
      ))}

      <Meta property="og:title" content={props.title} />
      <Meta property="og:description" content={description()} />
      <Meta property="og:type" content={props.type || "website"} />
      <Meta property="og:site_name" content={siteName()} />
      <Meta property="og:locale" content={props.locale || "en_US"} />
      <Meta property="og:locale:alternate" content="tvl" />
      {canonicalUrl() ? <Meta property="og:url" content={canonicalUrl()!} /> : null}
      {hasImage() ? <Meta property="og:image" content={imageUrl()} /> : null}
      {hasImage() ? <Meta property="og:image:width" content={String(imageWidth())} /> : null}
      {hasImage() ? <Meta property="og:image:height" content={String(imageHeight())} /> : null}
      {hasImage() ? <Meta property="og:image:alt" content={imageAlt()} /> : null}

      {props.publishedAt ? (
        <Meta property="article:published_time" content={props.publishedAt} />
      ) : null}
      {props.modifiedAt ? (
        <Meta property="article:modified_time" content={props.modifiedAt} />
      ) : null}
      {props.category ? (
        <Meta property="article:section" content={props.category} />
      ) : null}
      {props.keywords?.map((keyword) => (
        <Meta property="article:tag" content={keyword} />
      ))}

      <Meta name="twitter:card" content={hasImage() ? "summary_large_image" : "summary"} />
      <Meta name="twitter:title" content={props.title} />
      <Meta name="twitter:description" content={description()} />
      {hasImage() ? <Meta name="twitter:image" content={imageUrl()} /> : null}
      {hasImage() ? <Meta name="twitter:image:alt" content={imageAlt()} /> : null}
    </>
  );
}
