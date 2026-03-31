import { absoluteUrl } from "./site";

export interface BlogAuthorLink {
  label: string;
  href: string;
}

export interface BlogAuthor {
  slug: string;
  name: string;
  role: string;
  bio: string;
  initials: string;
  location?: string;
  href: string;
  links?: BlogAuthorLink[];
}

export const BLOG_AUTHORS: Record<string, BlogAuthor> = {
  "language-lab": {
    slug: "language-lab",
    name: "Language Lab",
    role: "Nonprofit research lab",
    bio:
      "Language Lab builds practical AI tools for endangered and low-resource languages, starting with Tuvaluan. The journal covers product launches, research updates, field deployment, and the engineering work required to make real language technology available where it usually does not.",
    initials: "LL",
    location: "Tuvalu / Remote",
    href: absoluteUrl("/demo"),
    links: [
      { label: "Project demo", href: absoluteUrl("/demo") },
      { label: "Live model", href: absoluteUrl("/chat") },
      {
        label: "GitHub",
        href: "https://github.com/G-structure/tuvalu-llm",
      },
      {
        label: "Hugging Face",
        href: "https://huggingface.co/datasets/FriezaForce/tv2en-cleaned",
      },
    ],
  },
};

function normalizeAuthorKey(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, "-");
}

export function resolveBlogAuthor(value: string): BlogAuthor {
  const key = normalizeAuthorKey(value);
  return (
    BLOG_AUTHORS[key] || {
      slug: key,
      name: value.trim(),
      role: "Contributor",
      bio: `${value.trim()} contributes to the Language Lab publication.`,
      initials: value
        .trim()
        .split(/\s+/)
        .slice(0, 2)
        .map((part) => part[0]?.toUpperCase() || "")
        .join(""),
      href: absoluteUrl(`/blog/author/${key}`),
    }
  );
}

export function getAllAuthors(): BlogAuthor[] {
  return Object.values(BLOG_AUTHORS);
}
