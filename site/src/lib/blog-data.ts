import { parseBlogMeta, parseBlogPost, type BlogPost, type BlogPostFull } from "./blog";

// Vite imports all .md files as raw strings at build time — works on CF Workers
const postFiles = import.meta.glob("../content/blog/*.md", {
  query: "?raw",
  eager: true,
  import: "default",
}) as Record<string, string>;

function slugFromPath(path: string): string {
  return path.replace(/^.*\//, "").replace(/\.md$/, "");
}

let _postsCache: BlogPost[] | null = null;

export function getAllPosts(): BlogPost[] {
  if (_postsCache) return _postsCache;
  const posts: BlogPost[] = [];
  for (const [path, raw] of Object.entries(postFiles)) {
    posts.push(parseBlogMeta(slugFromPath(path), raw));
  }
  posts.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  _postsCache = posts;
  return posts;
}

export function getPostBySlug(slug: string): BlogPostFull | null {
  for (const [path, raw] of Object.entries(postFiles)) {
    if (slugFromPath(path) === slug) {
      return parseBlogPost(slug, raw);
    }
  }
  return null;
}
