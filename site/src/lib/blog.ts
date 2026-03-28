import { Marked } from "marked";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import javascript from "highlight.js/lib/languages/javascript";
import typescript from "highlight.js/lib/languages/typescript";
import bash from "highlight.js/lib/languages/bash";
import json from "highlight.js/lib/languages/json";

hljs.registerLanguage("python", python);
hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("json", json);

export interface BlogPost {
  slug: string;
  title: string;
  description: string;
  date: string;
  image?: string;
  authors?: string[];
  tags?: string[];
}

export interface BlogPostFull extends BlogPost {
  html: string;
}

const marked = new Marked({
  renderer: {
    code({ text, lang }: { text: string; lang?: string }) {
      const language = lang && hljs.getLanguage(lang) ? lang : "plaintext";
      const highlighted =
        language !== "plaintext"
          ? hljs.highlight(text, { language }).value
          : text
              .replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;");
      const id = `code-${Math.random().toString(36).slice(2, 9)}`;
      return `<div class="blog-code group relative my-4">
        <div class="blog-code__header">
          <span>${language}</span>
          <button onclick="navigator.clipboard.writeText(document.getElementById('${id}').textContent).then(()=>{this.textContent='Copied!';setTimeout(()=>{this.textContent='Copy'},2000)})" class="blog-code__copy">Copy</button>
        </div>
        <pre class="blog-code__pre"><code id="${id}" class="hljs language-${language} text-sm">${highlighted}</code></pre>
      </div>`;
    },
    codespan({ text }: { text: string }) {
      return `<code class="blog-inline-code">${text}</code>`;
    },
    image({ href, text }: { href: string; text?: string }) {
      return `<figure class="blog-figure">
        <img src="${href}" alt="${text || ""}" loading="lazy" decoding="async" />
        ${text ? `<figcaption>${text}</figcaption>` : ""}
      </figure>`;
    },
  },
});

/** Parse YAML-like frontmatter (simple key: value, no nested objects needed) */
function parseFrontmatter(raw: string): { meta: Record<string, any>; body: string } {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  if (!match) return { meta: {}, body: raw };

  const meta: Record<string, any> = {};
  for (const line of match[1].split("\n")) {
    const idx = line.indexOf(":");
    if (idx === -1) continue;
    const key = line.slice(0, idx).trim();
    let val: any = line.slice(idx + 1).trim();
    // Handle arrays like ["a", "b"]
    if (val.startsWith("[") && val.endsWith("]")) {
      try {
        val = JSON.parse(val.replace(/'/g, '"'));
      } catch {
        /* keep as string */
      }
    }
    // Strip surrounding quotes
    if (typeof val === "string" && val.startsWith('"') && val.endsWith('"')) {
      val = val.slice(1, -1);
    }
    meta[key] = val;
  }
  return { meta, body: match[2] };
}

export function parseBlogPost(slug: string, raw: string): BlogPostFull {
  const { meta, body } = parseFrontmatter(raw);
  return {
    slug,
    title: meta.title || slug,
    description: meta.description || "",
    date: meta.date || "",
    image: meta.image,
    authors: meta.authors,
    tags: meta.tags,
    html: marked.parse(body) as string,
  };
}

export function parseBlogMeta(slug: string, raw: string): BlogPost {
  const { meta } = parseFrontmatter(raw);
  return {
    slug,
    title: meta.title || slug,
    description: meta.description || "",
    date: meta.date || "",
    image: meta.image,
    authors: meta.authors,
    tags: meta.tags,
  };
}
