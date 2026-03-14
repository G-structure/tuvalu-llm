import { Marked } from "marked";
import hljs from "highlight.js";

const marked = new Marked({
  renderer: {
    code({ text, lang }: { text: string; lang?: string }) {
      const language = lang && hljs.getLanguage(lang) ? lang : "plaintext";
      const highlighted = hljs.highlight(text, { language }).value;
      const id = `code-${Math.random().toString(36).slice(2, 9)}`;
      return `<div class="code-block group relative my-3">
        <div class="code-header flex items-center justify-between px-4 py-2 text-xs text-gray-400 bg-[#1e1e2e] rounded-t-lg border-b border-[#2a2a3e]">
          <span>${language}</span>
          <button onclick="navigator.clipboard.writeText(document.getElementById('${id}').textContent).then(()=>{this.textContent='Copied!';setTimeout(()=>{this.textContent='Copy'},2000)})" class="hover:text-white transition-colors px-2 py-0.5 rounded hover:bg-white/10">Copy</button>
        </div>
        <pre class="bg-[#1e1e2e] rounded-b-lg p-4 overflow-x-auto"><code id="${id}" class="hljs language-${language} text-sm">${highlighted}</code></pre>
      </div>`;
    },
    codespan({ text }: { text: string }) {
      return `<code class="bg-[#1e1e2e] px-1.5 py-0.5 rounded text-[#e2b86b] text-sm">${text}</code>`;
    },
  },
});

export function renderMarkdown(text: string): string {
  return marked.parse(text) as string;
}
