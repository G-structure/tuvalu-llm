import { A } from "@solidjs/router";
import LanguageToggle from "./LanguageToggle";
import type { LanguageMode } from "./LanguageToggle";

interface HeaderProps {
  langMode?: LanguageMode;
  onLangChange?: (mode: LanguageMode) => void;
}

export default function Header(props: HeaderProps) {
  return (
    <header class="bg-[var(--ocean-deep)] text-white border-b-2 border-[var(--gold)]">
      <div class="max-w-3xl mx-auto px-4 py-4 flex items-center justify-between">
        <A href="/" class="text-xl font-bold tracking-wide no-underline text-white">
          TALAFUTIPOLO
        </A>
        <div class="flex items-center gap-3">
          <A
            href="/chat"
            class="text-[var(--sky-dark)] hover:text-[var(--gold)] transition-colors no-underline text-sm"
            title="TVL Chat"
          >
            Chat
          </A>
          <A
            href="/search"
            class="text-[var(--sky-dark)] hover:text-[var(--gold)] transition-colors no-underline text-sm"
            title="Saili (Search)"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.3-4.3" />
            </svg>
          </A>
          <span class="text-xs text-[var(--sky-dark)] hidden sm:inline">
            Tuvaluan Football News
          </span>
          {props.langMode && props.onLangChange && (
            <LanguageToggle mode={props.langMode} onChange={props.onLangChange} />
          )}
        </div>
      </div>
    </header>
  );
}
