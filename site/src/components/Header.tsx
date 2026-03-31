import { A } from "@solidjs/router";
import { createSignal, Show } from "solid-js";
import LanguageToggle from "./LanguageToggle";
import type { LanguageMode } from "./LanguageToggle";

interface HeaderProps {
  langMode?: LanguageMode;
  onLangChange?: (mode: LanguageMode) => void;
}

export default function Header(props: HeaderProps) {
  const [menuOpen, setMenuOpen] = createSignal(false);

  return (
    <header class="bg-[var(--ocean-deep)] text-white border-b-2 border-[var(--gold)]">
      <div class="max-w-3xl mx-auto px-4 py-4 flex items-center justify-between">
        <A href="/" class="text-xl font-bold tracking-wide no-underline text-white">
          TALAFUTIPOLO
        </A>
        <div class="flex items-center gap-3">
          {/* Desktop nav links */}
          <nav class="hidden sm:flex items-center gap-3" aria-label="Main navigation">
            <A
              href="/blog"
              class="text-[var(--sky-dark)] hover:text-[var(--gold)] transition-colors no-underline text-sm"
            >
              Journal
            </A>
            <A
              href="/fatele"
              class="text-[var(--sky-dark)] hover:text-[var(--gold)] transition-colors no-underline text-sm"
            >
              Kominiti
            </A>
            <A
              href="/chat"
              class="text-[var(--sky-dark)] hover:text-[var(--gold)] transition-colors no-underline text-sm"
            >
              Chat
            </A>
            <A
              href="/chat/training"
              class="text-[var(--sky-dark)] hover:text-[var(--gold)] transition-colors no-underline text-sm"
            >
              Training
            </A>
            <A
              href="/chat/eval"
              class="text-[var(--sky-dark)] hover:text-[var(--gold)] transition-colors no-underline text-sm"
            >
              Eval
            </A>
          </nav>
          <A
            href="/search"
            class="text-[var(--sky-dark)] hover:text-[var(--gold)] transition-colors no-underline text-sm p-1 min-w-[36px] min-h-[36px] flex items-center justify-center"
            aria-label="Search"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.3-4.3" />
            </svg>
          </A>
          {props.langMode && props.onLangChange && (
            <LanguageToggle mode={props.langMode} onChange={props.onLangChange} />
          )}
          {/* Mobile hamburger */}
          <button
            type="button"
            class="sm:hidden text-[var(--sky-dark)] hover:text-[var(--gold)] transition-colors p-1 min-w-[36px] min-h-[36px] flex items-center justify-center"
            aria-label={menuOpen() ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen()}
            aria-controls="mobile-nav"
            onClick={() => setMenuOpen(!menuOpen())}
          >
            <Show
              when={menuOpen()}
              fallback={
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
                  <path d="M3 12h18M3 6h18M3 18h18" />
                </svg>
              }
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            </Show>
          </button>
        </div>
      </div>
      {/* Mobile nav */}
      <Show when={menuOpen()}>
        <nav
          id="mobile-nav"
          class="sm:hidden border-t border-white/10 px-4 py-3 flex flex-col gap-2"
          aria-label="Main navigation"
        >
          <A href="/blog" class="text-[var(--sky-dark)] hover:text-[var(--gold)] no-underline text-sm py-2" onClick={() => setMenuOpen(false)}>Journal</A>
          <A href="/fatele" class="text-[var(--sky-dark)] hover:text-[var(--gold)] no-underline text-sm py-2" onClick={() => setMenuOpen(false)}>Kominiti</A>
          <A href="/chat" class="text-[var(--sky-dark)] hover:text-[var(--gold)] no-underline text-sm py-2" onClick={() => setMenuOpen(false)}>Chat</A>
          <A href="/chat/training" class="text-[var(--sky-dark)] hover:text-[var(--gold)] no-underline text-sm py-2" onClick={() => setMenuOpen(false)}>Training</A>
          <A href="/chat/eval" class="text-[var(--sky-dark)] hover:text-[var(--gold)] no-underline text-sm py-2" onClick={() => setMenuOpen(false)}>Eval</A>
        </nav>
      </Show>
    </header>
  );
}
