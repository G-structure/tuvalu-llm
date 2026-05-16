import { A, useLocation } from "@solidjs/router";
import { createSignal, Show } from "solid-js";
import LanguageToggle from "./LanguageToggle";
import type { LanguageMode } from "./LanguageToggle";

interface HeaderProps {
  langMode?: LanguageMode;
  onLangChange?: (mode: LanguageMode) => void;
}

export default function Header(props: HeaderProps) {
  const [menuOpen, setMenuOpen] = createSignal(false);
  const location = useLocation();

  const navItems = [
    { href: "/", label: "Talafutipolo", homeSection: "news" },
    { href: "/chat", label: "TVL Chat", cta: true, badge: "AI" },
    { href: "/#latest", label: "Latest", homeSection: "latest" },
    { href: "/fatele", label: "Kominiti" },
    { href: "/chat/training", label: "Training" },
    { href: "/demo", label: "About" },
  ];

  const isActive = (item: (typeof navItems)[number]) => {
    if (item.href === "/#latest") return location.pathname === "/";
    if (item.href === "/") {
      return (
        location.pathname.startsWith("/articles") ||
        location.pathname.startsWith("/category")
      );
    }
    if (item.href === "/chat") return location.pathname === "/chat";
    return location.pathname === item.href || location.pathname.startsWith(`${item.href}/`);
  };

  return (
    <header class="site-header">
      <div class="site-header__bar">
        <A href="/" class="site-brand" aria-label="Talafutipolo home">
          <span class="site-brand__mark" aria-hidden="true" />
          <span class="site-brand__text">
            <span class="site-brand__name">Fenua</span>
            <span class="site-brand__sub">Intelligence</span>
          </span>
        </A>
        <div class="site-header__actions">
          {/* Desktop nav links */}
          <nav class="site-nav site-nav--desktop" aria-label="Main navigation">
            {navItems.map((item) => (
              <A
                href={item.href}
                class={`site-nav__link${isActive(item) ? " site-nav__link--active" : ""}${item.cta ? " site-nav__link--cta" : ""}`}
              >
                {item.label}
                {item.badge && <span class="site-nav__badge">{item.badge}</span>}
              </A>
            ))}
          </nav>
          <A
            href="/search"
            class="site-search-link no-underline"
            aria-label="Search"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.3-4.3" />
            </svg>
            <span>Search</span>
          </A>
          {props.langMode && props.onLangChange && (
            <LanguageToggle mode={props.langMode} onChange={props.onLangChange} />
          )}
          {!props.langMode && (
            <div class="site-language-segment" aria-label="Language display">
              <span class="site-language-segment__item site-language-segment__item--active">TV</span>
              <span class="site-language-segment__item">EN</span>
            </div>
          )}
          <div class="site-locale-pill" aria-label="Tuvalu locale">
            <span class="site-locale-pill__flag" aria-hidden="true" />
            <span>TVL</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="m6 9 6 6 6-6" />
            </svg>
          </div>
          {/* Mobile hamburger */}
          <button
            type="button"
            class="site-menu-button site-menu-button--mobile"
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
          class="site-mobile-nav"
          aria-label="Main navigation"
        >
          {navItems.map((item) => (
            <A
              href={item.href}
              class={`site-mobile-nav__link no-underline${isActive(item) ? " site-mobile-nav__link--active" : ""}`}
              onClick={() => setMenuOpen(false)}
            >
              {item.label}
            </A>
          ))}
        </nav>
      </Show>
    </header>
  );
}
