import { A } from "@solidjs/router";

const LEGAL_YEAR = 2026;

export default function SiteFooter() {
  return (
    <footer class="site-footer" aria-label="Site legal footer">
      <div class="site-footer__inner">
        <div class="site-footer__brand">
          <span class="site-footer__mark" aria-hidden="true" />
          <div>
            <strong>Fenua Intelligence</strong>
            <span>AI tools for Tuvaluan language, news, and community feedback.</span>
          </div>
        </div>
        <nav class="site-footer__links" aria-label="Legal links">
          <A href="/legal#privacy">Privacy Policy</A>
          <A href="/legal#terms">Terms of Service</A>
          <a href="mailto:admin@choruslanguagelabs.org">Contact</a>
        </nav>
        <p class="site-footer__notice">
          &copy; {LEGAL_YEAR} Chorus Language Labs LLC. Fenua Intelligence,
          TVL Chat, and Talafutipolo are trademarks or service marks of
          Chorus Language Labs LLC.
        </p>
      </div>
    </footer>
  );
}
