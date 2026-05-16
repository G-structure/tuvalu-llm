import OGMeta from "~/components/OGMeta";
import StructuredData from "~/components/StructuredData";
import { languageLabOrganization, languageLabWebsite } from "~/lib/seo";
import { absoluteUrl, SITE_META } from "~/lib/site";

const effectiveDate = "May 16, 2026";
const companyName = "Chorus Language Labs LLC";
const contactEmail = "admin@choruslanguagelabs.org";

export default function LegalPage() {
  const legalUrl = absoluteUrl("/legal");

  return (
    <main class="legal-page">
      <OGMeta
        title="Privacy Policy and Terms of Service"
        description="Privacy Policy and Terms of Service for Fenua Intelligence, TVL Chat, Talafutipolo, and Language Lab."
        url={legalUrl}
        image={SITE_META.defaultOgImage}
        siteName="Fenua Intelligence"
        titleSuffix="Fenua Intelligence"
      />
      <StructuredData
        data={[
          languageLabOrganization(),
          languageLabWebsite(),
          {
            "@context": "https://schema.org",
            "@type": "WebPage",
            name: "Privacy Policy and Terms of Service",
            url: legalUrl,
            dateModified: "2026-05-16",
            publisher: {
              "@type": "Organization",
              name: companyName,
              email: contactEmail,
            },
          },
        ]}
      />

      <section class="legal-hero">
        <div class="site-shell legal-hero__inner">
          <p class="legal-eyebrow">Legal</p>
          <h1>Privacy Policy and Terms of Service</h1>
          <p>
            Official notices for Fenua Intelligence, TVL Chat, Talafutipolo,
            the Language Lab pages, and related services operated by{" "}
            {companyName}.
          </p>
          <div class="legal-quicklinks" aria-label="Legal sections">
            <a href="#privacy">Privacy Policy</a>
            <a href="#terms">Terms of Service</a>
            <a href={`mailto:${contactEmail}`}>Contact</a>
          </div>
        </div>
      </section>

      <section class="site-shell legal-audit" aria-labelledby="data-audit">
        <div class="legal-section-heading">
          <p class="legal-eyebrow">Data audit</p>
          <h2 id="data-audit">What this site currently collects</h2>
          <p>
            This summary reflects the current code paths used by the site. It
            is included so visitors can understand the practical data flows
            before reading the full policy.
          </p>
        </div>

        <div class="legal-audit__grid">
          <article>
            <span>Device</span>
            <h3>Local storage and cache</h3>
            <p>
              The browser may store a pseudonymous session ID, island
              preference, chat conversations, the active chat ID, sync
              preference, and cached pages or assets for faster loading and
              offline fallback.
            </p>
          </article>
          <article>
            <span>Community</span>
            <h3>Article signals and feedback</h3>
            <p>
              Article votes, shares, reveal events, helpfulness ratings, mode
              preferences, correction text, optional island, session ID, and
              timestamps may be saved to measure usefulness and improve content.
            </p>
          </article>
          <article>
            <span>Chat</span>
            <h3>Conversations and corrections</h3>
            <p>
              Chat messages are sent to the model service to answer you. If sync
              is on, conversation JSON, titles, feedback, corrections, selected
              text, language mode, island, metadata, and timestamps may be saved
              for product improvement and training examples.
            </p>
          </article>
          <article>
            <span>Contact</span>
            <h3>Email and service metadata</h3>
            <p>
              Newsletter signup stores an email address. Hosting and security
              systems may process technical request data such as IP address,
              user agent, request path, timing, and error logs.
            </p>
          </article>
        </div>
      </section>

      <section id="privacy" class="site-shell legal-document" aria-labelledby="privacy-title">
        <div class="legal-document__header">
          <p class="legal-eyebrow">Effective {effectiveDate}</p>
          <h2 id="privacy-title">Privacy Policy</h2>
        </div>

        <div class="legal-copy">
          <h3>1. Scope</h3>
          <p>
            This Privacy Policy explains how {companyName} collects, uses,
            stores, and shares information when you use Fenua Intelligence, TVL
            Chat, Talafutipolo, Language Lab pages, and related features
            available through this site.
          </p>

          <h3>2. Information we collect</h3>
          <p>
            We collect information you provide directly, including newsletter
            email addresses, chat prompts, chat feedback, correction text,
            article feedback, article correction text, and any other content
            you choose to submit.
          </p>
          <p>
            We collect product-use information, including pseudonymous browser
            session IDs, optional island selections, language or mode
            preferences, conversation IDs, message IDs, timestamps, rating
            events, share or reveal events, sync preference, and metadata needed
            to operate and improve the service. We do not ask for precise
            location, but island selections and language feedback may still be
            meaningful in small communities.
          </p>
          <p>
            We use browser storage and a service worker cache so the site can
            load quickly, remember saved conversations on your device, and offer
            limited offline fallback for previously visited pages.
          </p>
          <p>
            We and our hosting providers may process technical information such
            as IP address, user agent, request URL, device/browser information,
            logs, performance data, and security events.
          </p>

          <h3>3. How we use information</h3>
          <p>
            We use information to provide the site and chat service, respond to
            prompts, save conversations when sync is enabled, maintain local
            saved chats, collect community feedback, improve Tuvaluan-English
            content quality, build evaluation and training examples, debug
            errors, protect against abuse, send requested updates, and comply
            with legal obligations.
          </p>

          <h3>4. AI processing and training choices</h3>
          <p>
            Chat messages must be transmitted to the model service in order to
            generate a response. The chat sync setting controls whether this
            site stores conversation transcripts in its database for later
            access, analysis, evaluation, and training examples. If sync is set
            to local only, conversations are intended to remain in browser
            storage on your device. Local-only mode does not prevent transient
            processing, security logs, error logs, or provider processing that
            is necessary to produce the live reply.
          </p>
          <p>
            Do not submit passwords, private keys, financial account data,
            health information, government identifiers, confidential business
            data, or other sensitive personal information unless we explicitly
            ask for it and explain why.
          </p>

          <h3>5. When we share information</h3>
          <p>
            We may share information with service providers that help us host,
            store, secure, cache, process, or deliver the service, including
            cloud infrastructure, font/CDN providers, email tooling,
            analytics/diagnostics if added in the future, and model-processing
            providers. We may also share information if required by law, to
            protect rights and safety, to investigate abuse, or as part of a
            merger, acquisition, financing, reorganization, or other business
            transfer.
          </p>
          <p>
            Based on the current implementation, we do not sell personal
            information or share it for cross-context behavioral advertising,
            and the current site does not include third-party advertising
            trackers.
          </p>

          <h3>6. Retention</h3>
          <p>
            Local conversations and preferences remain on your device until you
            delete them, clear browser storage, or change devices/browsers.
            Synced chat conversations are kept until deleted through the
            product or until we remove them under our retention and maintenance
            practices. Newsletter emails are kept until you ask to unsubscribe
            or delete them. Feedback, signals, corrections, and derived training
            examples may be retained to improve the service, unless deletion is
            required or requested and we can reasonably identify the record.
            Deletion may not remove aggregated, de-identified, backup, cache, or
            model-derived information that no longer identifies a specific user
            or that must be kept for security, legal, or operational reasons.
          </p>

          <h3>7. Your choices</h3>
          <p>
            You can turn chat sync off, delete saved conversations in the chat
            interface, clear local browser storage, avoid submitting optional
            island information, and contact us to request access, correction, or
            deletion where applicable. Some requests may require enough
            information to verify the request and locate the relevant email,
            session, conversation, or feedback record.
          </p>

          <h3>8. Security</h3>
          <p>
            We use reasonable administrative, technical, and organizational
            safeguards appropriate to the nature of the service. No internet
            service is perfectly secure, so please use care when deciding what
            to submit.
          </p>

          <h3>9. Children</h3>
          <p>
            The service is not directed to children under 13, and we do not
            knowingly collect personal information from children under 13. If
            you are under 13, do not submit personal information or use chat
            features unless a parent or guardian has provided any consent
            required by law. If you believe a child provided personal
            information, contact us so we can review and remove it where
            appropriate.
          </p>

          <h3>10. International use</h3>
          <p>
            Information may be processed in the United States and other places
            where our service providers operate. By using the service, you
            understand that information may be transferred to and processed in
            jurisdictions that may have different privacy laws than your own.
          </p>

          <h3>11. Changes and contact</h3>
          <p>
            We may update this Privacy Policy as the service changes. The latest
            version will be posted on this page with its effective date. Contact
            us at <a href={`mailto:${contactEmail}`}>{contactEmail}</a> with
            privacy questions or requests.
          </p>
        </div>
      </section>

      <section id="terms" class="site-shell legal-document" aria-labelledby="terms-title">
        <div class="legal-document__header">
          <p class="legal-eyebrow">Effective {effectiveDate}</p>
          <h2 id="terms-title">Terms of Service</h2>
        </div>

        <div class="legal-copy">
          <h3>1. Agreement to these Terms</h3>
          <p>
            These Terms of Service are an agreement between you and{" "}
            {companyName}. By accessing or using the service, you agree to these
            Terms and the Privacy Policy above. If you do not agree, do not use
            the service.
          </p>

          <h3>2. Eligibility and accounts</h3>
          <p>
            You must be able to form a binding agreement with us, or use the
            service with permission from a parent, guardian, school, or
            organization that is authorized to accept these Terms for you. The
            current service is mostly session-based and may not require an
            account, but saved local or synced chat features still depend on
            browser/session identifiers.
          </p>

          <h3>3. The service</h3>
          <p>
            The service includes Tuvaluan-English news, community feedback
            tools, AI chat, evaluation pages, training dashboards, demos,
            newsletters, and related language technology features. We may add,
            change, suspend, or discontinue features at any time.
          </p>

          <h3>4. AI output and editorial limits</h3>
          <p>
            AI outputs, translations, summaries, and generated answers may be
            inaccurate, incomplete, outdated, or offensive. They are provided
            for general information, language learning, research, and product
            exploration only. Do not rely on the service for medical, legal,
            financial, emergency, safety-critical, or other professional advice.
          </p>

          <h3>5. Your content and license to us</h3>
          <p>
            You retain any rights you have in prompts, messages, corrections,
            feedback, and other content you submit. You grant {companyName} a
            non-exclusive, worldwide, royalty-free license to host, store,
            reproduce, process, display, adapt, create derivative works from,
            analyze, and use that content as needed to operate, secure,
            evaluate, improve, preserve Tuvaluan language resources, and train
            the service, subject to the Privacy Policy and any product sync
            choices shown in the interface.
          </p>
          <p>
            You represent that you have the rights and permissions needed to
            submit the content you provide and that your content does not
            violate law, infringe another person's rights, or breach any duty of
            confidentiality.
          </p>

          <h3>6. Acceptable use</h3>
          <p>
            You may not use the service to break the law, violate another
            person's rights, submit malware, attempt unauthorized access,
            interfere with service operation, scrape or overload the service,
            impersonate others, submit abusive or hateful content, exploit
            children, generate instructions for wrongdoing, evade safety
            controls, or use outputs to make high-impact decisions about people
            without appropriate human review.
          </p>

          <h3>7. Third-party content and services</h3>
          <p>
            The service may display, link to, translate, summarize, or rely on
            third-party content, sources, model providers, hosting providers, or
            APIs. We are not responsible for third-party services or content,
            and their own terms and policies may apply.
          </p>

          <h3>8. Intellectual property</h3>
          <p>
            The service, software, design, trademarks, logos, and other
            materials we provide are owned by or licensed to {companyName} or
            its licensors, except for third-party content and user content.
            Fenua Intelligence, TVL Chat, Talafutipolo, and related names,
            logos, and product identifiers are trademarks or service marks of{" "}
            {companyName}. You may not use our marks in a way that suggests
            endorsement or affiliation without permission.
          </p>

          <h3>9. Suspension and removal</h3>
          <p>
            We may remove content, block requests, disable access, or suspend
            features if we believe these Terms are violated, the service is
            being abused, the content creates risk, or legal/security reasons
            require action.
          </p>

          <h3>10. Disclaimers</h3>
          <p>
            The service is provided "as is" and "as available." To the fullest
            extent permitted by law, we disclaim warranties of merchantability,
            fitness for a particular purpose, non-infringement, availability,
            accuracy, and reliability. Nothing in these Terms limits rights or
            warranties that cannot be waived under applicable law.
          </p>

          <h3>11. Limitation of liability</h3>
          <p>
            To the fullest extent permitted by law, {companyName} and its
            directors, officers, employees, contractors, and service providers
            will not be liable for indirect, incidental, consequential,
            exemplary, special, or punitive damages, or for lost profits,
            revenue, data, goodwill, or business opportunities arising from or
            related to your use of the service. Nothing in these Terms excludes
            or limits liability that cannot be excluded or limited under
            applicable law.
          </p>

          <h3>12. Indemnity</h3>
          <p>
            To the fullest extent permitted by law, you agree to defend,
            indemnify, and hold harmless {companyName} from claims, losses,
            liabilities, damages, costs, and expenses arising from your content,
            your use of the service, or your violation of these Terms.
          </p>

          <h3>13. Applicable law</h3>
          <p>
            These Terms will be interpreted under applicable law, without
            limiting any consumer, privacy, or public-interest protections that
            cannot be waived by contract. If any provision is unenforceable, the
            remaining provisions stay in effect.
          </p>

          <h3>14. Changes and contact</h3>
          <p>
            We may update these Terms as the service evolves. The latest version
            will be posted on this page with its effective date. Questions about
            these Terms can be sent to{" "}
            <a href={`mailto:${contactEmail}`}>{contactEmail}</a>.
          </p>
        </div>
      </section>
    </main>
  );
}
