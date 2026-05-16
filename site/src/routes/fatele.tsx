import { createAsync, cache } from "@solidjs/router";
import { For, Show } from "solid-js";
import { getFateleStats } from "~/lib/db";
import type { FateleStats } from "~/lib/types";
import { ISLANDS } from "~/lib/types";
import OGMeta from "~/components/OGMeta";
import { absoluteFootballUrl, FOOTBALL_META, SITE_ORIGINS } from "~/lib/site";

const loadFatele = cache(async () => {
  "use server";
  return await getFateleStats();
}, "fatele");

export const route = {
  load: () => loadFatele(),
};

export default function FatelePage() {
  const stats = createAsync(() => loadFatele());

  const islandData = () => {
    const s = stats();
    if (!s) return ISLANDS.map((name) => ({ island: name, count: 0 }));
    const map = new Map(s.islands.map((i) => [i.island, i.count]));
    return ISLANDS.map((name) => ({
      island: name,
      count: map.get(name) || 0,
    }));
  };

  const maxCount = () => Math.max(1, ...islandData().map((d) => d.count));
  const modePrefs = () => stats()?.mode_preferences || [];
  const maxModeCount = () => Math.max(1, ...modePrefs().map((d) => d.count), 1);

  return (
    <main class="site-page lagoon-subpage community-dashboard-page">
      <OGMeta
        title="Kominiti"
        description="Community dashboard — help translate football news into Tuvaluan"
        url={absoluteFootballUrl("/fatele")}
        image={FOOTBALL_META.communityOgImage}
        imageOrigin={SITE_ORIGINS.football}
        imageWidth={FOOTBALL_META.defaultOgImageWidth}
        imageHeight={FOOTBALL_META.defaultOgImageHeight}
        imageAlt="Talafutipolo community social card for football translation work."
        siteName={FOOTBALL_META.productName}
        titleSuffix={FOOTBALL_META.productName}
      />

      <section class="site-hero site-hero--compact lagoon-subhero">
        <div class="site-shell site-shell--wide lagoon-subhero__grid">
          <div>
            <p class="site-kicker">Kominiti</p>
            <h1 class="site-title">Kominiti signal room.</h1>
            <p class="site-lede">
              Real reading notes from across the islands, shaped into
              translation data that can help Fenua sound more natural in
              Tuvaluan and English.
            </p>
          </div>
          <Show when={stats()}>
            {(s) => (
              <aside class="lagoon-subhero__panel">
                <span>This month</span>
                <strong>{s().total_this_month}</strong>
                <em>community signals</em>
              </aside>
            )}
          </Show>
        </div>
      </section>

      <div class="site-shell site-shell--wide community-dashboard-shell">
        <Show when={stats()}>
          {(s) => (
            <section class="community-overview">
              <div class="community-overview__copy">
                <p class="site-kicker">This month</p>
                <h2>{s().total_this_month} signals collected</h2>
                <p>
                  Each vote, note, and correction becomes a reusable review
                  item for improving Tuvaluan translations.
                </p>
              </div>

              <div class="community-metrics" aria-label="Community signal counts">
                <div class="community-metric">
                  <strong>{s().article_feedback_count}</strong>
                  <span>coach notes</span>
                </div>
                <div class="community-metric">
                  <strong>{s().corrections_count}</strong>
                  <span>corrections</span>
                </div>
                <div class="community-metric">
                  <strong>{s().helpful_yes}</strong>
                  <span>helpful votes</span>
                </div>
                <div class="community-metric">
                  <strong>{s().helpful_no}</strong>
                  <span>needs-work votes</span>
                </div>
              </div>
            </section>
          )}
        </Show>

        <div class="community-grid">
          <section class="community-panel">
            <div class="community-panel__head">
              <div>
                <p class="site-kicker">Participation</p>
                <h2>Island signals</h2>
              </div>
              <span>{ISLANDS.length} islands</span>
            </div>
            <div class="community-bars">
              <For each={islandData()}>
                {(d) => (
                  <div class="community-bar-row">
                    <div>
                      <span>{d.island}</span>
                      <span>{d.count}</span>
                    </div>
                    <div class="community-bar">
                      <div
                        class="community-bar__fill community-bar__fill--gold"
                        style={{ width: `${(d.count / maxCount()) * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </For>
            </div>
          </section>

          <section class="community-panel">
            <div class="community-panel__head">
              <div>
                <p class="site-kicker">Reading</p>
                <h2>Mode preference</h2>
              </div>
              <span>TV / EN</span>
            </div>
            <Show
              when={modePrefs().length > 0}
              fallback={
                <div class="community-empty">
                  No mode votes yet. Open any article and submit a coaching note.
                </div>
              }
            >
              <div class="community-bars community-bars--mode">
                <For each={modePrefs()}>
                  {(d) => (
                    <div class="community-bar-row">
                      <div>
                        <span>{d.mode}</span>
                        <span>{d.count}</span>
                      </div>
                      <div class="community-bar">
                        <div
                          class="community-bar__fill"
                          style={{ width: `${(d.count / maxModeCount()) * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                </For>
              </div>
            </Show>

            <div class="community-data-card">
              <span>Training value</span>
              <strong>Preference + correction review</strong>
              <p>
                These signals can be exported as supervised corrections and
                ranking examples for future tuning.
              </p>
            </div>
          </section>
        </div>

        <section class="community-help">
          <div>
            <p class="site-kicker">How to help</p>
            <h2>Pefea e fesoasoani ai?</h2>
          </div>
          <div class="community-help__grid">
            <p>
              Faitau tala i te gagana Tuvalu. Togi se vote māfai e tonu te
              kupu, pe fakailoa mai māfai e seki tonu. Tusi mai se fakaleiga
              fou māfai e isi sau manatu.
            </p>
            <p>
              Read articles in Tuvaluan, vote on whether a translation sounds
              right, and leave a short correction when it needs work. The
              dashboard turns that community knowledge into training-ready
              review data.
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}
