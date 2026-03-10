import { createAsync, cache } from "@solidjs/router";
import { For, Show } from "solid-js";
import { getFateleStats } from "~/lib/db";
import type { FateleStats } from "~/lib/types";
import { ISLANDS } from "~/lib/types";
import OGMeta from "~/components/OGMeta";

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

  return (
    <main class="max-w-3xl mx-auto pb-16 px-4">
      <OGMeta
        title="Te Fatele | TALAFUTIPOLO"
        description="Community dashboard — help translate football news into Tuvaluan"
      />

      <div class="pt-6 pb-4 text-center">
        <h1 class="text-2xl font-bold text-gray-900">Te Fatele</h1>
        <p class="mt-1 text-sm text-gray-500">
          Te galuega a te kominiti — Community effort
        </p>
      </div>

      {/* Hero stat */}
      <Show when={stats()}>
        {(s) => (
          <div class="bg-[var(--ocean-deep)] text-white rounded-xl p-6 text-center mb-6">
            <div class="text-4xl font-bold text-[var(--gold)]">{s().total_this_month}</div>
            <div class="text-sm text-[var(--sky-dark)] mt-1">
              fakailoga i te masina nei (signals this month)
            </div>
          </div>
        )}
      </Show>

      {/* Island progress bars */}
      <div class="space-y-3">
        <For each={islandData()}>
          {(d) => (
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span class="font-medium text-gray-700">{d.island}</span>
                <span class="text-gray-400">{d.count}</span>
              </div>
              <div class="h-3 bg-[var(--sky-dark)] rounded-full overflow-hidden">
                <div
                  class="h-full bg-[var(--gold)] rounded-full transition-all duration-500"
                  style={{ width: `${(d.count / maxCount()) * 100}%` }}
                />
              </div>
            </div>
          )}
        </For>
      </div>

      {/* Help text */}
      <div class="mt-8 p-4 bg-white rounded-xl text-sm text-gray-600 leading-relaxed border border-[var(--sky-dark)]">
        <p class="font-medium text-gray-900 mb-2">Pefea e fesoasoani ai?</p>
        <p>
          Faitau tala i te gagana Tuvalu. Kapiti te 👍🏾 pe afai e tonu te
          kupu, pe te 👎🏾 pe afai e seki tonu. Tou fakailoga e fesoasoani ki
          te fakalelei o te masini liliu.
        </p>
        <p class="mt-2 text-gray-400">
          Read articles in Tuvaluan. Tap 👍🏾 if a translation sounds good, or
          👎🏾 if it sounds wrong. Your feedback helps improve the translation
          model.
        </p>
      </div>
    </main>
  );
}
