import { createAsync, cache, A } from "@solidjs/router";
import { Show } from "solid-js";
import { getFateleTeaserCount } from "~/lib/db";

const loadTeaser = cache(async () => {
  "use server";
  return await getFateleTeaserCount();
}, "fatele-teaser");

export default function FateleTeaser() {
  const count = createAsync(() => loadTeaser());

  return (
    <div class="fatele-teaser">
      <A
        href="/fatele"
        class="fatele-teaser__link"
      >
        <span>
          Kominiti
          <Show when={typeof count() === "number"}>
            {" "}&middot; {count()} i te masina nei
          </Show>
        </span>
        <span aria-hidden="true">&rarr;</span>
      </A>
    </div>
  );
}
