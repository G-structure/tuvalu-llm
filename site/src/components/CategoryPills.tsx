import { A, useLocation } from "@solidjs/router";
import { For } from "solid-js";
import type { Category } from "~/lib/types";

interface CategoryPillsProps {
  categories: Category[];
}

export default function CategoryPills(props: CategoryPillsProps) {
  const location = useLocation();

  const isActive = (slug: string) => {
    return location.pathname === `/category/${slug}`;
  };

  const isAll = () => {
    return location.pathname === "/";
  };

  return (
    <div class="category-scroll category-pills">
      <A
        href="/"
        class={`category-pill ${
          isAll()
            ? "category-pill--active"
            : ""
        }`}
      >
        Katoa
      </A>
      <For each={props.categories}>
        {(cat) => (
          <A
            href={`/category/${cat.slug}`}
            class={`category-pill capitalize ${
              isActive(cat.slug)
                ? "category-pill--active"
                : ""
            }`}
          >
            {cat.slug.replace(/-/g, " ")}
          </A>
        )}
      </For>
    </div>
  );
}
