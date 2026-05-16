import { createSignal } from "solid-js";

export type LanguageMode = "tv" | "en" | "tv+en";

const MODE_LABELS: Record<LanguageMode, string> = {
  tv: "TV",
  en: "EN",
  "tv+en": "TV+EN",
};

const MODE_CYCLE: LanguageMode[] = ["tv", "en", "tv+en"];

interface LanguageToggleProps {
  mode: LanguageMode;
  onChange: (mode: LanguageMode) => void;
}

export default function LanguageToggle(props: LanguageToggleProps) {
  const next = () => {
    const idx = MODE_CYCLE.indexOf(props.mode);
    return MODE_CYCLE[(idx + 1) % MODE_CYCLE.length];
  };

  return (
    <button
      onClick={() => props.onChange(next())}
      class="language-toggle"
      aria-label={`Language: ${MODE_LABELS[props.mode]}. Tap to switch.`}
    >
      {MODE_LABELS[props.mode]}
    </button>
  );
}
