import { For } from "solid-js";

type StructuredDataRecord = Record<string, unknown>;

export default function StructuredData(props: { data: StructuredDataRecord | Array<StructuredDataRecord> }) {
  const records = () => Array.isArray(props.data) ? props.data : [props.data];
  // Escape closing script tags to prevent XSS via JSON-LD injection.
  const safeJson = (record: StructuredDataRecord) =>
    JSON.stringify(record).replace(/<\/script/gi, "<\\/script");

  return (
    <For each={records()}>
      {(record) => (
        <script
          type="application/ld+json"
          innerHTML={safeJson(record)}
        />
      )}
    </For>
  );
}
