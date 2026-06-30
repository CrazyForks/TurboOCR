import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { clsx } from "clsx";
import { Copy } from "lucide-react";
import type { DocPage, ResultMap } from "../lib/doc/types";
import type { Hovered } from "./viewer/types";
import { KatexBlock } from "./KatexBlock";
import { Button, Spinner, Badge } from "./ui";

type Tab = "text" | "tables" | "formulas" | "markdown" | "json";

function TableFrame({ html }: { html: string }) {
  const doc = `<!doctype html><meta charset="utf-8"><style>
    body{margin:0;font:13px ui-sans-serif,system-ui,sans-serif;color:#18181b}
    table{border-collapse:collapse;width:100%}
    td,th{border:1px solid #d4d4d8;padding:3px 6px;vertical-align:top}
  </style>${html}`;
  return (
    <iframe
      sandbox=""
      srcDoc={doc}
      className="h-56 w-full rounded-lg border border-neutral-200 bg-white"
      title="table"
    />
  );
}

function TabButton({
  active,
  disabled,
  onClick,
  children,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className={clsx(
        "px-3 py-2 text-sm font-medium transition-colors disabled:opacity-30",
        active
          ? "border-b-2 border-indigo-600 text-neutral-900"
          : "border-b-2 border-transparent text-neutral-500 hover:text-neutral-800",
      )}
    >
      {children}
    </button>
  );
}

function PageLabel({ index, multi }: { index: number; multi: boolean }) {
  if (!multi) return null;
  return (
    <div className="sticky top-0 -mx-3 mb-1 bg-white/95 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-neutral-400">
      Page {index + 1}
    </div>
  );
}

export function Inspector({
  pages,
  results,
  onHover,
  fetchMarkdown,
}: {
  pages: DocPage[];
  results: ResultMap;
  onHover: (h: Hovered) => void;
  fetchMarkdown: () => Promise<{ markdown: string; degraded: string | null }>;
}) {
  const [tab, setTab] = useState<Tab>("text");
  const [md, setMd] = useState<string | null>(null);
  const [mdLoading, setMdLoading] = useState(false);
  const [mdErr, setMdErr] = useState<string | null>(null);

  const multi = pages.length > 1;
  const respOf = (p: DocPage) => results[p.id]?.response ?? null;
  const any = pages.some((p) => results[p.id]);
  const tableCount = pages.reduce((n, p) => n + (respOf(p)?.tables?.length ?? 0), 0);
  const formulaCount = pages.reduce((n, p) => n + (respOf(p)?.formulas?.length ?? 0), 0);

  async function loadMarkdown() {
    setMdLoading(true);
    setMdErr(null);
    try {
      const { markdown } = await fetchMarkdown();
      setMd(markdown);
    } catch (e) {
      setMdErr(e instanceof Error ? e.message : String(e));
    } finally {
      setMdLoading(false);
    }
  }

  const copyText = () =>
    navigator.clipboard?.writeText(
      pages
        .map((p) => (respOf(p)?.results ?? []).map((r) => r.text).join("\n"))
        .join("\n\n"),
    );

  return (
    <div className="flex h-full flex-col bg-white">
      <div className="flex items-center gap-1 border-b border-neutral-200 px-2">
        <TabButton active={tab === "text"} onClick={() => setTab("text")}>
          Text
        </TabButton>
        <TabButton
          active={tab === "tables"}
          disabled={tableCount === 0}
          onClick={() => setTab("tables")}
        >
          Tables {tableCount > 0 && <Badge>{tableCount}</Badge>}
        </TabButton>
        <TabButton
          active={tab === "formulas"}
          disabled={formulaCount === 0}
          onClick={() => setTab("formulas")}
        >
          Formulas {formulaCount > 0 && <Badge>{formulaCount}</Badge>}
        </TabButton>
        <TabButton active={tab === "markdown"} onClick={() => setTab("markdown")}>
          Markdown
        </TabButton>
        <TabButton active={tab === "json"} onClick={() => setTab("json")}>
          JSON
        </TabButton>
        {tab === "text" && any && (
          <button
            onClick={copyText}
            title="Copy all text"
            className="ml-auto mr-1 flex items-center gap-1 rounded-md px-2 py-1 text-xs text-neutral-500 hover:bg-neutral-100"
          >
            <Copy className="h-3.5 w-3.5" /> Copy
          </button>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-3">
        {!any && (
          <div className="mt-10 text-center text-sm text-neutral-400">
            Upload an image or PDF — OCR runs automatically.
          </div>
        )}

        {tab === "text" &&
          pages.map((p, i) => {
            const lines = respOf(p)?.results ?? [];
            if (!lines.length) return null;
            return (
              <div key={p.id} className="mb-2">
                <PageLabel index={i} multi={multi} />
                {lines.map((l, j) => (
                  <div
                    key={j}
                    onMouseEnter={() => onHover({ page: i, kind: "line", idx: j })}
                    onMouseLeave={() => onHover(null)}
                    className="flex items-start gap-2 rounded-md px-2 py-1 text-sm hover:bg-neutral-100"
                  >
                    <span className="mt-0.5 w-8 shrink-0 text-right text-xs tabular-nums text-neutral-300">
                      {l.confidence.toFixed(2)}
                    </span>
                    <span className="text-neutral-800">{l.text}</span>
                  </div>
                ))}
              </div>
            );
          })}

        {tab === "tables" &&
          pages.map((p, i) => {
            const tables = respOf(p)?.tables ?? [];
            return tables.map((t, j) => (
              <div
                key={`${p.id}-${j}`}
                className="mb-4"
                onMouseEnter={() => onHover({ page: i, kind: "table", idx: j })}
                onMouseLeave={() => onHover(null)}
              >
                <div className="mb-1 text-xs text-neutral-400">
                  {multi ? `page ${i + 1} · ` : ""}table #{j} · conf {t.confidence.toFixed(2)}
                </div>
                <TableFrame html={t.html} />
              </div>
            ));
          })}

        {tab === "formulas" &&
          pages.map((p, i) => {
            const formulas = respOf(p)?.formulas ?? [];
            return formulas.map((f, j) => (
              <div
                key={`${p.id}-${j}`}
                onMouseEnter={() => onHover({ page: i, kind: "formula", idx: j })}
                onMouseLeave={() => onHover(null)}
                className="mb-3 rounded-lg border border-neutral-200 bg-neutral-50 p-3"
              >
                <div className="mb-2 text-xs text-neutral-400">
                  {multi ? `page ${i + 1} · ` : ""}formula #{j} · conf {f.confidence.toFixed(2)}
                </div>
                <KatexBlock tex={f.latex} />
                <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-all text-xs text-neutral-400">
                  {f.latex}
                </pre>
              </div>
            ));
          })}

        {tab === "markdown" && (
          <div>
            <div className="mb-3 flex items-center gap-2">
              <Button onClick={loadMarkdown} disabled={mdLoading || !any}>
                {mdLoading ? <Spinner /> : null}
                {md ? "Re-render" : "Render markdown"}
              </Button>
              <span className="text-xs text-neutral-400">
                POST /ocr/markdown per page (GPU build, layout required).
              </span>
            </div>
            {mdErr && <div className="text-sm text-red-600">{mdErr}</div>}
            {md != null && (
              <article className="max-w-none text-sm leading-relaxed text-neutral-800 [&_h1]:mt-4 [&_h1]:text-lg [&_h1]:font-semibold [&_h2]:mt-3 [&_h2]:font-semibold [&_img]:my-2 [&_img]:max-w-full [&_table]:border-collapse [&_td]:border [&_td]:border-neutral-200 [&_td]:px-1.5">
                <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                  {md}
                </ReactMarkdown>
              </article>
            )}
          </div>
        )}

        {tab === "json" && any && (
          <pre className="overflow-auto rounded-lg bg-neutral-50 p-3 text-xs text-neutral-700 ring-1 ring-neutral-200">
            {JSON.stringify(
              pages.map((p, i) => ({ page: i, ...(respOf(p) ?? {}) })),
              null,
              2,
            )}
          </pre>
        )}
      </div>
    </div>
  );
}
