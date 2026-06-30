import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FileUp } from "lucide-react";
import { Header } from "./components/Header";
import { Controls, type Aggregate } from "./components/Controls";
import { DocViewer } from "./components/viewer/DocViewer";
import { Inspector } from "./components/Inspector";
import type { Hovered, Layers } from "./components/viewer/types";
import { useLocalStorage } from "./lib/useLocalStorage";
import { getCapabilities, runMarkdown, runOcr, runOcrPdf } from "./lib/api";
import { openDoc, type DocSource } from "./lib/pdf/docSource";
import { saveFile, saveSnapshot, loadState, type RasterRec } from "./lib/persist";
import { exportSearchablePdf, exportText, exportJson } from "./lib/export/exporters";
import type { DocModel, RasterMap, ResultMap } from "./lib/doc/types";
import type { Capabilities, RunOptions } from "./lib/types";

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

const sniffImageType = (buf: ArrayBuffer): string => {
  const b = new Uint8Array(buf);
  if (b[0] === 0xff && b[1] === 0xd8) return "image/jpeg";
  if (b[0] === 0x89 && b[1] === 0x50) return "image/png";
  if (b[0] === 0x52 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x46) return "image/webp";
  return "application/octet-stream";
};

export default function App() {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [capsErr, setCapsErr] = useState<string | null>(null);
  const [doc, setDoc] = useState<DocModel | null>(null);
  const [rasters, setRasters] = useState<RasterMap>({});
  const [results, setResults] = useState<ResultMap>({});
  const [opts, setOpts] = useLocalStorage<RunOptions>("turbo.opts", {
    layout: true,
    tables: false,
    formulas: false,
  });
  const [selectText, setSelectText] = useLocalStorage("turbo.selectText", true);
  const [layers, setLayers] = useState<Layers>({
    lines: true,
    layout: true,
    tables: true,
    formulas: true,
  });
  const [hovered, setHovered] = useState<Hovered>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingDoc, setLoadingDoc] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  // True until the persisted-session lookup finishes, so we don't flash the
  // upload dropzone for a frame before a cached document is restored.
  const [restoring, setRestoring] = useState(true);

  const sourceRef = useRef<DocSource | null>(null);
  const rastersRef = useRef<RasterMap>(rasters);
  rastersRef.current = rasters;
  const resultsRef = useRef<ResultMap>(results);
  resultsRef.current = results;
  const docRef = useRef<DocModel | null>(doc);
  docRef.current = doc;
  const renderSeq = useRef(0);
  const ocrDone = useRef<Set<number>>(new Set());
  const pdfBytesRef = useRef<ArrayBuffer | null>(null);
  const [pdfKey, setPdfKey] = useState(0);
  const fileRef = useRef<File | null>(null);
  // When set to a serialized opts string, the current `results` were restored
  // from cache for those opts — skip the OCR/reset effects so a refresh doesn't
  // re-OCR. Cleared on a new file or when opts actually change.
  const hydratedRef = useRef<string | null>(null);

  useEffect(() => {
    getCapabilities()
      .then(setCaps)
      .catch((e) => setCapsErr(msg(e)));
  }, []);

  useEffect(() => {
    if (!caps) return;
    setOpts((o) => ({
      layout: o.layout && caps.features.layout,
      tables: o.tables && caps.features.tables,
      formulas: o.formulas && caps.features.formulas,
    }));
  }, [caps, setOpts]);

  // IMAGE path only. PDFs are handled by the single-request effect below (the
  // server rasterizes + OCRs the whole document in one shot), so there is no
  // client-side rendering pool or per-page OCR for them.
  //
  // Stream page rasters with a concurrency pool so several pages render at once
  // and their OCR requests fire in parallel almost immediately, instead of OCR
  // being starved by serial rendering. Each completed raster fills its slot +
  // kicks OCR (the OCR effect below).
  useEffect(() => {
    const src = sourceRef.current;
    if (!doc || doc.kind !== "image" || !src) return;
    const seq = ++renderSeq.current;
    let cancelled = false;
    const pages = doc.pages;
    let next = 0;
    const RENDER_CONCURRENCY = 4;
    const worker = async () => {
      while (!cancelled && seq === renderSeq.current) {
        const i = next++;
        if (i >= pages.length) return;
        const id = pages[i].id;
        try {
          const r = await src.renderPage(i);
          if (cancelled || seq !== renderSeq.current) {
            URL.revokeObjectURL(r.url);
            return;
          }
          setRasters((m) => ({ ...m, [id]: r }));
        } catch (e) {
          if (!cancelled)
            setResults((rr) => ({ ...rr, [id]: { status: "error", response: null, error: msg(e) } }));
        }
      }
    };
    void Promise.all(Array.from({ length: Math.min(RENDER_CONCURRENCY, pages.length) }, worker));
    return () => {
      cancelled = true;
    };
  }, [doc]);

  // Reset OCR bookkeeping whenever the IMAGE document or stage toggles change.
  // (PDFs manage their own results via the single-request effect below.) Skip
  // when results were just restored from cache for the current opts.
  useEffect(() => {
    if (doc?.kind !== "image") return;
    if (hydratedRef.current === JSON.stringify(opts)) return;
    hydratedRef.current = null; // moved off the restored opts — forget the cache
    ocrDone.current = new Set();
    setResults({});
  }, [doc, opts]);

  // OCR each page once its raster is available; re-OCRs all on a toggle change.
  useEffect(() => {
    if (!doc || doc.kind !== "image") return;
    for (const p of doc.pages) {
      const raster = rasters[p.id];
      if (!raster || ocrDone.current.has(p.id)) continue;
      ocrDone.current.add(p.id);
      setResults((r) => ({ ...r, [p.id]: { status: "running", response: r[p.id]?.response ?? null } }));
      runOcr(raster.bytes, "image/png", opts)
        .then(({ response, elapsedMs }) =>
          setResults((r) => ({ ...r, [p.id]: { status: "done", response, elapsedMs } })),
        )
        .catch((e) =>
          setResults((r) => ({ ...r, [p.id]: { status: "error", response: null, error: msg(e) } })),
        );
    }
  }, [doc, opts, rasters]);

  // PDF path: one request OCRs + rasterizes the whole document server-side and
  // returns every page at once. Runs on a new PDF (pdfKey) and on a stage toggle
  // (opts); the first run pulls page images inline, later toggles refresh only
  // the results so the page images don't reload/flash.
  useEffect(() => {
    const buf = pdfBytesRef.current;
    if (doc?.kind !== "pdf" || !buf) return;
    if (hydratedRef.current === JSON.stringify(opts)) return; // restored from cache
    hydratedRef.current = null; // moved off the restored opts — forget the cache
    let cancelled = false;
    const needImages = Object.keys(rastersRef.current).length === 0;
    if (needImages) setLoadingDoc(true);
    else
      setResults((prev) => {
        const next: ResultMap = {};
        for (const k of Object.keys(prev)) next[+k] = { ...prev[+k], status: "running" };
        return next;
      });
    runOcrPdf(buf, opts, needImages)
      .then(({ pages, elapsedMs }) => {
        if (cancelled) return;
        if (needImages) {
          setDoc((d) => (d ? { ...d, pages: pages.map((p) => ({ id: p.index, w: p.w, h: p.h })) } : d));
          const rast: RasterMap = {};
          for (const p of pages) if (p.url && p.bytes) rast[p.index] = { url: p.url, bytes: p.bytes };
          setRasters(rast);
        }
        const res: ResultMap = {};
        for (const p of pages) res[p.index] = { status: "done", response: p.response, elapsedMs };
        setResults(res);
      })
      .catch((e) => {
        if (!cancelled) setError(msg(e));
      })
      .finally(() => {
        if (!cancelled) setLoadingDoc(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pdfKey, opts]);

  const onFile = useCallback(async (file: File) => {
    setError(null);
    setResults({});
    setRasters({});
    setLoadingDoc(true);
    sourceRef.current?.destroy();
    sourceRef.current = null;
    Object.values(rastersRef.current).forEach((r) => URL.revokeObjectURL(r.url));
    pdfBytesRef.current = null;
    ocrDone.current = new Set();
    fileRef.current = file;
    hydratedRef.current = null; // a fresh file always (re-)OCRs
    void saveFile(file); // restore-only-doc fallback until the full snapshot lands
    try {
      const buf = await file.arrayBuffer();
      const head = new Uint8Array(buf.slice(0, 5));
      const isPdf =
        file.type === "application/pdf" ||
        (head[0] === 0x25 && head[1] === 0x50 && head[2] === 0x44 && head[3] === 0x46);
      if (isPdf) {
        // The PDF effect (keyed on pdfKey) fills pages + rasters + results.
        pdfBytesRef.current = buf;
        setDoc({ name: file.name, kind: "pdf", pages: [] });
        setPdfKey((k) => k + 1);
      } else {
        const src = await openDoc(file);
        sourceRef.current = src;
        setDoc({
          name: src.name,
          kind: src.kind,
          pages: src.pageSizes.map((s, i) => ({ id: i, w: s.w, h: s.h })),
        });
        setLoadingDoc(false);
      }
    } catch (e) {
      setError(msg(e));
      setLoadingDoc(false);
    }
  }, []);

  // Restore the last session on load (survives a page refresh). A full snapshot
  // restores the document + page images + OCR results with NO re-render and NO
  // re-OCR; a file-only record (reload mid-processing) falls back to re-running.
  useEffect(() => {
    let cancelled = false;
    loadState()
      .then(async (st) => {
        if (cancelled || !st) return;
        const file = new File([st.file], st.name, { type: st.file.type });
        if (!st.full || JSON.stringify(st.opts) !== JSON.stringify(opts)) {
          onFile(file); // re-OCR (no cached results, or cached for other opts)
          return;
        }
        setLoadingDoc(false);
        // Full restore — rebuild state directly, suppress the OCR effects.
        fileRef.current = file;
        if (st.kind === "pdf") pdfBytesRef.current = await st.file.arrayBuffer();
        const rast: RasterMap = {};
        for (const r of st.rasters)
          rast[r.id] = { url: URL.createObjectURL(new Blob([r.bytes], { type: r.type })), bytes: r.bytes };
        const res: ResultMap = {};
        for (const r of st.results)
          res[r.id] = { status: "done", response: r.response, elapsedMs: r.elapsedMs };
        ocrDone.current = new Set(st.pages.map((p) => p.id));
        hydratedRef.current = JSON.stringify(st.opts);
        setRasters(rast);
        setResults(res);
        setDoc({ name: st.name, kind: st.kind, pages: st.pages.map((p) => ({ id: p.id, w: p.w, h: p.h })) });
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setRestoring(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onFile]);

  // Persist the full session once every page is done, so the next reload skips
  // rendering and OCR entirely.
  useEffect(() => {
    const d = doc;
    const file = fileRef.current;
    if (!d || !file || d.pages.length === 0) return;
    if (!d.pages.every((p) => results[p.id]?.status === "done")) return;
    const rasterRecs: RasterRec[] = [];
    for (const p of d.pages) {
      const r = rastersRef.current[p.id];
      if (r) rasterRecs.push({ id: p.id, bytes: r.bytes, type: sniffImageType(r.bytes) });
    }
    void saveSnapshot({
      file,
      name: d.name,
      kind: d.kind,
      opts,
      pages: d.pages.map((p) => ({ id: p.id, w: p.w, h: p.h })),
      rasters: rasterRecs,
      results: d.pages.map((p) => ({
        id: p.id,
        response: results[p.id]?.response ?? null,
        elapsedMs: results[p.id]?.elapsedMs,
      })),
    });
  }, [doc, results, opts]);

  const fetchMarkdown = useCallback(async () => {
    const d = docRef.current;
    if (!d) throw new Error("No document loaded");
    const parts: string[] = [];
    for (const p of d.pages) {
      const raster = rastersRef.current[p.id];
      if (!raster) continue;
      const b = new Uint8Array(raster.bytes);
      const ct = b[0] === 0xff && b[1] === 0xd8 ? "image/jpeg" : "image/png";
      const { markdown } = await runMarkdown(raster.bytes, ct);
      parts.push(markdown);
    }
    return { markdown: parts.join("\n\n---\n\n"), degraded: null };
  }, []);

  const onExportPdf = useCallback(async () => {
    const d = docRef.current;
    if (!d) return;
    setExporting(true);
    try {
      await exportSearchablePdf(d.name, d.pages, rastersRef.current, resultsRef.current);
    } catch (e) {
      setError(msg(e));
    } finally {
      setExporting(false);
    }
  }, []);
  const onExportText = useCallback(() => {
    const d = docRef.current;
    if (d) exportText(d.name, d.pages, resultsRef.current);
  }, []);
  const onExportJson = useCallback(() => {
    const d = docRef.current;
    if (d) exportJson(d.name, d.pages, resultsRef.current);
  }, []);

  const agg = useMemo<Aggregate>(() => {
    const pages = doc?.pages ?? [];
    let layout = false;
    let tables = false;
    let formulas = false;
    let done = 0;
    let running = 0;
    const degraded = new Set<string>();
    for (const p of pages) {
      const r = results[p.id];
      if (!r) continue;
      if (r.status === "running") running++;
      if (r.status === "done") done++;
      const resp = r.response;
      if (resp?.layout?.length) layout = true;
      if (resp?.tables?.length) tables = true;
      if (resp?.formulas?.length) formulas = true;
      if (resp?.text_degraded) degraded.add("text");
      if (resp?.table_degraded) degraded.add("table");
      if (resp?.formula_degraded) degraded.add("formula");
    }
    return { layout, tables, formulas, done, running, total: pages.length, degraded: [...degraded] };
  }, [doc, results]);

  return (
    <div className="flex h-full flex-col bg-neutral-50 text-neutral-900">
      <Header capsErr={capsErr} />
      <Controls
        onFile={onFile}
        fileName={doc?.name ?? null}
        caps={caps}
        opts={opts}
        setOpts={setOpts}
        layers={layers}
        setLayers={setLayers}
        selectText={selectText}
        setSelectText={setSelectText}
        agg={agg}
        onExportPdf={onExportPdf}
        onExportText={onExportText}
        onExportJson={onExportJson}
        exporting={exporting}
        hasDoc={!!doc}
      />

      {error && (
        <div className="border-b border-red-200 bg-red-50 px-5 py-2 text-sm text-red-700">{error}</div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-[1fr_420px]">
        <div
          className="relative min-h-0"
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) onFile(f);
          }}
        >
          {doc ? (
            <DocViewer
              pages={doc.pages}
              rasters={rasters}
              results={results}
              layers={layers}
              selectText={selectText}
              hovered={hovered}
            />
          ) : restoring ? (
            <div className="flex h-full items-center justify-center bg-neutral-50" />
          ) : (
            <div className="flex h-full items-center justify-center bg-neutral-50 p-8">
              <label className="flex cursor-pointer flex-col items-center gap-3 rounded-2xl border-2 border-dashed border-neutral-300 bg-white px-16 py-14 text-center transition-colors hover:border-indigo-300 hover:bg-indigo-50/40">
                <FileUp className="h-8 w-8 text-neutral-400" />
                <div className="font-serif text-[15px] text-neutral-700">
                  Drop an image or PDF here, or click to upload
                </div>
                <div className="font-mono text-[10px] uppercase tracking-[0.08em] text-neutral-400">
                  OCR runs automatically · PNG · JPEG · WebP · TIFF · PDF
                </div>
                <input
                  type="file"
                  accept="image/*,application/pdf"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onFile(f);
                  }}
                />
              </label>
            </div>
          )}

          {loadingDoc && (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/70 backdrop-blur-sm">
              <span className="rounded-lg bg-white px-4 py-2 text-sm font-medium text-neutral-700 shadow-sm ring-1 ring-neutral-200">
                Opening…
              </span>
            </div>
          )}

          {dragOver && (
            <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center bg-indigo-500/5 ring-2 ring-inset ring-indigo-400">
              <span className="rounded-lg bg-white px-4 py-2 text-sm font-medium text-indigo-700 shadow-sm">
                Drop to OCR
              </span>
            </div>
          )}
        </div>

        <aside className="min-h-0 border-l border-neutral-200">
          <Inspector
            pages={doc?.pages ?? []}
            results={results}
            onHover={setHovered}
            fetchMarkdown={fetchMarkdown}
          />
        </aside>
      </div>
    </div>
  );
}
