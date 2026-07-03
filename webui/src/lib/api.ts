import type { Capabilities, OcrResponse, RunOptions } from "./types";

// Everything goes through the same-origin /api prefix (proxied to the
// TurboOCR server by Vite in dev / nginx in prod), so there is no CORS.
const BASE = "/api";

async function fail(r: Response): Promise<never> {
  let detail = `${r.status} ${r.statusText}`;
  try {
    const j = await r.json();
    if (j?.error?.code) detail = `${j.error.code}: ${j.error.message ?? ""}`;
  } catch {
    /* non-JSON body */
  }
  throw new Error(detail);
}

export async function getCapabilities(): Promise<Capabilities> {
  const r = await fetch(`${BASE}/capabilities`);
  if (!r.ok) return fail(r);
  return r.json();
}

export interface OcrRun {
  response: OcrResponse;
  elapsedMs: number;
}

export async function runOcr(
  bytes: ArrayBuffer,
  contentType: string,
  opts: RunOptions,
): Promise<OcrRun> {
  const q = new URLSearchParams();
  // reading_order auto-enables layout server-side and gives us reading order
  // for the text panel whenever layout is on.
  if (opts.layout) {
    q.set("layout", "1");
    q.set("reading_order", "1");
  }
  if (opts.tables) q.set("tables", "1");
  if (opts.formulas) q.set("formulas", "1");

  const t0 = performance.now();
  const r = await fetch(`${BASE}/ocr/raw?${q.toString()}`, {
    method: "POST",
    headers: { "Content-Type": contentType || "application/octet-stream" },
    body: bytes,
  });
  if (!r.ok) return fail(r);
  const response = (await r.json()) as OcrResponse;
  return { response, elapsedMs: performance.now() - t0 };
}

export interface PdfPage {
  index: number;
  w: number;
  h: number;
  url: string; // object URL of the page image ("" when images weren't requested)
  bytes: ArrayBuffer | null;
  response: OcrResponse;
}
export interface PdfRun {
  pages: PdfPage[];
  elapsedMs: number;
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// OCR a WHOLE PDF in ONE request. The server rasterizes natively and pipelines
// the pages, returning every page's results + (optionally) its image inline —
// far faster than the browser rasterizing each page with pdf.js and firing N
// separate /ocr requests. width/height are the server's pixel space, identical
// to the bounding-box coordinates, so overlays align exactly.
export async function runOcrPdf(
  bytes: ArrayBuffer,
  opts: RunOptions,
  withImages: boolean,
): Promise<PdfRun> {
  const q = new URLSearchParams();
  if (opts.layout) {
    q.set("layout", "1");
    q.set("reading_order", "1");
  }
  if (opts.tables) q.set("tables", "1");
  if (opts.formulas) q.set("formulas", "1");
  if (withImages) {
    // JPEG: small payload, embeds directly into the searchable PDF (pdf-lib
    // embedJpg) and is decodable by the server if re-sent (markdown export).
    q.set("images", "inline");
    q.set("format", "jpeg");
    q.set("quality", "85");
  }

  const t0 = performance.now();
  const r = await fetch(`${BASE}/ocr/pdf?${q.toString()}`, {
    method: "POST",
    headers: { "Content-Type": "application/pdf" },
    body: bytes,
  });
  if (!r.ok) return fail(r);
  const data = (await r.json()) as { pages?: Array<Record<string, unknown>> };
  const pages: PdfPage[] = (data.pages ?? []).map((p) => {
    const { image_b64, image_content_type, ...resp } = p as unknown as {
      image_b64?: string;
      image_content_type?: string;
      page?: number;
      page_index?: number;
      width?: number;
      height?: number;
    } & OcrResponse;
    let url = "";
    let buf: ArrayBuffer | null = null;
    if (image_b64) {
      const u8 = base64ToBytes(image_b64);
      // Copy into a plain ArrayBuffer: u8.buffer is typed ArrayBufferLike
      // (could be a SharedArrayBuffer as of TS 5.7's typed-array generics),
      // which neither PdfPage.bytes nor BlobPart accepts.
      const ab = new ArrayBuffer(u8.byteLength);
      new Uint8Array(ab).set(u8);
      buf = ab;
      url = URL.createObjectURL(new Blob([ab], { type: image_content_type || "image/jpeg" }));
    }
    return {
      index: resp.page_index ?? (resp.page ? resp.page - 1 : 0),
      w: resp.width ?? 0,
      h: resp.height ?? 0,
      url,
      bytes: buf,
      response: resp as OcrResponse,
    };
  });
  return { pages, elapsedMs: performance.now() - t0 };
}

export async function runMarkdown(
  bytes: ArrayBuffer,
  contentType: string,
): Promise<{ markdown: string; degraded: string | null }> {
  const r = await fetch(`${BASE}/ocr/markdown`, {
    method: "POST",
    headers: { "Content-Type": contentType || "application/octet-stream" },
    body: bytes,
  });
  if (!r.ok) return fail(r);
  const markdown = await r.text();
  return { markdown, degraded: r.headers.get("X-OCR-Degraded") };
}
