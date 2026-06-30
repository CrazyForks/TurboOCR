import { pdfjsLib } from "./pdfjs";
import { canvasToPng, loadImage } from "./raster";

export interface RenderedRaster {
  url: string;
  bytes: ArrayBuffer;
}
export interface PageSize {
  w: number;
  h: number;
}

// A document exposes page SIZES immediately (cheap metadata) so the viewer can
// show page slots instantly, and rasterizes each page lazily/on demand
// (renderPage) so the first page appears fast and the rest stream in.
export interface DocSource {
  name: string;
  kind: "image" | "pdf";
  pageSizes: PageSize[];
  renderPage: (i: number) => Promise<RenderedRaster>;
  destroy: () => void;
}

// ~108 DPI: fast to rasterize + small PNGs (quicker OCR upload) while staying
// close to the server's default 100 DPI for accuracy.
const PDF_SCALE = 1.5;

function isPdf(file: File, head: Uint8Array): boolean {
  if (file.type === "application/pdf") return true;
  return head[0] === 0x25 && head[1] === 0x50 && head[2] === 0x44 && head[3] === 0x46;
}

export async function openDoc(file: File): Promise<DocSource> {
  const buf = await file.arrayBuffer();
  const head = new Uint8Array(buf.slice(0, 5));
  return isPdf(file, head) ? openPdf(file, buf) : openImage(file, buf);
}

async function openImage(file: File, buf: ArrayBuffer): Promise<DocSource> {
  const objectUrl = URL.createObjectURL(new Blob([buf], { type: file.type || "image/png" }));
  const img = await loadImage(objectUrl);
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  let cached: RenderedRaster | null = null;
  return {
    name: file.name,
    kind: "image",
    pageSizes: [{ w, h }],
    renderPage: async () => {
      if (cached) return cached;
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("no 2d canvas context");
      ctx.drawImage(img, 0, 0);
      cached = await canvasToPng(canvas);
      return cached;
    },
    destroy: () => URL.revokeObjectURL(objectUrl),
  };
}

async function openPdf(file: File, buf: ArrayBuffer): Promise<DocSource> {
  const pdf = await pdfjsLib.getDocument({ data: buf.slice(0) }).promise;
  const pageSizes: PageSize[] = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const p = await pdf.getPage(i);
    const vp = p.getViewport({ scale: PDF_SCALE });
    pageSizes.push({ w: Math.ceil(vp.width), h: Math.ceil(vp.height) });
    p.cleanup();
  }
  return {
    name: file.name,
    kind: "pdf",
    pageSizes,
    renderPage: async (i: number) => {
      const p = await pdf.getPage(i + 1);
      const vp = p.getViewport({ scale: PDF_SCALE });
      const canvas = document.createElement("canvas");
      canvas.width = Math.ceil(vp.width);
      canvas.height = Math.ceil(vp.height);
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("no 2d canvas context");
      await p.render({ canvasContext: ctx, viewport: vp }).promise;
      const r = await canvasToPng(canvas);
      p.cleanup();
      return r;
    },
    destroy: () => {
      void pdf.destroy();
    },
  };
}
