import { PDFDocument, StandardFonts, setTextRenderingMode, TextRenderingMode, type PDFImage } from "pdf-lib";
import { aabb } from "../geometry";
import type { OcrResponse } from "../types";

export interface ExportPage {
  w: number;
  h: number;
  bytes: ArrayBuffer; // page raster (PNG, JPEG, or WebP)
  response: OcrResponse | null;
}

const isPng = (b: Uint8Array) =>
  b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47;
const isJpeg = (b: Uint8Array) => b[0] === 0xff && b[1] === 0xd8;

// pdf-lib embeds only PNG/JPEG. The server now returns WebP page images, so
// transcode anything else to PNG via a canvas before embedding.
async function embedRaster(doc: PDFDocument, bytes: ArrayBuffer): Promise<PDFImage> {
  const u8 = new Uint8Array(bytes);
  if (isPng(u8)) return doc.embedPng(bytes);
  if (isJpeg(u8)) return doc.embedJpg(bytes);
  const bmp = await createImageBitmap(new Blob([u8 as BlobPart]));
  const canvas = document.createElement("canvas");
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  canvas.getContext("2d")?.drawImage(bmp, 0, 0);
  bmp.close?.();
  const blob: Blob | null = await new Promise((res) => canvas.toBlob(res, "image/png"));
  if (!blob) throw new Error("page image transcode failed");
  return doc.embedPng(await blob.arrayBuffer());
}

// Helvetica's WinAnsi encoding can't represent every glyph; replace anything
// outside it with a space so a stray char never aborts the whole export.
function winAnsiSafe(s: string): string {
  let out = "";
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    out += c >= 0x20 && c <= 0xff && c !== 0x7f ? ch : " ";
  }
  return out;
}

// Build a searchable "sandwich" PDF: each page is the raster image with a real
// OCR text layer drawn in INVISIBLE text-rendering mode (Tr 3) positioned per
// line — selectable/searchable in every PDF viewer. PDF origin is bottom-left,
// hence the Y-flip. Tr 3 persists across subsequent drawText calls, so it is
// pushed once per page after the image.
export async function buildSearchablePdf(pages: ExportPage[]): Promise<Uint8Array> {
  const doc = await PDFDocument.create();
  const font = await doc.embedFont(StandardFonts.Helvetica);

  for (const p of pages) {
    const img = await embedRaster(doc, p.bytes);
    const page = doc.addPage([p.w, p.h]);
    page.drawImage(img, { x: 0, y: 0, width: p.w, height: p.h });
    page.pushOperators(setTextRenderingMode(TextRenderingMode.Invisible));

    for (const line of p.response?.results ?? []) {
      if (!line.text) continue;
      const r = aabb(line.bounding_box);
      if (r.w <= 0 || r.h <= 0) continue;
      const size = Math.max(2, r.h * 0.8);
      try {
        page.drawText(winAnsiSafe(line.text), { x: r.x, y: p.h - r.y - size, size, font });
      } catch {
        /* skip a line that still fails to encode */
      }
    }
  }

  return doc.save();
}
