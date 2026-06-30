import { buildSearchablePdf } from "../pdf/searchablePdf";
import { downloadBlob, baseName } from "./download";
import type { DocPage, RasterMap, ResultMap } from "../doc/types";

export async function exportSearchablePdf(
  name: string,
  pages: DocPage[],
  rasters: RasterMap,
  results: ResultMap,
): Promise<void> {
  const exportPages = pages
    .filter((p) => rasters[p.id])
    .map((p) => ({
      w: p.w,
      h: p.h,
      bytes: rasters[p.id].bytes,
      response: results[p.id]?.response ?? null,
    }));
  const bytes = await buildSearchablePdf(exportPages);
  downloadBlob(
    `${baseName(name)}.ocr.pdf`,
    new Blob([bytes as BlobPart], { type: "application/pdf" }),
  );
}

export function exportText(name: string, pages: DocPage[], results: ResultMap): void {
  const text = pages
    .map((p) => (results[p.id]?.response?.results ?? []).map((r) => r.text).join("\n"))
    .join("\n\n");
  downloadBlob(`${baseName(name)}.txt`, new Blob([text], { type: "text/plain" }));
}

export function exportJson(name: string, pages: DocPage[], results: ResultMap): void {
  const data = pages.map((p, i) => ({ page: i, ...(results[p.id]?.response ?? {}) }));
  downloadBlob(
    `${baseName(name)}.json`,
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
}
