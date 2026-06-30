import type { OcrResponse } from "../types";

// A page is just its dimensions + id; the rasterized image arrives separately
// (streamed) so the viewer can show page slots before pixels are ready.
export interface DocPage {
  id: number;
  w: number;
  h: number;
}

export interface DocModel {
  name: string;
  kind: "image" | "pdf";
  pages: DocPage[];
}

export type PageStatus = "pending" | "running" | "done" | "error";

export interface PageResult {
  status: PageStatus;
  response: OcrResponse | null;
  elapsedMs?: number;
  error?: string;
}

export type ResultMap = Record<number, PageResult>;
export type RasterMap = Record<number, { url: string; bytes: ArrayBuffer }>;
