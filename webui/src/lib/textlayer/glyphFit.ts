import type { Poly } from "../types";
import { measureWidth } from "./measureText";

// One transparent span per OCR line, stretched (scaleX) to the FULL line-box
// width so the whole line — including inter-word space and the empty right
// margin — is a selectable target (no horizontal dead zones). buildTextLayer
// also gives each span enough height to fill the gap to the next line, so there
// are no vertical dead zones either. That full tiling is what stops empty-space
// clicks from anchoring to the top of the document.
export interface SpanGeom {
  text: string;
  left: number;
  top: number;
  width: number; // boxW: rendered + hit width in natural px (== OCR box width)
  fontSize: number; // == box height (line-height:1 => content box == OCR box)
  scaleX: number; // boxW / measuredWidth -> rendered + hit width == box width
  angle: number;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// Polygon order is [tl, tr, br, bl] in integer px (lib/types.ts Poly).
export function lineToSpanGeom(box: Poly, text: string): SpanGeom {
  const [tl, tr, , bl] = box;
  const wdx = tr[0] - tl[0];
  const wdy = tr[1] - tl[1];
  const boxW = Math.hypot(wdx, wdy);
  const boxH = Math.hypot(bl[0] - tl[0], bl[1] - tl[1]);
  const fontSize = Math.max(1, boxH);
  const measured = measureWidth(text, fontSize);
  const scaleX = measured > 0 ? clamp(boxW / measured, 0.02, 50) : 1;
  const angle = Math.abs(wdy) > 0.5 ? (Math.atan2(wdy, wdx) * 180) / Math.PI : 0;
  return { text, left: tl[0], top: tl[1], width: boxW, fontSize, scaleX, angle };
}
