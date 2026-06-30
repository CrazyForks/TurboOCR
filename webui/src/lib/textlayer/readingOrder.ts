import type { OcrResponse, TextLine } from "../types";

// Order the selectable text lines by PURE GEOMETRY: top to bottom, then left to
// right within a row. Selection follows this order, so a drag is always
// spatially monotonic — it can only ever cover lines BETWEEN the start point and
// the cursor, never anything above the start or below the cursor.
//
// We deliberately do NOT use the server `reading_order` here: it orders for
// reading (e.g. a whole right column before the left one, jumping back up the
// page), which makes a drag grab far-away, scattered lines. Position is the only
// thing that matches what the cursor is doing.
export function orderLines(resp: OcrResponse | null | undefined): TextLine[] {
  const results = resp?.results ?? [];
  if (results.length === 0) return results;

  const cy = (l: TextLine) => (l.bounding_box[0][1] + l.bounding_box[2][1]) / 2;
  const left = (l: TextLine) => l.bounding_box[0][0];
  const height = (l: TextLine) => Math.abs(l.bounding_box[2][1] - l.bounding_box[0][1]);

  // Row band ≈ 0.7 of the median line height, so lines on the same visual row
  // group together and order left→right; rows order top→bottom. Integer bands
  // keep the comparator a strict, transitive total order.
  const hs = results.map(height).sort((a, b) => a - b);
  const band = Math.max(6, (hs[hs.length >> 1] || 12) * 0.7);

  return [...results].sort((a, b) => {
    const ra = Math.round(cy(a) / band);
    const rb = Math.round(cy(b) / band);
    return ra - rb || left(a) - left(b);
  });
}
