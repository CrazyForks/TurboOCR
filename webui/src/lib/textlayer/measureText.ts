// Shared offscreen 2D context for intrinsic text-width measurement.
// ctx.measureText returns the natural advance width at an exact font+size,
// synchronously, with NO reflow and independent of any CSS transform later
// applied to the span. That intrinsic width is the denominator for per-span
// scaleX. Measured in IMAGE px so the ratio is scale-invariant — zoom is the
// ancestor transform's job, never here.

// MUST stay byte-identical to the span font-family applied in index.css, or the
// measured width and the rendered width diverge and scaleX drifts.
export const TEXT_FONT_FAMILY =
  '"Inter Variable", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

let ctx: CanvasRenderingContext2D | null = null;
let cachedFont = "";

function context(): CanvasRenderingContext2D | null {
  if (ctx) return ctx;
  if (typeof document === "undefined") return null;
  ctx = document.createElement("canvas").getContext("2d");
  return ctx;
}

export function measureWidth(text: string, fontPx: number): number {
  if (!text) return 0;
  const c = context();
  if (!c) return 0;
  const font = `${fontPx}px ${TEXT_FONT_FAMILY}`;
  if (font !== cachedFont) {
    c.font = font;
    cachedFont = font;
  }
  return c.measureText(text).width;
}
