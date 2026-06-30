import { useEffect, useRef } from "react";
import type { TextLine } from "../../lib/types";
import { buildTextLayer, type LineBox } from "../../lib/textlayer/buildTextLayer";
import { createSelectionController } from "../../lib/textlayer/selectionController";

// React owns only the empty container; spans are built imperatively. Selection
// is driven geometrically by a controller (see selectionController.ts) — no
// reliance on native caret-from-point over empty space, which was the source of
// the snap-to-top / select-all bug.
export function TextSelectLayer({
  lines,
  w,
  h,
  enabled,
}: {
  lines: TextLine[];
  w: number;
  h: number;
  enabled: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const div = ref.current;
    if (!div) return;
    if (!enabled) {
      div.replaceChildren();
      return;
    }
    let cancelled = false;
    let boxes: LineBox[] = buildTextLayer(div, lines);
    const dispose = createSelectionController(div, () => boxes);
    // Web fonts can resolve after first paint; rebuild once ready so the
    // measured width (=> scaleX) matches the finally-rendered face.
    const fonts = (document as Document & { fonts?: FontFaceSet }).fonts;
    fonts?.ready?.then(() => {
      if (!cancelled) boxes = buildTextLayer(div, lines);
    });
    return () => {
      cancelled = true;
      dispose();
      div.replaceChildren();
    };
  }, [lines, w, h, enabled]);

  return (
    <div
      ref={ref}
      className="textLayer"
      style={{
        width: w,
        height: h,
        display: enabled ? undefined : "none",
      }}
    />
  );
}
