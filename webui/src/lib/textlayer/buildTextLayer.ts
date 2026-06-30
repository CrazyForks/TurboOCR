import type { TextLine } from "../types";
import { lineToSpanGeom } from "./glyphFit";
import { TEXT_FONT_FAMILY } from "./measureText";

// A built line: its span element plus the axis-aligned box (natural px) the
// span visually covers. The selection controller hit-tests against these boxes
// so a drag into empty space maps to the NEAREST line geometrically — never to
// the document start (the root cause of the old "snaps to top / selects all").
export interface LineBox {
  span: HTMLSpanElement;
  text: string;
  left: number;
  top: number;
  width: number;
  height: number;
}

// One transparent span per line, stretched to the full line-box width (scaleX)
// and given enough height to reach the next line. Selection itself is driven
// geometrically by the controller (setBaseAndExtent), not by native caret
// hit-testing over these spans. Returns the line boxes in reading order.
export function buildTextLayer(container: HTMLElement, lines: TextLine[]): LineBox[] {
  container.replaceChildren();
  const geoms = lines
    .filter((l) => l.text)
    .map((l) => lineToSpanGeom(l.bounding_box, l.text));

  const boxes: LineBox[] = [];
  const frag = document.createDocumentFragment();
  for (let i = 0; i < geoms.length; i++) {
    const g = geoms[i];
    // Fill the vertical gap down to the next line (when it sits directly below,
    // not a column break) so inter-line empty space is still selectable.
    let height = g.fontSize;
    const next = geoms[i + 1];
    if (next && g.angle === 0 && next.angle === 0) {
      const gap = next.top - g.top;
      if (gap > g.fontSize && gap < g.fontSize * 2.4) height = gap;
    }

    const span = document.createElement("span");
    span.setAttribute("role", "presentation");
    span.textContent = g.text;
    const s = span.style;
    s.left = `${g.left}px`;
    s.top = `${g.top}px`;
    s.height = `${height}px`;
    s.fontFamily = TEXT_FONT_FAMILY;
    s.fontSize = `${g.fontSize}px`;
    s.setProperty("--scale-x", String(g.scaleX));
    if (g.angle) s.setProperty("--rotate", `${g.angle}deg`);
    frag.append(span);
    boxes.push({ span, text: g.text, left: g.left, top: g.top, width: g.width, height });

    const br = document.createElement("br");
    br.setAttribute("role", "presentation");
    frag.append(br);
  }
  container.append(frag);
  return boxes;
}
