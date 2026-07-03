import type { LineBox } from "./buildTextLayer";

// SOTA text selection (Preview/PDF.js style) driven entirely by geometry.
//
// Why not native selection? The browser's caretRangeFromPoint resolves an
// empty-space point to the nearest text node in DOM order, which on a sparse,
// multi-column OCR layer collapses to the document start — that is the old
// "drag into a gap and it snaps to the top / selects everything above" bug.
//
// Instead we keep the two things the user described: the anchor (where the drag
// started) and the live pointer. On every move we hit-test the pointer to the
// NEAREST line box (never the top), pick the caret WITHIN that line, and drive
// the selection with setBaseAndExtent(anchor -> focus). Empty space therefore
// extends toward the nearest line in the drag direction, exactly like Preview.
//
// Spans are laid out in reading order (== DOM order), so the range between any
// two carets is the contiguous reading-order run between them.

interface Caret {
  node: Node;
  offset: number;
  line: number; // index into getBoxes() (reading order)
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// Browser hit-test only — the reading-order `line` index is attached by the
// caller (caretInLine), so this returns a Caret minus `line`.
function caretFromPoint(x: number, y: number): Omit<Caret, "line"> | null {
  const d = document as Document & {
    caretRangeFromPoint?: (x: number, y: number) => Range | null;
    caretPositionFromPoint?: (x: number, y: number) => { offsetNode: Node; offset: number } | null;
  };
  if (d.caretRangeFromPoint) {
    const r = d.caretRangeFromPoint(x, y);
    return r ? { node: r.startContainer, offset: r.startOffset } : null;
  }
  if (d.caretPositionFromPoint) {
    const p = d.caretPositionFromPoint(x, y);
    return p ? { node: p.offsetNode, offset: p.offset } : null;
  }
  return null;
}

// Walk word boundaries within a single line's text (for double-click).
function wordRange(text: string, offset: number): [number, number] {
  const isWord = (c: string) => /[\p{L}\p{N}_]/u.test(c);
  if (!text) return [0, 0];
  let start = clamp(offset, 0, text.length);
  let end = start;
  if (start < text.length && isWord(text[start])) {
    while (start > 0 && isWord(text[start - 1])) start--;
    while (end < text.length && isWord(text[end])) end++;
  } else if (start > 0 && isWord(text[start - 1])) {
    while (start > 0 && isWord(text[start - 1])) start--;
    end = offset;
  } else {
    return [start, start];
  }
  return [start, end];
}

export function createSelectionController(
  container: HTMLElement,
  getBoxes: () => LineBox[],
): () => void {
  let dragging = false;
  let anchor: Caret | null = null;
  let lastX = 0;
  let lastY = 0;
  let raf = 0;
  let scrollEl: HTMLElement | null = null;

  const findScrollParent = (): HTMLElement | null => {
    let n = container.parentElement;
    while (n) {
      const oy = getComputedStyle(n).overflowY;
      if ((oy === "auto" || oy === "scroll") && n.scrollHeight > n.clientHeight) return n;
      n = n.parentElement;
    }
    return null;
  };

  // client px -> natural px of the page (container is scaled by an ancestor
  // transform: scale(zoom), so divide by the measured scale).
  const toNatural = (clientX: number, clientY: number) => {
    const r = container.getBoundingClientRect();
    const scale = r.width / (container.offsetWidth || 1) || 1;
    return { x: (clientX - r.left) / scale, y: (clientY - r.top) / scale };
  };

  // Nearest line index: vertical proximity dominates (pick the row), then
  // horizontal (pick the column). A point inside a box scores 0 on that axis.
  const nearestLineIndex = (nx: number, ny: number, boxes: LineBox[]): number => {
    let best = -1;
    let bestScore = Infinity;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      const dy = ny < b.top ? b.top - ny : ny > b.top + b.height ? ny - (b.top + b.height) : 0;
      const dx = nx < b.left ? b.left - nx : nx > b.left + b.width ? nx - (b.left + b.width) : 0;
      const score = dy * 64 + dx;
      if (score < bestScore) {
        bestScore = score;
        best = i;
      }
    }
    return best;
  };

  // Caret within a chosen line: clamp the pointer INTO the span's client rect
  // and ask the browser for the exact caret there. Because the point is now
  // guaranteed over real text, this never snaps elsewhere; it just gives precise
  // per-character placement. Falls back to a proportional estimate.
  const caretInLine = (b: LineBox, line: number, clientX: number, clientY: number): Caret => {
    const r = b.span.getBoundingClientRect();
    const x = clamp(clientX, r.left + 0.5, r.right - 0.5);
    const y = clamp(clientY, r.top + 0.5, r.bottom - 0.5);
    const c = caretFromPoint(x, y);
    if (c && b.span.contains(c.node)) return { ...c, line };
    const node = b.span.firstChild ?? b.span;
    const len = node.textContent?.length ?? 0;
    const frac = r.width > 0 ? (x - r.left) / r.width : 0;
    return { node, offset: clamp(Math.round(frac * len), 0, len), line };
  };

  const resolve = (clientX: number, clientY: number): Caret | null => {
    const { x, y } = toNatural(clientX, clientY);
    const boxes = getBoxes();
    const idx = nearestLineIndex(x, y, boxes);
    return idx < 0 ? null : caretInLine(boxes[idx], idx, clientX, clientY);
  };

  // Drive the native selection from the anchor (where the drag started) to the
  // exact caret under the cursor. Standard flow selection (like Preview): it
  // ends precisely at the cursor and never extends past it — nothing below or to
  // the right of the cursor is ever selected.
  const apply = (focus: Caret | null) => {
    if (!anchor || !focus) return;
    const sel = window.getSelection();
    if (!sel) return;
    try {
      sel.setBaseAndExtent(anchor.node, anchor.offset, focus.node, focus.offset);
    } catch {
      /* node detached mid-rebuild */
    }
  };

  const autoScroll = () => {
    raf = 0;
    if (!dragging) return;
    if (scrollEl) {
      const r = scrollEl.getBoundingClientRect();
      const EDGE = 48;
      let dy = 0;
      if (lastY < r.top + EDGE) dy = -(EDGE - (lastY - r.top));
      else if (lastY > r.bottom - EDGE) dy = EDGE - (r.bottom - lastY);
      if (dy) {
        scrollEl.scrollTop += dy * 0.5;
        apply(resolve(lastX, lastY));
      }
    }
    raf = requestAnimationFrame(autoScroll);
  };

  const onMove = (e: MouseEvent) => {
    if (!dragging) return;
    lastX = e.clientX;
    lastY = e.clientY;
    apply(resolve(e.clientX, e.clientY));
    e.preventDefault();
  };

  const onUp = () => {
    if (!dragging) return;
    dragging = false;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    document.removeEventListener("mousemove", onMove, true);
    document.removeEventListener("mouseup", onUp, true);
  };

  const onDown = (e: MouseEvent) => {
    if (e.button !== 0) return;
    const start = resolve(e.clientX, e.clientY);
    if (!start) return;
    // We own selection; stop the browser starting its own (mis-anchoring) drag.
    e.preventDefault();

    // Double / triple click: select word / line without a drag.
    if (e.detail >= 2) {
      const { x, y } = toNatural(e.clientX, e.clientY);
      const boxes = getBoxes();
      const b = boxes[nearestLineIndex(x, y, boxes)];
      const node = b?.span.firstChild;
      if (b && node) {
        const sel = window.getSelection();
        if (e.detail === 2) {
          const [s, en] = wordRange(b.text, start.offset);
          sel?.setBaseAndExtent(node, s, node, en);
        } else {
          sel?.setBaseAndExtent(node, 0, node, b.text.length);
        }
      }
      anchor = start;
      return;
    }

    anchor = start;
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    scrollEl = findScrollParent();
    window.getSelection()?.removeAllRanges();
    apply(start); // collapsed at anchor (also clears any prior selection)
    document.addEventListener("mousemove", onMove, true);
    document.addEventListener("mouseup", onUp, true);
    raf = requestAnimationFrame(autoScroll);
  };

  container.addEventListener("mousedown", onDown);

  return () => {
    container.removeEventListener("mousedown", onDown);
    onUp();
  };
}
