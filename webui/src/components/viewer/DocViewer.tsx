import { useCallback, useEffect, useRef, useState } from "react";
import type { DocPage, RasterMap, ResultMap } from "../../lib/doc/types";
import type { Hovered, Layers } from "./types";
import { PageView } from "./PageView";
import { ZoomControls } from "./ZoomControls";

export function DocViewer({
  pages,
  rasters,
  results,
  layers,
  selectText,
  hovered,
}: {
  pages: DocPage[];
  rasters: RasterMap;
  results: ResultMap;
  layers: Layers;
  selectText: boolean;
  hovered: Hovered;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const maxW = Math.max(1, ...pages.map((p) => p.w));

  // Virtualize the heavy per-page layers (overlay SVG + text spans): only pages
  // near the viewport mount them, so selection/scroll stay fast on big PDFs
  // (thousands of nodes otherwise). The raster <img> always shows.
  const [active, setActive] = useState<Set<number>>(new Set());
  const observerRef = useRef<IntersectionObserver | null>(null);
  const elements = useRef<Map<number, HTMLElement>>(new Map());

  useEffect(() => {
    const root = scrollRef.current;
    if (!root) return;
    const io = new IntersectionObserver(
      (entries) =>
        setActive((prev) => {
          const next = new Set(prev);
          for (const e of entries) {
            const id = Number((e.target as HTMLElement).dataset.pageId);
            if (e.isIntersecting) next.add(id);
            else next.delete(id);
          }
          return next;
        }),
      { root, rootMargin: "1200px 0px" },
    );
    observerRef.current = io;
    elements.current.forEach((el) => io.observe(el));
    return () => {
      io.disconnect();
      observerRef.current = null;
    };
  }, [pages.length]);

  const registerEl = useCallback((id: number, el: HTMLElement | null) => {
    const map = elements.current;
    const prev = map.get(id);
    if (prev) observerRef.current?.unobserve(prev);
    if (el) {
      map.set(id, el);
      observerRef.current?.observe(el);
    } else {
      map.delete(id);
    }
  }, []);

  const fit = () => {
    const c = scrollRef.current;
    if (!c) return;
    setZoom(Math.min(1, +((c.clientWidth - 48) / maxW).toFixed(3)));
  };

  useEffect(() => {
    fit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [maxW, pages.length]);

  const multi = pages.length > 1;

  return (
    <div className="relative h-full">
      <div ref={scrollRef} className="h-full select-none overflow-auto bg-neutral-100 p-6">
        <div className="mx-auto flex w-fit flex-col items-center gap-6">
          {pages.map((p, i) => (
            <PageView
              key={p.id}
              page={p}
              raster={rasters[p.id]}
              result={results[p.id]}
              index={i}
              zoom={zoom}
              layers={layers}
              selectText={selectText}
              hovered={hovered?.page === i ? hovered : null}
              multi={multi}
              active={active.has(p.id)}
              registerEl={registerEl}
            />
          ))}
        </div>
      </div>
      <ZoomControls zoom={zoom} setZoom={setZoom} onFit={fit} />
    </div>
  );
}
