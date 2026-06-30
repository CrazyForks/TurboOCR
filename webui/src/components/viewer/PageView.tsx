import { useEffect, useMemo, useRef } from "react";
import type { DocPage, PageResult } from "../../lib/doc/types";
import type { Hovered, Layers } from "./types";
import { OverlayBoxes } from "./OverlayBoxes";
import { TextSelectLayer } from "./TextSelectLayer";
import { orderLines } from "../../lib/textlayer/readingOrder";
import { Spinner } from "../ui";

// One page. The slot sizes itself from cheap metadata (shows instantly); the
// raster streams in. The heavy overlay + text layers mount only when `active`
// (near the viewport) so big multi-page docs stay fast. Scaling uses an
// ancestor transform (NOT CSS zoom, which breaks selection).
export function PageView({
  page,
  raster,
  result,
  index,
  zoom,
  layers,
  selectText,
  hovered,
  multi,
  active,
  registerEl,
}: {
  page: DocPage;
  raster?: { url: string };
  result: PageResult | undefined;
  index: number;
  zoom: number;
  layers: Layers;
  selectText: boolean;
  hovered: Hovered;
  multi: boolean;
  active: boolean;
  registerEl: (id: number, el: HTMLElement | null) => void;
}) {
  const outerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    registerEl(page.id, outerRef.current);
    return () => registerEl(page.id, null);
  }, [page.id, registerEl]);

  const response = result?.response ?? null;
  // Re-group lines into reading order so within-column selection is contiguous.
  const lines = useMemo(() => orderLines(response), [response]);
  return (
    <div
      ref={outerRef}
      data-page-id={page.id}
      className="relative rounded-paper bg-paper shadow-paper ring-1 ring-[var(--paper-line)]"
      style={{ width: Math.round(page.w * zoom), height: Math.round(page.h * zoom) }}
    >
      <div
        className="absolute left-0 top-0"
        style={{ width: page.w, height: page.h, transform: `scale(${zoom})`, transformOrigin: "0 0" }}
      >
        {raster ? (
          <>
            <img
              src={raster.url}
              width={page.w}
              height={page.h}
              draggable={false}
              className="block select-none rounded-paper bg-paper"
              alt={`page ${index + 1}`}
            />
            {active && (
              <>
                <OverlayBoxes w={page.w} h={page.h} data={response} layers={layers} hovered={hovered} />
                <TextSelectLayer lines={lines} w={page.w} h={page.h} enabled={selectText} />
              </>
            )}
          </>
        ) : (
          <div className="h-full w-full animate-pulse bg-neutral-100" style={{ width: page.w, height: page.h }} />
        )}
      </div>

      {multi && (
        <div className="absolute left-2 top-2 rounded-thumb bg-white/90 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-indigo-600 ring-1 ring-neutral-200">
          {index + 1}
        </div>
      )}
      {!raster && (
        <div className="absolute right-2 top-2" title="rendering">
          <Spinner />
        </div>
      )}
      {raster && result?.status === "running" && (
        <div className="absolute right-2 top-2" title="OCR">
          <Spinner />
        </div>
      )}
      {result?.status === "error" && (
        <div className="absolute inset-x-0 top-0 bg-low-bg px-2 py-1 text-xs text-low">
          {result.error}
        </div>
      )}
    </div>
  );
}
