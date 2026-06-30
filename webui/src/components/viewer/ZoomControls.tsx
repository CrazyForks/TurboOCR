import { Minus, Plus, Maximize2 } from "lucide-react";

export function ZoomControls({
  zoom,
  setZoom,
  onFit,
}: {
  zoom: number;
  setZoom: (z: number) => void;
  onFit: () => void;
}) {
  const step = (d: number) =>
    setZoom(Math.min(8, Math.max(0.1, +(zoom + d).toFixed(2))));
  const btn =
    "flex h-8 w-8 items-center justify-center text-neutral-600 hover:bg-neutral-100";
  return (
    <div className="absolute bottom-4 right-4 flex items-center overflow-hidden rounded-lg border border-neutral-200 bg-white/90 shadow-sm backdrop-blur">
      <button className={btn} onClick={() => step(-0.1)} title="Zoom out">
        <Minus className="h-4 w-4" />
      </button>
      <span className="w-12 text-center text-xs tabular-nums text-neutral-600">
        {Math.round(zoom * 100)}%
      </span>
      <button className={btn} onClick={() => step(0.1)} title="Zoom in">
        <Plus className="h-4 w-4" />
      </button>
      <button
        className={btn + " border-l border-neutral-200"}
        onClick={onFit}
        title="Fit to width"
      >
        <Maximize2 className="h-4 w-4" />
      </button>
    </div>
  );
}
