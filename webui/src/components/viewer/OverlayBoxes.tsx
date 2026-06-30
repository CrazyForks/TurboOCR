import type { OcrResponse, Poly } from "../../lib/types";
import { colorForClass, LINE_COLOR } from "../../lib/colors";
import type { Hovered, Layers } from "./types";

function points(box: Poly): string {
  return box.map((p) => `${p[0]},${p[1]}`).join(" ");
}

// A small class tag at a region's top-left (e.g. "header", "paragraph title",
// "table"), sitting just above the box when there's room.
function RegionLabel({ box, label, color }: { box: Poly; label: string; color: string }) {
  const text = label.replace(/_/g, " ");
  const x = box[0][0];
  const y = box[0][1];
  const fs = 13;
  const h = fs + 5;
  const wTag = text.length * fs * 0.58 + 8;
  const ty = y - h >= 0 ? y - h : y;
  return (
    <g>
      <rect x={x} y={ty} width={wTag} height={h} rx={2} fill={color} opacity={0.92} />
      <text
        x={x + 4}
        y={ty + fs}
        fontSize={fs}
        fontFamily="ui-sans-serif, system-ui, sans-serif"
        fill="#fff"
      >
        {text}
      </text>
    </g>
  );
}

function Quad({
  box,
  stroke,
  fill,
  active,
  dashed,
}: {
  box: Poly;
  stroke: string;
  fill: string;
  active?: boolean;
  dashed?: boolean;
}) {
  return (
    <polygon
      points={points(box)}
      fill={active ? "rgba(79,70,229,0.16)" : fill}
      stroke={active ? "#4f46e5" : stroke}
      strokeWidth={active ? 2.5 : 1.25}
      strokeDasharray={dashed ? "6 4" : undefined}
      vectorEffect="non-scaling-stroke"
    />
  );
}

// Pure overlay of OCR boxes; never intercepts pointer events so the text layer
// underneath stays selectable.
export function OverlayBoxes({
  w,
  h,
  data,
  layers,
  hovered,
}: {
  w: number;
  h: number;
  data: OcrResponse | null;
  layers: Layers;
  hovered: Hovered;
}) {
  const hoveredLayoutId =
    hovered?.kind === "line" && data?.results[hovered.idx]?.layout_id != null
      ? data.results[hovered.idx].layout_id
      : undefined;

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      width={w}
      height={h}
      className="pointer-events-none absolute inset-0"
    >
      {layers.layout &&
        data?.layout?.map((r, i) => {
          const c = colorForClass(r.class_id);
          return (
            <g key={`r${i}`}>
              <Quad
                box={r.bounding_box}
                stroke={c.stroke}
                fill={c.fill}
                dashed={r.class_id < 0}
                active={
                  (hovered?.kind === "region" && hovered.idx === i) ||
                  (hoveredLayoutId != null && r.id === hoveredLayoutId)
                }
              />
              <RegionLabel box={r.bounding_box} label={r.class} color={c.stroke} />
            </g>
          );
        })}
      {layers.tables &&
        data?.tables?.map((t, i) => {
          const c = colorForClass(21);
          return (
            <Quad
              key={`t${i}`}
              box={t.bounding_box}
              stroke={c.stroke}
              fill={c.fill}
              active={hovered?.kind === "table" && hovered.idx === i}
            />
          );
        })}
      {layers.formulas &&
        data?.formulas?.map((f, i) => {
          const c = colorForClass(5);
          return (
            <Quad
              key={`f${i}`}
              box={f.bounding_box}
              stroke={c.stroke}
              fill={c.fill}
              active={hovered?.kind === "formula" && hovered.idx === i}
            />
          );
        })}
      {layers.lines &&
        data?.results?.map((l, i) => (
          <Quad
            key={`l${i}`}
            box={l.bounding_box}
            stroke={LINE_COLOR.stroke}
            fill={LINE_COLOR.fill}
            active={hovered?.kind === "line" && hovered.idx === i}
          />
        ))}
    </svg>
  );
}
