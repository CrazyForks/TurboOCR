import type { Poly } from "./types";

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

// Axis-aligned bounding box of a 4-point polygon (handles slanted quads).
export function aabb(box: Poly): Rect {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of box) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}
