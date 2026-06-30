export type Hovered =
  | { page: number; kind: "line" | "region" | "table" | "formula"; idx: number }
  | null;

export interface Layers {
  lines: boolean;
  layout: boolean;
  tables: boolean;
  formulas: boolean;
}
