// Mirrors the TurboOCR HTTP JSON contract (include/turbo_ocr/common/serialization.h).
// Coordinates are 4-point polygons [tl, tr, br, bl] in INTEGER PIXELS of the
// decoded image (top-left origin), never normalized. Optional keys are OMITTED
// when empty — treat a missing key as an empty array.

export type Point = [number, number];
export type Poly = [Point, Point, Point, Point];

export interface TextLine {
  id?: number;
  text: string;
  confidence: number;
  bounding_box: Poly;
  source?: string;
  layout_id?: number;
}

export interface LayoutRegion {
  id?: number;
  class: string;
  class_id: number;
  confidence: number;
  bounding_box: Poly;
}

export interface TableItem {
  layout_id: number;
  html: string;
  confidence: number;
  bounding_box: Poly;
}

export interface FormulaItem {
  layout_id: number;
  latex: string;
  confidence: number;
  bounding_box: Poly;
}

export interface Block {
  id?: number;
  layout_id?: number;
  class?: string;
  bounding_box: Poly;
  content?: string;
  order_index?: number;
}

export interface OcrResponse {
  results: TextLine[];
  layout?: LayoutRegion[];
  reading_order?: number[];
  blocks?: Block[];
  tables?: TableItem[];
  formulas?: FormulaItem[];
  text_degraded?: boolean;
  text_warning?: string;
  table_degraded?: boolean;
  table_warning?: string;
  formula_degraded?: boolean;
  formula_warning?: string;
}

export interface Capabilities {
  build: "gpu" | "cpu" | string;
  features: {
    layout: boolean;
    tables: boolean;
    formulas: boolean;
    autorotate: boolean;
    [k: string]: unknown;
  };
  limits?: {
    max_body_mb?: number;
    max_image_dim?: number;
    max_batch_images?: number;
  };
  endpoints?: string[];
  [k: string]: unknown;
}

export interface RunOptions {
  layout: boolean;
  tables: boolean;
  formulas: boolean;
}
