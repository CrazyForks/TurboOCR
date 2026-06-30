// The 25 layout classes (include/turbo_ocr/layout/layout_types.h) grouped into
// a few legible color families. class_id -1 is the synthesized
// "SupplementaryRegion" catch-all.

export interface Hue {
  family: string;
  stroke: string;
  fill: string;
}

const FAMILIES: Record<string, { hue: number; sat: number; classes: number[] }> = {
  Title: { hue: 38, sat: 92, classes: [6, 7, 17] }, // doc_title, figure_title, paragraph_title
  Text: { hue: 205, sat: 85, classes: [0, 2, 4, 19, 22, 23] }, // abstract, aside_text, content, reference_content, text, vertical_text
  Table: { hue: 270, sat: 80, classes: [21] },
  Figure: { hue: 150, sat: 70, classes: [3, 14] }, // chart, image
  Formula: { hue: 330, sat: 85, classes: [5, 11, 15] }, // display_formula, formula_number, inline_formula
  Meta: { hue: 220, sat: 12, classes: [1, 8, 9, 10, 12, 13, 16, 18, 20, 24] }, // headers/footers/refs/etc.
};

const CLASS_TO_FAMILY = new Map<number, string>();
for (const [name, def] of Object.entries(FAMILIES)) {
  for (const c of def.classes) CLASS_TO_FAMILY.set(c, name);
}

function hue(h: number, s: number): Omit<Hue, "family"> {
  // Tuned for a light/white canvas: saturated mid-dark stroke, faint fill.
  return {
    stroke: `hsl(${h} ${s}% 45%)`,
    fill: `hsl(${h} ${s}% 55% / 0.12)`,
  };
}

export function colorForClass(classId: number): Hue {
  const fam = CLASS_TO_FAMILY.get(classId);
  if (!fam) {
    // Unknown class or SupplementaryRegion (-1): neutral gray.
    return { family: "Other", ...hue(220, 6) };
  }
  const def = FAMILIES[fam];
  return { family: fam, ...hue(def.hue, def.sat) };
}

// Distinct, thin color for raw OCR text-line quads (not layout regions).
export const LINE_COLOR: Hue = { family: "Text line", ...hue(160, 65) };

export const LEGEND: { label: string; stroke: string }[] = [
  ...Object.entries(FAMILIES).map(([label, def]) => ({
    label,
    stroke: hue(def.hue, def.sat).stroke,
  })),
  { label: "Text line", stroke: LINE_COLOR.stroke },
];
