#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace turbo_ocr::table {

// One OCR text-line on the original page.
struct OcrLine {
    // Axis-aligned bbox in original-image pixels: [x1, y1, x2, y2].
    std::array<float, 4> bbox;
    std::string text;
};

// One cell quad with the OCR indices that fall inside it.
struct MatchedCell {
    // Axis-aligned bbox derived from the quad (min/max over x_even/y_odd of
    // the 8 coords) — matches PaddleX's quad→bbox conversion in
    // match_table_and_ocr.
    std::array<float, 4> bbox;
    // Indices into the input OCR slice. Empty for cells with no match.
    std::vector<std::size_t> ocr_indices;
};

// PaddleX `match_table_and_ocr` literal — `compute_inter > 0.7`.
inline constexpr float MATCH_INTER_THRESHOLD = 0.7f;

// PaddleX-equivalent intersect ratio: inter_area / box_b_area.
// +1 byte-equal preservation: PaddleX uses `(x_right - x_left + 1)` style
// width/height inclusive arithmetic.
float compute_inter(const std::array<float, 4>& a, const std::array<float, 4>& b);

// Quad → axis-aligned bbox over the 8 coords laid out as
// [x1, y1, x2, y2, x3, y3, x4, y4].
std::array<float, 4> quad_to_bbox(const std::array<int, 8>& quad);

// Match OCR lines to cell quads. For each cell returns the OCR indices whose
// box satisfies compute_inter(cell, ocr) > MATCH_INTER_THRESHOLD.
std::vector<MatchedCell> match_cells_to_ocr(
    const std::vector<std::array<int, 8>>& cells,
    const std::vector<OcrLine>& ocr);

} // namespace turbo_ocr::table
