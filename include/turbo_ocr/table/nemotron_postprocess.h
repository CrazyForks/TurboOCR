#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace turbo_ocr::table {

// One detected box from the YOLOX head, scaled back to original-image pixels.
// `bbox` is xyxy. `class_id` is the upstream label (1 = cell, 2 = row,
// 3 = column; 0 = border and 4 = header are dropped by `decode_nemotron`).
struct NemotronBox {
  std::array<float, 4> bbox;   // x1, y1, x2, y2 in original image pixels
  float                score = 0.0f;
  int                  class_id = -1;
};

// Decoded triplet for downstream consumers. Each vector is already
// per-class NMS'd at iou=0.25 and filtered at score>=0.05.
struct NemotronDecode {
  std::vector<NemotronBox> cells;    // class_id == 1
  std::vector<NemotronBox> rows;     // class_id == 2
  std::vector<NemotronBox> columns;  // class_id == 3
  std::vector<NemotronBox> headers;  // class_id == 4 — column-header rows
};

// One HTML cell ready for OCR text fill.
struct NemotronCell {
  int row_idx = -1;            // 0-based row index in the assembled table
  int col_idx = -1;            // 0-based starting column index
  int colspan = 1;
  int rowspan = 1;
  std::array<float, 4> bbox;   // xyxy in original image pixels, fed to cell_matcher
};

struct NemotronAssembly {
  std::vector<NemotronCell> cells;
  int num_rows = 0;
  int num_cols = 0;
  // Set of row indices that fall inside a detected header bbox. Used by
  // `nemotron_to_html` to wrap header rows in `<thead>` and use `<th>`
  // cells, matching the OmniDocBench GT structure on many tables.
  std::vector<int> header_rows;
};

inline constexpr int   kNemotronCanvas    = 1024;
inline constexpr float kNemotronConfThresh = 0.05f;     // floor for cells
inline constexpr float kNemotronRowColScoreThresh = 0.50f; // row/col are dense
inline constexpr float kNemotronHeaderScoreThresh = 0.45f; // headers are rare per table
inline constexpr float kNemotronCellNmsIoU = 0.25f;
inline constexpr float kNemotronRowColNmsIoU = 0.55f;   // row/col overlap heavily
inline constexpr int   kNemotronMaxRows    = 60;
inline constexpr int   kNemotronMaxCols    = 30;
inline constexpr int   kNemotronNumQueries = 21504;
inline constexpr int   kNemotronStride     = 10;       // cxcywh + obj + 5 cls
// Kept for ABI compat with any pre-existing call sites referencing the
// legacy single-threshold name.
inline constexpr float kNemotronNmsIoU     = kNemotronCellNmsIoU;

// Decode raw model output. `preds` is row-major `[num_queries, kNemotronStride]`
// for one image; `ori_w`/`ori_h` are the original image dimensions before
// letterbox. Output is in original-image pixels.
NemotronDecode decode_nemotron(const float* preds,
                                std::size_t num_queries,
                                int ori_w,
                                int ori_h);

// Build the <table>...</table> string + per-cell bboxes from a decoded set.
// Each grid cell takes its bbox from the Cell-class detection that best
// matches it (IoU); when no cell box covers a grid slot the bbox is the
// intersection of the row band and the column band.
NemotronAssembly assemble_nemotron(const NemotronDecode& decoded);

// Compose the final HTML using the assembled cells. OCR text per cell is
// supplied in the same order as `assembly.cells`; empty strings are valid.
std::string nemotron_to_html(const NemotronAssembly& assembly,
                              const std::vector<std::string>& cell_texts);

} // namespace turbo_ocr::table
