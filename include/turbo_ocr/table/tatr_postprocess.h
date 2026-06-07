#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "turbo_ocr/table/nemotron_postprocess.h"

namespace turbo_ocr::table {

// TATR head: 7-class logits + 4-vec normalized cxcywh boxes per query.
// Class ids per Microsoft Table Transformer:
//   0 table, 1 column, 2 row, 3 column header,
//   4 projected row header, 5 spanning cell, 6 no-object (filtered).
// We map TATR's classes onto the NemotronDecode triplet so the existing
// NemotronAssembly + nemotron_to_html pipeline can render unchanged:
//   rows    ← class 2
//   columns ← class 1
//   cells   ← class 5 (spanning cells; non-spanning grid is filled by
//             row×col intersection inside assemble_nemotron)
//   headers ← classes 3 and 4
struct TatrBox {
  std::array<float, 4> bbox;  // x1, y1, x2, y2 in original region pixels
  float                score = 0.0f;
  int                  class_id = -1;
};

inline constexpr int   kTatrCanvas       = 800;
inline constexpr int   kTatrNumQueries   = 125;
inline constexpr int   kTatrNumClasses   = 7;
inline constexpr int   kTatrBgClass      = 6;
// Conservative defaults — DETR-family models concentrate confidence so a
// 0.5 floor for rows/cols/headers and a 0.6 floor for spanning cells avoids
// junk while keeping recall high. Tunable from the env if needed later.
inline constexpr float kTatrRowColThresh = 0.50f;
inline constexpr float kTatrCellThresh   = 0.60f;
inline constexpr float kTatrHeaderThresh = 0.50f;
inline constexpr float kTatrRowColNmsIoU = 0.55f;
inline constexpr float kTatrCellNmsIoU   = 0.30f;
inline constexpr int   kTatrMaxRows      = 60;
inline constexpr int   kTatrMaxCols      = 30;

// Decode TATR raw logits + pred_boxes for one image. `logits` is row-major
// [num_queries, kTatrNumClasses], `pred_boxes` is [num_queries, 4] (cxcywh,
// each in [0,1] over the padded letterbox canvas, NOT over the original
// region). `letterbox_size` is kTatrCanvas; `new_w`/`new_h` are the post-
// resize dims of the region that landed inside the canvas (so we can undo
// the letterbox padding). `ori_w`/`ori_h` are the original region size.
NemotronDecode decode_tatr(const float* logits,
                           const float* pred_boxes,
                           std::size_t num_queries,
                           int letterbox_size,
                           int new_w,
                           int new_h,
                           int ori_w,
                           int ori_h);

} // namespace turbo_ocr::table
