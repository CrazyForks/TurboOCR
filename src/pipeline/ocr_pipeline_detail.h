#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "turbo_ocr/common/types.h"        // OCRResultItem, Box
#include "turbo_ocr/pipeline/pipeline_result.h"  // OcrPipelineResult, finalize_deferred

namespace turbo_ocr::pipeline::detail {

// No-silent-failure guard for the base OCR/recognition stage. Detection found
// `num_boxes` text regions but recognition produced no usable text — flag the
// result degraded so the response can never be a clean empty 200 that looks
// identical to a genuinely text-free page. A page with zero detections is NOT
// degraded (correctly text-free). Mirrors the formula/table degraded contract.
inline void flag_text_degraded(OcrPipelineResult &out, std::size_t num_boxes) {
  if (num_boxes > 0 && out.results.empty()) {
    out.text_degraded = true;
    out.text_warning =
        "text stage degraded: detection found " + std::to_string(num_boxes) +
        " text region(s) but recognition produced no usable text "
        "(all crops decoded empty/blank; not a genuinely blank page)";
  }
}

// Backend-independent table-region adjustment (kept out of the recognizers so
// the env knobs live in one place and backends receive an already-adjusted
// box):
//   TABLE_CROP_MODE=detunion — snap to the tight AABB of the det text boxes
//     inside the layout box (so the region can only tighten).
//   TABLE_CROP_MARGIN — expand by this fraction per side (default 0.03, the
//     measured best on the 117-table set; layout boxes tend to clip border
//     rows/cols and structure-TEDS is sensitive to missing edge cells).
Box adjust_table_region(const Box &in,
                        const std::vector<OCRResultItem> &results);

} // namespace turbo_ocr::pipeline::detail
