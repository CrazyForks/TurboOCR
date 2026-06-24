#pragma once

#include <vector>

#include "turbo_ocr/common/types.h"        // OCRResultItem, Box
#include "turbo_ocr/pipeline/pipeline_result.h"  // OcrPipelineResult, finalize_deferred

namespace turbo_ocr::pipeline::detail {

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
