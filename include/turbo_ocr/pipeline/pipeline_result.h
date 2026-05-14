#pragma once

#include <vector>

#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/router/router_types.h"

namespace turbo_ocr::pipeline {

/// Bundles text OCR results + optional layout detections from a single
/// pipeline run. `layout` is empty when layout was not requested or the
/// pipeline has no layout model. `reading_order` is filled only when
/// reading-order assignment was requested. `tables` / `formulas` are
/// populated by the CUA router stages — empty by default so the
/// back-compat serializer emits a byte-identical response on text-only
/// pages.
struct OcrPipelineResult {
  std::vector<OCRResultItem>             results;
  std::vector<layout::LayoutBox>         layout;
  std::vector<int>                       reading_order;
  std::vector<router::TableResult>       tables;
  std::vector<router::FormulaResult>     formulas;
};

} // namespace turbo_ocr::pipeline
