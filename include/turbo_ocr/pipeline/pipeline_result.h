#pragma once

#include <future>
#include <string>
#include <vector>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/router/router_types.h"

namespace turbo_ocr::formula { class IFormulaRecognizer; }
namespace turbo_ocr::table   { class ITableRecognizer; }

namespace turbo_ocr::pipeline {

// Async-decouple carrier (highly-async serving path). When a remote backend's
// run is deferred (defer_external + supports_async), the GPU pipeline worker
// submits crops non-blocking and stashes one future + its target metadata per
// region HERE, then returns immediately — freeing its GPU slot. finalize_
// deferred() awaits + parses + assembles OFF the GPU worker (on the work-pool
// thread). Empty on the synchronous path, so text-only / local-backend / gRPC
// / PDF / batch results are byte-identical.
struct PendingCrop {
  int                      layout_id = -1;
  turbo_ocr::Box           box{};
  float                    score = 0.0f;
  std::future<std::string> fut;   // raw endpoint response (pre-parse)
};
struct PendingExternal {
  const formula::IFormulaRecognizer *formula_rec = nullptr; // for parse_async_result
  const table::ITableRecognizer     *table_rec   = nullptr;
  std::vector<PendingCrop> formula;
  std::vector<PendingCrop> table;
  [[nodiscard]] bool empty() const noexcept {
    return formula.empty() && table.empty();
  }
};

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
  PendingExternal                        pending;  // deferred async work (empty on sync path)
};

// Await + parse + assemble any deferred external (VLM) work into out.tables /
// out.formulas, then clear out.pending. Called on the work-pool / HTTP thread
// (NOT a GPU pipeline worker) after the deferred run_with_layout returns. A
// no-op when out.pending is empty, so it is always safe to call unconditionally.
void finalize_deferred(OcrPipelineResult &out);

} // namespace turbo_ocr::pipeline
