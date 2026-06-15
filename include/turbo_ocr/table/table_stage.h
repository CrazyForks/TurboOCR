#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/router/router_types.h"
#include "turbo_ocr/table/slanext.h"
#include "turbo_ocr/table/table_cell.h"
#include "turbo_ocr/table/table_cls.h"

namespace turbo_ocr::table {

// Umbrella orchestrator for the table modality (plan 02 §7).
// Owns 1× TableCls + 2× TableCellDet (wired / wireless) + 2× SlaNeXt
// (wired / wireless). The dedicated table_stream_ lives one level up on
// OcrPipeline — TableStage is fed it per-call so the same stage object
// can be reused across the pipeline's lifetime.
//
// Caller flow (OcrPipeline::run_with_layout):
//   stage.run(gpu_img, layout, ocr_results, table_stream_, det_only_event_);
//
// The stage:
//   1. Bails immediately if `layout` contains no class_id == 21 cells —
//      this is the text-only short-circuit.  Zero CUDA calls.
//   2. waitEvent(det_only_event) on the caller-supplied stream.
//   3. Phase A: classify all regions via TableCls (batched, ≤8).
//   4. Phase B: per variant, run CellDet (batch=1 in v1) and SlaNeXt on
//      the same stream.  Both engines self-sync inside their collect().
//   5. Phase C: cell↔OCR match (R-tree) + HTML reconstruct, append to
//      the returned vector keyed by layout_id (= index in `layout`).
class TableStage {
public:
  TableStage() = default;
  ~TableStage() = default;
  TableStage(const TableStage&)            = delete;
  TableStage& operator=(const TableStage&) = delete;

  // Initialise all five TRT engines. Pass an empty string for any
  // wireless slot to disable wireless support — every classified-as-
  // wireless region will be routed through the wired engines instead.
  // Returns false if any *required* engine (cls / wired pair) fails.
  [[nodiscard]] bool init(const std::string& table_cls_trt,
                          const std::string& cell_wired_trt,
                          const std::string& cell_wireless_trt,
                          const std::string& slanext_wired_trt,
                          const std::string& slanext_wireless_trt);

  // Run the stage on a page. `layout` is the full layout vector — the
  // stage filters internally for class_id == 21 cells. `ocr_lines` is
  // the rec output for the page (used by the cell matcher).
  //
  // Returns one TableResult per processed region, with `layout_id` set
  // to the index of the originating layout cell. Empty vector when no
  // table layout cells exist — no CUDA work is dispatched in that case.
  std::vector<router::TableResult>
  run(const GpuImage& page,
      const std::vector<layout::LayoutBox>& layout,
      const std::vector<OCRResultItem>& ocr_lines,
      cudaStream_t stream,
      cudaEvent_t det_only_event);

  [[nodiscard]] bool has_wireless() const noexcept {
    return cell_[1] != nullptr && slanext_[1] != nullptr;
  }

private:
  std::unique_ptr<TableCls>     cls_;
  std::unique_ptr<TableCellDet> cell_[2];      // [0]=wired, [1]=wireless
  std::unique_ptr<SlaNeXt>      slanext_[2];
};

} // namespace turbo_ocr::table
