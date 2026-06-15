#include "turbo_ocr/table/table_stage.h"

#include <algorithm>
#include <iostream>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/table/cell_matcher.h"
#include "turbo_ocr/table/html_reconstruct.h"

namespace turbo_ocr::table {

namespace {

constexpr int kTableLayoutClassId = 21;

// Cheap AABB intersect test — `inner` center inside `outer` rect.
bool ocr_center_in_rect(const std::array<int, 4>& outer,
                        const Box& q) {
  int cx = 0, cy = 0;
  for (int k = 0; k < 4; ++k) { cx += q[k][0]; cy += q[k][1]; }
  cx /= 4; cy /= 4;
  return cx >= outer[0] && cx <= outer[2] && cy >= outer[1] && cy <= outer[3];
}

} // namespace

bool TableStage::init(const std::string& table_cls_trt,
                      const std::string& cell_wired_trt,
                      const std::string& cell_wireless_trt,
                      const std::string& slanext_wired_trt,
                      const std::string& slanext_wireless_trt) {
  cls_ = std::make_unique<TableCls>();
  if (!cls_->load_model(table_cls_trt)) {
    std::cerr << "[table_stage] failed to load table-cls engine\n";
    cls_.reset();
    return false;
  }

  cell_[0] = std::make_unique<TableCellDet>();
  if (!cell_[0]->load_model(cell_wired_trt)) {
    std::cerr << "[table_stage] failed to load wired cell-det engine\n";
    return false;
  }
  slanext_[0] = std::make_unique<SlaNeXt>();
  if (!slanext_[0]->load_model(slanext_wired_trt)) {
    std::cerr << "[table_stage] failed to load wired SLANeXt engine\n";
    return false;
  }

  // Wireless half is optional in v1: if either path is empty we leave the
  // slot null and re-route wireless-classified regions through wired.
  if (!cell_wireless_trt.empty() && !slanext_wireless_trt.empty()) {
    cell_[1] = std::make_unique<TableCellDet>();
    if (!cell_[1]->load_model(cell_wireless_trt)) {
      std::cerr << "[table_stage] wireless cell-det failed to load — "
                   "falling back to wired for all regions\n";
      cell_[1].reset();
    }
    slanext_[1] = std::make_unique<SlaNeXt>();
    if (!slanext_[1]->load_model(slanext_wireless_trt)) {
      std::cerr << "[table_stage] wireless SLANeXt failed to load — "
                   "falling back to wired for all regions\n";
      slanext_[1].reset();
    }
    if (!cell_[1] || !slanext_[1]) {
      cell_[1].reset();
      slanext_[1].reset();
    }
  }

  return true;
}

std::vector<router::TableResult>
TableStage::run(const GpuImage& page,
                const std::vector<layout::LayoutBox>& layout,
                const std::vector<OCRResultItem>& ocr_lines,
                cudaStream_t stream,
                cudaEvent_t det_only_event) {
  std::vector<router::TableResult> results;
  if (!cls_ || layout.empty()) return results;

  // 1) Collect class_id == 21 regions, preserving original layout index.
  std::vector<int>  layout_ids;       // index into `layout`
  std::vector<Box>  regions;          // page-pixel quad for each table cell
  layout_ids.reserve(8);
  regions.reserve(8);
  for (std::size_t i = 0; i < layout.size(); ++i) {
    if (layout[i].class_id == kTableLayoutClassId) {
      layout_ids.push_back(static_cast<int>(i));
      regions.push_back(layout[i].box);
    }
  }
  if (regions.empty()) return results;        // text-only short-circuit

  // 2) Sync entry — wait for det/cls on the caller stream to finish
  //    writing gpu_img before we crop from it.
  if (det_only_event) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, det_only_event, 0));
  }

  // 3) Phase A: classify all regions. Internally syncs `stream` for the
  //    (N, 2) D2H readback — needed before we can dispatch Phase B.
  auto variants = cls_->classify_batch(page, regions, stream);
  if (variants.size() != regions.size()) {
    std::cerr << "[table_stage] classifier returned " << variants.size()
              << " labels for " << regions.size() << " regions\n";
    return results;
  }

  // 4) Phase B: cell-det + SLANeXt per region. v1 is batch=1 per
  //    table-gpu's note; once the engines support batched enqueue we'll
  //    group same-variant regions and replay this loop.
  results.reserve(regions.size());
  for (std::size_t r = 0; r < regions.size(); ++r) {
    // Pick variant slot, falling back to wired when wireless is missing.
    int v = (variants[r] == TableVariant::Wireless) ? 1 : 0;
    if (!cell_[v] || !slanext_[v]) v = 0;

    auto bb = aabb(regions[r]);
    const int rx = std::clamp(bb[0], 0, page.cols - 1);
    const int ry = std::clamp(bb[1], 0, page.rows - 1);
    const int rw = std::clamp(bb[2] - rx, 1, page.cols - rx);
    const int rh = std::clamp(bb[3] - ry, 1, page.rows - ry);

    // Cell-det: enqueue → collect (internally waits on its own D2H event).
    if (!cell_[v]->enqueue(page, rx, ry, rw, rh, stream)) {
      std::cerr << "[table_stage] cell-det enqueue failed for region "
                << r << '\n';
      continue;
    }
    auto cells_det = cell_[v]->collect();

    // SLANeXt: same stream, separate event. infer() = enqueue + collect.
    auto structure = slanext_[v]->infer(page, regions[r], stream);
    if (structure.structure.empty()) continue;

    // 5) Phase C — pure CPU postprocess: cell↔OCR + HTML reconstruct.
    // Filter OCR lines to those whose center falls inside the table
    // region.  Keeps the R-tree small (<200 candidates per typical
    // table) and avoids matching cells against text from other regions.
    std::vector<OcrLine>     region_ocr;
    std::vector<std::string> region_texts;
    region_ocr.reserve(ocr_lines.size());
    region_texts.reserve(ocr_lines.size());
    const std::array<int, 4> rect_aabb{bb[0], bb[1], bb[2], bb[3]};
    for (const auto& it : ocr_lines) {
      if (!ocr_center_in_rect(rect_aabb, it.box)) continue;
      auto line_bb = aabb(it.box);
      region_ocr.push_back(OcrLine{
          {static_cast<float>(line_bb[0]), static_cast<float>(line_bb[1]),
           static_cast<float>(line_bb[2]), static_cast<float>(line_bb[3])},
          it.text});
      region_texts.push_back(it.text);
    }

    // SLANeXt cells are quads; the matcher consumes the same 8-int
    // layout that StructureCell carries.
    std::vector<std::array<int, 8>> quads;
    quads.reserve(structure.cells.size());
    for (const auto& c : structure.cells) quads.push_back(c.bbox);

    auto matched = match_cells_to_ocr(quads, region_ocr);
    auto html    = reconstruct_html(structure.structure, matched, region_texts);

    router::TableResult tr;
    tr.layout_id = layout_ids[r];
    tr.html      = std::move(html);
    tr.score     = structure.structure_score;
    tr.box       = regions[r];
    results.push_back(std::move(tr));

    (void)cells_det;  // RT-DETR detections currently unused by the matcher
                      // (it uses SLANeXt's own per-token quads). Reserved
                      // for the cross-check pass once cell_matcher exposes
                      // an alternate path.
  }

  return results;
}

} // namespace turbo_ocr::table
