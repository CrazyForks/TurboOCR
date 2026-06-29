#include "turbo_ocr/table/vlm_table_recognizer.h"

#include <iostream>

namespace turbo_ocr::table {

bool VLMTableRecognizer::load() {
  auto v = std::make_unique<VLMTable>();
  if (!v->init()) {
    std::cerr << "[vlm-table] init/health-check failed — tables disabled\n";
    return false;
  }
  vlm_ = std::move(v);
  std::cout << "[Pipeline] Table backend=vlm (PaddleOCR-VL via vLLM, OTSL->HTML)\n";
  return true;
}

std::vector<router::TableResult>
VLMTableRecognizer::run(const GpuImage &page, const std::vector<Box> &regions,
                        const std::vector<OCRResultItem> & /*page_ocr*/,
                        cudaStream_t stream) {
  std::vector<router::TableResult> out;
  out.reserve(regions.size());
  // VLMTable reads the table image directly; page_ocr unused.
  std::vector<std::string> htmls = vlm_->run(page, regions, stream);
  for (std::size_t i = 0; i < regions.size(); ++i) {
    router::TableResult tr;
    tr.layout_id = -1;  // stamped by caller
    tr.html      = (i < htmls.size()) ? std::move(htmls[i]) : std::string();
    tr.score     = 0.0f;
    tr.box       = regions[i];
    out.push_back(std::move(tr));
  }
  return out;
}

} // namespace turbo_ocr::table
