#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/table/table_recognizer.h"
#include "turbo_ocr/table/slanext_enc_split.h"
#include "turbo_ocr/table/table_cls.h"

namespace turbo_ocr::table {

// SLANeXt encoder-split table backend (default). Owns the wired encoder, an
// optional wireless encoder, and an optional table_cls router. Reads its env
// knobs (TABLE_SLANEXT_ENCODER_ONNX + decoder/dict defaults next to it,
// TABLE_SLANEXT_WIRELESS_ENCODER_ONNX, TABLE_CLS_ONNX) in load(). Per region:
// TRT FP16 encoder + host GRU decode -> structure tokens + per-cell quads;
// cells filled from the page text-OCR via match_cells_to_ocr + reconstruct_html.
class SlanextTableRecognizer final : public ITableRecognizer {
public:
  [[nodiscard]] bool load() override;
  [[nodiscard]] std::vector<router::TableResult>
  run(const GpuImage &page, const std::vector<Box> &regions,
      const std::vector<OCRResultItem> &page_ocr, cudaStream_t stream) override;
  [[nodiscard]] bool is_ready() const noexcept override { return wired_ != nullptr; }
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return "slanext";
  }
  void set_cell_recognizer(recognition::PaddleRec *r) noexcept override {
    cell_rec_ = r;
  }

private:
  std::unique_ptr<SlanextEncSplit> wired_;
  std::unique_ptr<SlanextEncSplit> wireless_;
  std::unique_ptr<TableCls>        cls_;
  recognition::PaddleRec          *cell_rec_ = nullptr;  // not owned; per-cell fill
};

} // namespace turbo_ocr::table
