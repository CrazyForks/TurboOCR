#pragma once

#include <memory>
#include <vector>

#include "turbo_ocr/classification/cpu_doc_orientation.h"
#include "turbo_ocr/classification/cpu_paddle_cls.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/detection/cpu_paddle_det.h"
#include "turbo_ocr/layout/cpu_paddle_layout.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/pipeline/i_ocr_pipeline.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/recognition/cpu_paddle_rec.h"

namespace turbo_ocr::pipeline {

class CpuOcrPipeline : public IOcrPipeline {
public:
  CpuOcrPipeline();
  ~CpuOcrPipeline() noexcept override = default;

  [[nodiscard]] bool init(const std::string &det_model, const std::string &rec_model,
                          const std::string &rec_dict, const std::string &cls_model = "") override;

  [[nodiscard]] bool load_layout_model(const std::string &onnx_path);

  // Load the document-orientation model (ONNX). Optional; powers autorotate.
  [[nodiscard]] bool load_doc_ori_model(const std::string &onnx_path);
  [[nodiscard]] bool has_doc_ori() const noexcept { return use_doc_ori_; }
  // Page's detected clockwise rotation (0/90/180/270), or 0 if unavailable.
  [[nodiscard]] int detect_orientation(const cv::Mat &bgr);

  void warmup() override;

  [[nodiscard]] std::vector<OCRResultItem> run(const cv::Mat &img) override;

  /// Run OCR + optional layout detection.
  /// `want_reading_order` is opt-in and has no effect when layout is
  /// unavailable — the returned vector stays empty so the JSON serializer
  /// omits the key.
  [[nodiscard]] OcrPipelineResult run_with_layout(const cv::Mat &img,
                                                   bool want_layout = false,
                                                   bool want_reading_order = false);

private:
  std::unique_ptr<detection::CpuPaddleDet> det_;
  std::unique_ptr<classification::CpuPaddleCls> cls_;
  std::unique_ptr<recognition::CpuPaddleRec> rec_;
  std::unique_ptr<layout::CpuPaddleLayout> layout_;
  std::unique_ptr<classification::CpuDocOrientation> doc_ori_;

  bool use_cls_ = false;
  bool use_doc_ori_ = false;
};

} // namespace turbo_ocr::pipeline
