#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "turbo_ocr/engine/cpu_engine.h"

namespace turbo_ocr::classification {

/// CPU document-orientation classifier (PP-LCNet_x1_0_doc_ori, ONNX Runtime).
/// Mirror of DocOrientation for the CPU-only build. Detects a page's
/// clockwise rotation ∈ {0,90,180,270}.
class CpuDocOrientation {
public:
  CpuDocOrientation() = default;
  ~CpuDocOrientation() noexcept = default;

  [[nodiscard]] bool load_model(const std::string &model_path);

  /// Page's detected clockwise rotation (0/90/180/270), or 0 if unavailable.
  [[nodiscard]] int detect(const cv::Mat &bgr);

private:
  std::unique_ptr<engine::CpuEngine> engine_;
};

} // namespace turbo_ocr::classification
