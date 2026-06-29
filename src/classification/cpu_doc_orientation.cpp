#include "turbo_ocr/classification/cpu_doc_orientation.h"

#include "turbo_ocr/classification/doc_orientation_common.h"

using namespace turbo_ocr::classification;
using turbo_ocr::engine::CpuEngine;

bool CpuDocOrientation::load_model(const std::string &model_path) {
  engine_ = std::make_unique<CpuEngine>(model_path);
  return engine_->load();
}

int CpuDocOrientation::detect(const cv::Mat &bgr) {
  if (!engine_ || bgr.empty() || bgr.type() != CV_8UC3) return 0;
  std::vector<float> input = doc_ori_preprocess(bgr);
  auto res = engine_->infer(
      input.data(), {1, 3, kDocOriSize, kDocOriSize});
  if (res.data.size() < 4) return 0;
  return doc_ori_label(res.data.data());
}
