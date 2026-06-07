#include "turbo_ocr/detection/cpu_paddle_det.h"
#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/detection/det_postprocess.h"
#include "turbo_ocr/common/stage_profiler.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <opencv2/imgproc.hpp>

using namespace turbo_ocr;
using namespace turbo_ocr::detection;
using turbo_ocr::engine::CpuEngine;

namespace {
// Fuse resize-normalize-pack into one split + 3 scaled convertTo passes
// (writes straight into the NCHW buffer). Changes float rounding vs the
// per-channel MatExpr path, so it sits behind an env flag for A/B until the
// lead confirms box count + text_sig are unchanged.
bool det_fused_pre() {
  static const bool e = [] {
    const char *v = std::getenv("TURBO_DET_FUSED_PRE");
    return v && v[0] == '1';
  }();
  return e;
}
} // namespace

bool CpuPaddleDet::load_model(const std::string &model_path) {
  // Same DET_MAX_SIDE source as the GPU detector + the TRT engine builder.
  kMaxSideLen = read_det_max_side();
  engine_ = std::make_unique<CpuEngine>(model_path);
  return engine_->load();
}

std::vector<Box> CpuPaddleDet::run(const cv::Mat &img) {
  int h = img.rows;
  int w = img.cols;
  float ratio = 1.0f;
  if (std::max(h, w) > kMaxSideLen) {
    ratio = (h > w) ? static_cast<float>(kMaxSideLen) / h
                    : static_cast<float>(kMaxSideLen) / w;
  }
  int resize_h = std::max(static_cast<int>(round(h * ratio / 32.0) * 32), 32);
  int resize_w = std::max(static_cast<int>(round(w * ratio / 32.0) * 32), 32);

  const int plane_size = resize_h * resize_w;
  input_data_buf_.resize(3 * plane_size);

  {
    turbo_ocr::prof::Scope _s(turbo_ocr::prof::DET_PRE);
    cv::resize(img, resized_, cv::Size(resize_w, resize_h));

    if (det_fused_pre()) {
      // Single pass: BGR uint8 -> planar RGB float with per-channel
      // (1/(255*std)) scale and (-mean/std) shift, written into the NCHW buffer.
      cv::split(resized_, bgr_);
      cv::Mat r_plane(resize_h, resize_w, CV_32F, input_data_buf_.data());
      cv::Mat g_plane(resize_h, resize_w, CV_32F,
                      input_data_buf_.data() + plane_size);
      cv::Mat b_plane(resize_h, resize_w, CV_32F,
                      input_data_buf_.data() + 2 * plane_size);
      bgr_[2].convertTo(r_plane, CV_32F, 1.0 / (255.0 * 0.229), -0.485 / 0.229);
      bgr_[1].convertTo(g_plane, CV_32F, 1.0 / (255.0 * 0.224), -0.456 / 0.224);
      bgr_[0].convertTo(b_plane, CV_32F, 1.0 / (255.0 * 0.225), -0.406 / 0.225);
    } else {
      // Default path (bit-identical to the original): convert, split, normalize
      // per channel, pack RGB. resized_/float_img_ reused to avoid big allocs.
      resized_.convertTo(float_img_, CV_32F, 1.0 / 255.0);
      cv::Mat channels[3];
      cv::split(float_img_, channels);
      channels[0] = (channels[0] - 0.406f) / 0.225f; // B
      channels[1] = (channels[1] - 0.456f) / 0.224f; // G
      channels[2] = (channels[2] - 0.485f) / 0.229f; // R

      // NCHW, RGB order (PaddleOCR convention)
      std::memcpy(input_data_buf_.data(), channels[2].data,
                  plane_size * sizeof(float));
      std::memcpy(input_data_buf_.data() + plane_size, channels[1].data,
                  plane_size * sizeof(float));
      std::memcpy(input_data_buf_.data() + 2 * plane_size, channels[0].data,
                  plane_size * sizeof(float));
    }
    input_shape_buf_ = {1, 3, static_cast<int64_t>(resize_h),
                        static_cast<int64_t>(resize_w)};
  }

  CpuEngine::InferResult result;
  {
    turbo_ocr::prof::Scope _s(turbo_ocr::prof::DET_INFER);
    result = engine_->infer(input_data_buf_.data(), input_shape_buf_);
  }

  if (result.data.empty())
    return {};

  turbo_ocr::prof::Scope _s(turbo_ocr::prof::DET_POST);

  // Output shape: [1, 1, resize_h, resize_w]
  cv::Mat pred_map(resize_h, resize_w, CV_32F, result.data.data());

  // Threshold to bitmap. compare(CMP_GT) yields a CV_8U 0/255 mask in one SIMD
  // pass -- identical predicate to THRESH_BINARY, no float temp / extra convert.
  cv::compare(pred_map, kDetDbThresh, bitmap_, cv::CMP_GT);

  // Find contours and extract boxes
  return extract_boxes_from_bitmap(
      pred_map, bitmap_, h, w, resize_h, resize_w,
      kDetDbBoxThresh, kDetDbUnclipRatio, kMinBoxSide, kMinUnclippedSide,
      shifted_buf_, mask_buf_, contours_buf_, hierarchy_buf_);
}
