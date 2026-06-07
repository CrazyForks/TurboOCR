#include "turbo_ocr/classification/cpu_paddle_cls.h"
#include "turbo_ocr/recognition/crop_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <opencv2/imgproc.hpp>

using namespace turbo_ocr::classification;
using turbo_ocr::engine::CpuEngine;
using turbo_ocr::Box;
using turbo_ocr::recognition::get_rotate_crop_image;

namespace {
// Pack one BGR crop into the NCHW RGB float plane at `dst` (3*plane floats):
// resize to (W,H), normalize pixel/127.5-1.0, split, RGB order. Matches the
// scalar path exactly so batched and per-crop inference are bit-identical.
inline void pack_cls_crop(const cv::Mat &crop, int H, int W, float *dst) {
  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(W, H));
  cv::Mat float_img;
  resized.convertTo(float_img, CV_32F, 1.0 / 127.5, -1.0);
  const int plane = H * W;
  cv::Mat channels[3];
  cv::split(float_img, channels);
  std::memcpy(dst, channels[2].data, plane * sizeof(float));
  std::memcpy(dst + plane, channels[1].data, plane * sizeof(float));
  std::memcpy(dst + 2 * plane, channels[0].data, plane * sizeof(float));
}

inline bool cls_batch_enabled() {
  static const bool e = [] {
    const char *v = std::getenv("CLS_BATCH");
    return v && v[0] == '1';
  }();
  return e;
}
} // namespace

bool CpuPaddleCls::load_model(const std::string &model_path) {
  engine_ = std::make_unique<CpuEngine>(model_path);
  return engine_->load();
}

void CpuPaddleCls::run(const cv::Mat &img, std::vector<Box> &boxes) {
  if (boxes.empty())
    return;

  const int plane = kClsImageH * kClsImageW;

  // Batched path: one ORT Run for every box (cls.onnx has a fixed 80x160
  // input and a dynamic batch dim, so all crops share one tensor). Output
  // is {N,2}; row i drives the same swap decision as the scalar path.
  if (cls_batch_enabled()) {
    const int n = static_cast<int>(boxes.size());
    std::vector<float> input_buf(static_cast<size_t>(n) * 3 * plane);
    for (int i = 0; i < n; ++i) {
      cv::Mat crop = get_rotate_crop_image(img, boxes[i]);
      pack_cls_crop(crop, kClsImageH, kClsImageW,
                    input_buf.data() + static_cast<size_t>(i) * 3 * plane);
    }
    const std::vector<int64_t> shape = {n, 3, kClsImageH, kClsImageW};
    auto result = engine_->infer_batch(input_buf.data(), shape);
    if (static_cast<int>(result.data.size()) >= 2 * n) {
      for (int i = 0; i < n; ++i) {
        float s0 = result.data[2 * i];
        float s180 = result.data[2 * i + 1];
        if (s180 > s0 && s180 > kClsThresh) {
          std::swap(boxes[i][0], boxes[i][2]);
          std::swap(boxes[i][1], boxes[i][3]);
        }
      }
    }
    return;
  }

  std::vector<float> input_buf(3 * plane);
  const std::vector<int64_t> input_shape = {1, 3, kClsImageH, kClsImageW};
  for (size_t i = 0; i < boxes.size(); ++i) {
    cv::Mat crop = get_rotate_crop_image(img, boxes[i]);
    pack_cls_crop(crop, kClsImageH, kClsImageW, input_buf.data());
    auto result = engine_->infer(input_buf.data(), input_shape);
    if (result.data.size() >= 2) {
      float score_0 = result.data[0];
      float score_180 = result.data[1];
      if (score_180 > score_0 && score_180 > kClsThresh) {
        std::swap(boxes[i][0], boxes[i][2]);
        std::swap(boxes[i][1], boxes[i][3]);
      }
    }
  }
}
