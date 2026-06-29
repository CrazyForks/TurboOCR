#include "turbo_ocr/layout/cpu_paddle_layout.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/layout/layout_postfilter.h"

namespace turbo_ocr::layout {

struct CpuPaddleLayout::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "CpuLayout"};
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions allocator;
};

CpuPaddleLayout::CpuPaddleLayout() = default;
CpuPaddleLayout::~CpuPaddleLayout() noexcept = default;

bool CpuPaddleLayout::load_model(const std::string &onnx_path) {
  impl_ = std::make_unique<Impl>();

  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(2);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  try {
    impl_->session = std::make_unique<Ort::Session>(
        impl_->env, onnx_path.c_str(), opts);
    std::cout << "[cpu_layout] Loaded " << onnx_path << '\n';
    return true;
  } catch (const Ort::Exception &e) {
    std::cerr << "[cpu_layout] Failed to load " << onnx_path << ": "
              << e.what() << '\n';
    impl_.reset();
    return false;
  }
}

std::vector<LayoutBox> CpuPaddleLayout::run(const cv::Mat &img,
                                             float score_threshold) {
  std::vector<LayoutBox> out;
  if (!impl_ || !impl_->session) return out;

  const int orig_h = img.rows;
  const int orig_w = img.cols;

  // 1. Preprocess: resize to 800x800, normalize to [0,1], CHW
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(kInputSize, kInputSize));

  std::vector<float> input_chw(3 * kInputSize * kInputSize);
  constexpr int plane = kInputSize * kInputSize;
  cv::Mat bgr[3];
  cv::split(resized, bgr);
  cv::Mat p_b(kInputSize, kInputSize, CV_32F, input_chw.data());
  cv::Mat p_g(kInputSize, kInputSize, CV_32F, input_chw.data() + plane);
  cv::Mat p_r(kInputSize, kInputSize, CV_32F, input_chw.data() + 2 * plane);
  bgr[0].convertTo(p_b, CV_32F, 1.0 / 255.0); // B
  bgr[1].convertTo(p_g, CV_32F, 1.0 / 255.0); // G
  bgr[2].convertTo(p_r, CV_32F, 1.0 / 255.0); // R

  // 2. Build inputs: im_shape=[800,800], scale_factor=[800/h, 800/w]
  std::array<float, 2> im_shape = {
      static_cast<float>(kInputSize), static_cast<float>(kInputSize)};
  std::array<float, 2> scale_factor = {
      static_cast<float>(kInputSize) / static_cast<float>(orig_h),
      static_cast<float>(kInputSize) / static_cast<float>(orig_w)};

  // 3. Create ONNX tensors
  auto memory_info = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);

  std::array<int64_t, 4> img_shape = {1, 3, kInputSize, kInputSize};
  std::array<int64_t, 2> vec_shape = {1, 2};

  auto img_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_chw.data(), input_chw.size(),
      img_shape.data(), img_shape.size());
  auto im_shape_tensor = Ort::Value::CreateTensor<float>(
      memory_info, im_shape.data(), im_shape.size(),
      vec_shape.data(), vec_shape.size());
  auto sf_tensor = Ort::Value::CreateTensor<float>(
      memory_info, scale_factor.data(), scale_factor.size(),
      vec_shape.data(), vec_shape.size());

  // 4. Get input/output names from model
  size_t num_inputs = impl_->session->GetInputCount();
  size_t num_outputs = impl_->session->GetOutputCount();

  std::vector<std::string> input_names_str, output_names_str;
  std::vector<const char *> input_names, output_names;
  for (size_t i = 0; i < num_inputs; ++i) {
    auto name = impl_->session->GetInputNameAllocated(i, impl_->allocator);
    input_names_str.push_back(name.get());
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    auto name = impl_->session->GetOutputNameAllocated(i, impl_->allocator);
    output_names_str.push_back(name.get());
  }
  for (auto &s : input_names_str) input_names.push_back(s.c_str());
  for (auto &s : output_names_str) output_names.push_back(s.c_str());

  // 5. Map input names to tensors (order may vary)
  std::vector<Ort::Value> inputs;
  inputs.reserve(num_inputs);
  for (const auto &name : input_names_str) {
    if (name.find("image") != std::string::npos)
      inputs.push_back(std::move(img_tensor));
    else if (name == "im_shape")
      inputs.push_back(std::move(im_shape_tensor));
    else if (name == "scale_factor")
      inputs.push_back(std::move(sf_tensor));
  }

  if (inputs.size() != num_inputs) {
    std::cerr << "[cpu_layout] Input name mismatch\n";
    return out;
  }

  // 6. Run inference
  std::vector<Ort::Value> outputs;
  try {
    outputs = impl_->session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(), inputs.data(), inputs.size(),
        output_names.data(), output_names.size());
  } catch (const Ort::Exception &e) {
    std::cerr << "[cpu_layout] Inference failed: " << e.what() << '\n';
    return out;
  }

  // 7. Find the (N, 7) detection output
  const float *dets = nullptr;
  int n_rows = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto info = outputs[i].GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    if (shape.size() == 2 && shape[1] == 7) {
      dets = outputs[i].GetTensorData<float>();
      n_rows = static_cast<int>(shape[0]);
      break;
    }
  }
  if (!dets || n_rows <= 0) return out;
  n_rows = std::min(n_rows, kMaxDetections);

  // 8. Decode detections
  out.reserve(static_cast<size_t>(n_rows));
  for (int i = 0; i < n_rows; ++i) {
    const float *row = dets + i * 7;
    float score = row[1];
    if (score < score_threshold) continue;

    int cls = static_cast<int>(row[0]);
    if (cls < 0 || cls >= static_cast<int>(kLayoutLabels.size())) continue;

    int x0 = std::clamp(static_cast<int>(row[2]), 0, orig_w - 1);
    int y0 = std::clamp(static_cast<int>(row[3]), 0, orig_h - 1);
    int x1 = std::clamp(static_cast<int>(row[4]), 0, orig_w - 1);
    int y1 = std::clamp(static_cast<int>(row[5]), 0, orig_h - 1);
    if (x1 <= x0 || y1 <= y0) continue;

    LayoutBox lb;
    lb.class_id = cls;
    lb.score = score;
    lb.read_order = static_cast<int>(row[6]);
    lb.box[0] = {x0, y0};
    lb.box[1] = {x1, y0};
    lb.box[2] = {x1, y1};
    lb.box[3] = {x0, y1};
    out.push_back(lb);
  }

  // 9. Shared NMS + oversized-image drop + LAYOUT_MERGE_MODE reconciliation.
  return postfilter_layout_boxes(std::move(out), orig_h, orig_w);
}

} // namespace turbo_ocr::layout
