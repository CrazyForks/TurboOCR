#include "turbo_ocr/layout/cpu_paddle_layout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/layout/layout_postfilter.h"

namespace turbo_ocr::layout {

// One CpuPaddleLayout drives one ONNX session and, like its GPU sibling
// PaddleLayout, owns per-instance staging buffers reused across run() calls —
// so a single detector instance is single-threaded by construction (the
// pipeline pool gives each worker its own detector). Everything that does not
// change between calls — the input/output name tables, the input-name→role
// mapping, the CPU MemoryInfo, and the ORT input tensors themselves (thin views
// over the member staging buffers) — is resolved once at load time. run() then
// only refreshes buffer *contents* and calls Session::Run, with zero per-call
// heap churn for names, tensor objects, or the CHW image buffer.
struct CpuPaddleLayout::Impl {
  enum class InputRole { kImage, kImShape, kScaleFactor };

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "CpuLayout"};
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memory_info{
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

  // Model IO, resolved once in resolve_io().
  std::vector<std::string> input_name_storage;
  std::vector<std::string> output_name_storage;
  std::vector<const char *> input_names;   // c_str() views into *_storage
  std::vector<const char *> output_names;
  std::vector<InputRole> input_roles;      // parallel to input_names
  bool inputs_resolved = false;

  // Staging buffers (reused every run; never reallocated after load).
  cv::Mat resized;
  cv::Mat bgr[3];
  std::vector<float> input_chw =
      std::vector<float>(3 * static_cast<size_t>(kInputSize) * kInputSize);
  std::array<float, 2> im_shape{static_cast<float>(kInputSize),
                                static_cast<float>(kInputSize)};
  std::array<float, 2> scale_factor{1.0f, 1.0f};

  // ORT input Values built once as persistent views over the buffers above.
  // Session::Run reads them read-only, so they stay valid across calls as long
  // as the backing buffers don't move (they never do).
  std::vector<Ort::Value> input_tensors;

  void resolve_io() {
    const size_t ni = session->GetInputCount();
    const size_t no = session->GetOutputCount();

    input_name_storage.reserve(ni);
    for (size_t i = 0; i < ni; ++i)
      input_name_storage.emplace_back(
          session->GetInputNameAllocated(i, allocator).get());
    output_name_storage.reserve(no);
    for (size_t i = 0; i < no; ++i)
      output_name_storage.emplace_back(
          session->GetOutputNameAllocated(i, allocator).get());

    input_names.reserve(ni);
    for (const auto &s : input_name_storage) input_names.push_back(s.c_str());
    output_names.reserve(no);
    for (const auto &s : output_name_storage) output_names.push_back(s.c_str());

    // Map each input slot to a role (same rule the per-call loop used before:
    // any name containing "image" is the image tensor; exact "im_shape" /
    // "scale_factor" for the two metadata tensors). An unclassifiable slot
    // leaves inputs_resolved false so run() reports the mismatch, matching the
    // prior "Input name mismatch" behaviour.
    input_roles.reserve(ni);
    for (const auto &name : input_name_storage) {
      if (name.find("image") != std::string::npos)
        input_roles.push_back(InputRole::kImage);
      else if (name == "im_shape")
        input_roles.push_back(InputRole::kImShape);
      else if (name == "scale_factor")
        input_roles.push_back(InputRole::kScaleFactor);
      else
        return; // inputs_resolved stays false
    }

    build_input_tensors();
    inputs_resolved = true;
  }

  void build_input_tensors() {
    const std::array<int64_t, 4> img_shape{1, 3, kInputSize, kInputSize};
    const std::array<int64_t, 2> vec_shape{1, 2};
    input_tensors.clear();
    input_tensors.reserve(input_roles.size());
    for (const auto role : input_roles) {
      switch (role) {
        case InputRole::kImage:
          input_tensors.push_back(Ort::Value::CreateTensor<float>(
              memory_info, input_chw.data(), input_chw.size(),
              img_shape.data(), img_shape.size()));
          break;
        case InputRole::kImShape:
          input_tensors.push_back(Ort::Value::CreateTensor<float>(
              memory_info, im_shape.data(), im_shape.size(),
              vec_shape.data(), vec_shape.size()));
          break;
        case InputRole::kScaleFactor:
          input_tensors.push_back(Ort::Value::CreateTensor<float>(
              memory_info, scale_factor.data(), scale_factor.size(),
              vec_shape.data(), vec_shape.size()));
          break;
      }
    }
  }
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
  } catch (const Ort::Exception &e) {
    std::cerr << "[cpu_layout] Failed to load " << onnx_path << ": "
              << e.what() << '\n';
    impl_.reset();
    return false;
  }

  impl_->resolve_io();
  if (!impl_->inputs_resolved)
    std::cerr << "[cpu_layout] " << onnx_path
              << " has unexpected input names; run() will report a mismatch\n";
  std::cout << "[cpu_layout] Loaded " << onnx_path << '\n';
  return true;
}

std::vector<LayoutBox> CpuPaddleLayout::run(const cv::Mat &img,
                                             float score_threshold) {
  std::vector<LayoutBox> out;
  if (!impl_ || !impl_->session) return out;
  Impl &im = *impl_;
  if (!im.inputs_resolved) {
    std::cerr << "[cpu_layout] Input name mismatch\n";
    return out;
  }

  const int orig_h = img.rows;
  const int orig_w = img.cols;

  // 1. Preprocess: resize to 800x800, normalize to [0,1], CHW. Writes straight
  //    into the persistent CHW staging buffer that the cached image tensor
  //    already views — no reallocation, no extra copy.
  cv::resize(img, im.resized, cv::Size(kInputSize, kInputSize));

  constexpr int plane = kInputSize * kInputSize;
  cv::split(im.resized, im.bgr);
  cv::Mat p_b(kInputSize, kInputSize, CV_32F, im.input_chw.data());
  cv::Mat p_g(kInputSize, kInputSize, CV_32F, im.input_chw.data() + plane);
  cv::Mat p_r(kInputSize, kInputSize, CV_32F, im.input_chw.data() + 2 * plane);
  im.bgr[0].convertTo(p_b, CV_32F, 1.0 / 255.0); // B
  im.bgr[1].convertTo(p_g, CV_32F, 1.0 / 255.0); // G
  im.bgr[2].convertTo(p_r, CV_32F, 1.0 / 255.0); // R

  // 2. Refresh metadata buffers in place: im_shape=[800,800],
  //    scale_factor=[800/h, 800/w]. The cached tensors view these arrays.
  im.im_shape = {static_cast<float>(kInputSize),
                 static_cast<float>(kInputSize)};
  im.scale_factor = {
      static_cast<float>(kInputSize) / static_cast<float>(orig_h),
      static_cast<float>(kInputSize) / static_cast<float>(orig_w)};

  // 3. Run inference on the pre-built input tensors / name tables.
  std::vector<Ort::Value> outputs;
  try {
    outputs = im.session->Run(
        Ort::RunOptions{nullptr},
        im.input_names.data(), im.input_tensors.data(), im.input_tensors.size(),
        im.output_names.data(), im.output_names.size());
  } catch (const Ort::Exception &e) {
    std::cerr << "[cpu_layout] Inference failed: " << e.what() << '\n';
    return out;
  }

  // 4. Find the (N, 7) detection output.
  const float *dets = nullptr;
  int n_rows = 0;
  for (auto &output : outputs) {
    const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() == 2 && shape[1] == 7) {
      dets = output.GetTensorData<float>();
      n_rows = static_cast<int>(shape[0]);
      break;
    }
  }
  if (!dets) {
    // No (N,7) tensor among the outputs = unexpected model output, not a
    // genuinely empty page. Fail loud rather than return an empty layout that
    // reads as "clean" downstream (no-silent-failure).
    std::cerr << "[cpu_layout] model produced no (N,7) detection output; "
                 "returning empty layout — check the layout model\n";
    return out;
  }
  if (n_rows <= 0) return out; // genuinely zero detections — fine
  n_rows = std::min(n_rows, kMaxDetections);

  // 5. Decode detections.
  out.reserve(static_cast<size_t>(n_rows));
  for (int i = 0; i < n_rows; ++i) {
    const float *row = dets + i * 7;
    const float score = row[1];
    if (score < score_threshold) continue;

    const int cls = static_cast<int>(row[0]);
    if (cls < 0 || cls >= static_cast<int>(kLayoutLabels.size())) continue;

    const int x0 = std::clamp(static_cast<int>(row[2]), 0, orig_w - 1);
    const int y0 = std::clamp(static_cast<int>(row[3]), 0, orig_h - 1);
    const int x1 = std::clamp(static_cast<int>(row[4]), 0, orig_w - 1);
    const int y1 = std::clamp(static_cast<int>(row[5]), 0, orig_h - 1);
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

  // 6. Shared NMS + oversized-image drop + LAYOUT_MERGE_MODE reconciliation.
  return postfilter_layout_boxes(std::move(out), orig_h, orig_w);
}

} // namespace turbo_ocr::layout
