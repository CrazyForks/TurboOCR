#include "turbo_ocr/table/tatr_table_struct.h"

#include <algorithm>
#include <iostream>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/kernels/kernels.h"

namespace turbo_ocr::table {

using engine::TrtEngine;

TatrTableStruct::~TatrTableStruct() noexcept {
  if (d2h_event_) cudaEventDestroy(d2h_event_);
}

bool TatrTableStruct::discover_tensor_names() {
  const auto& ins = engine_->input_names();
  if (ins.empty()) {
    std::cerr << "[tatr] no input tensors\n";
    return false;
  }
  input_name_ = ins.front();

  const auto& outs = engine_->output_names();
  if (outs.size() < 2) {
    std::cerr << "[tatr] expected 2 output tensors, got " << outs.size() << '\n';
    return false;
  }
  // Resolve by name — ONNX exporters do not guarantee output order. TATR
  // names them "logits" and "pred_boxes" per the HF DetrForObjectDetection
  // head; fall back to size-based heuristic if names diverge.
  logits_name_.clear();
  boxes_name_.clear();
  for (const auto& n : outs) {
    if (n.find("logit") != std::string::npos) logits_name_ = n;
    else if (n.find("box") != std::string::npos) boxes_name_ = n;
  }
  if (logits_name_.empty() || boxes_name_.empty()) {
    logits_name_ = outs[0];
    boxes_name_  = outs[1];
    std::cerr << "[tatr] could not resolve outputs by name; defaulted to "
              << logits_name_ << " + " << boxes_name_ << '\n';
  }
  return true;
}

bool TatrTableStruct::init_buffers() {
  d_image_.reset(static_cast<std::size_t>(1) * 3 * kInputSize * kInputSize);
  d_logits_.reset(static_cast<std::size_t>(1) * kNumQueries * kNumClasses);
  d_boxes_.reset(static_cast<std::size_t>(1) * kNumQueries * 4);
  h_logits_.reset(static_cast<std::size_t>(1) * kNumQueries * kNumClasses);
  h_boxes_.reset(static_cast<std::size_t>(1) * kNumQueries * 4);

  engine_->set_tensor_address(input_name_,  d_image_.get());
  engine_->set_tensor_address(logits_name_, d_logits_.get());
  engine_->set_tensor_address(boxes_name_,  d_boxes_.get());

  CUDA_CHECK(cudaEventCreateWithFlags(&d2h_event_, cudaEventDisableTiming));
  return true;
}

bool TatrTableStruct::load_model(const std::string& trt_path) {
  engine_ = std::make_unique<TrtEngine>(trt_path);
  if (!engine_->load()) {
    std::cerr << "[tatr] failed to load TRT engine: " << trt_path << '\n';
    return false;
  }
  if (!discover_tensor_names()) return false;
  if (!init_buffers()) return false;
  return true;
}

NemotronDecode TatrTableStruct::infer(const GpuImage& page,
                                       const Box& region,
                                       cudaStream_t stream) {
  NemotronDecode empty{};
  if (!engine_) return empty;

  auto bb = aabb(region);
  const int rx = std::clamp(bb[0], 0, page.cols - 1);
  const int ry = std::clamp(bb[1], 0, page.rows - 1);
  const int rw = std::clamp(bb[2] - rx, 1, page.cols - rx);
  const int rh = std::clamp(bb[3] - ry, 1, page.rows - ry);
  if (rw <= 0 || rh <= 0) return empty;

  // Compute letterbox-scaled dims so postprocess can undo the padding.
  const float long_side = static_cast<float>(std::max(rw, rh));
  const float s = static_cast<float>(kInputSize) / long_side;
  int new_w = static_cast<int>(std::lround(static_cast<float>(rw) * s));
  int new_h = static_cast<int>(std::lround(static_cast<float>(rh) * s));
  new_w = std::clamp(new_w, 1, kInputSize);
  new_h = std::clamp(new_h, 1, kInputSize);

  kernels::cuda_fused_tatr_pre(
      page, rx, ry, rw, rh, d_image_.get(), stream);

  nvinfer1::Dims4 in_dims{1, 3, kInputSize, kInputSize};
  if (!engine_->set_input_shape(input_name_, in_dims)) return empty;
  if (!engine_->execute(stream)) {
    std::cerr << "[tatr] TRT execute failed\n";
    return empty;
  }

  CUDA_CHECK(cudaMemcpyAsync(h_logits_.get(), d_logits_.get(),
                              sizeof(float) * kNumQueries * kNumClasses,
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_boxes_.get(), d_boxes_.get(),
                              sizeof(float) * kNumQueries * 4,
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(d2h_event_, stream));
  CUDA_CHECK(cudaEventSynchronize(d2h_event_));

  return decode_tatr(h_logits_.get(), h_boxes_.get(),
                     kNumQueries, kInputSize, new_w, new_h, rw, rh);
}

}  // namespace turbo_ocr::table
