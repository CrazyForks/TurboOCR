#include "turbo_ocr/table/slanext.h"

#include <algorithm>
#include <iostream>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/table/slanext_postprocess.h"

namespace turbo_ocr::table {

using turbo_ocr::engine::TrtEngine;

SlaNeXt::~SlaNeXt() noexcept {
  if (d2h_event_) cudaEventDestroy(d2h_event_);
}

bool SlaNeXt::discover_tensor_names() {
  // Single input — (1, 3, 488, 488).
  const auto& ins = engine_->input_names();
  if (ins.empty()) {
    std::cerr << "[slanext] no input tensors\n";
    return false;
  }
  name_image_ = ins.front();

  // Two outputs: structure_probs (last dim == V=50) and loc_preds (total
  // elements per token == 8; either trailing dim 8 or (4, 2)).
  const auto& outs = engine_->output_names();
  if (outs.size() < 2) {
    std::cerr << "[slanext] expected ≥2 outputs, got " << outs.size() << '\n';
    return false;
  }
  for (const auto& n : outs) {
    auto d = engine_->tensor_shape(n);
    if (d.nbDims == 0) continue;
    const int last = d.d[d.nbDims - 1];
    if (last == kVocab) {
      name_probs_ = n;
    } else {
      // 8 (trailing-8 layout) or 2 (trailing-(4,2) layout) — either is the
      // loc tensor. Total per token element count = 8 in both cases.
      if (name_loc_.empty()) name_loc_ = n;
    }
  }
  if (name_probs_.empty() || name_loc_.empty()) {
    std::cerr << "[slanext] could not classify outputs; got: ";
    for (const auto& n : outs) std::cerr << n << " ";
    std::cerr << '\n';
    return false;
  }
  return true;
}

bool SlaNeXt::init_buffers() {
  d_image_.reset(static_cast<std::size_t>(1) * 3 * kInputSize * kInputSize);
  d_loc_.reset(static_cast<std::size_t>(1) * kMaxTokens * 8);
  d_probs_.reset(static_cast<std::size_t>(1) * kMaxTokens * kVocab);

  h_loc_.reset(static_cast<std::size_t>(1) * kMaxTokens * 8);
  h_probs_.reset(static_cast<std::size_t>(1) * kMaxTokens * kVocab);

  engine_->set_tensor_address(name_image_, d_image_.get());
  engine_->set_tensor_address(name_loc_,   d_loc_.get());
  engine_->set_tensor_address(name_probs_, d_probs_.get());

  CUDA_CHECK(cudaEventCreateWithFlags(&d2h_event_, cudaEventDisableTiming));

  if (!has_dict_) {
    dict_ = default_dict();
    has_dict_ = true;
  }
  return true;
}

bool SlaNeXt::load_model(const std::string& trt_path) {
  engine_ = std::make_unique<TrtEngine>(trt_path);
  if (!engine_->load()) {
    std::cerr << "[slanext] failed to load TRT engine: " << trt_path << '\n';
    return false;
  }
  if (!discover_tensor_names()) return false;
  if (!init_buffers()) return false;
  return true;
}

bool SlaNeXt::enqueue(const GpuImage& page,
                     int rect_x, int rect_y,
                     int rect_w, int rect_h,
                     cudaStream_t stream) {
  if (!engine_) return false;
  if (rect_w <= 0 || rect_h <= 0) return false;

  rect_x = std::clamp(rect_x, 0, page.cols - 1);
  rect_y = std::clamp(rect_y, 0, page.rows - 1);
  rect_w = std::clamp(rect_w, 1, page.cols - rect_x);
  rect_h = std::clamp(rect_h, 1, page.rows - rect_y);

  pending_rect_w_ = rect_w;
  pending_rect_h_ = rect_h;

  // Fused preprocess: sub-rect → ResizeByLong(488) + ImageNet + bottom-right
  // pad → BGR CHW. Writes directly into d_image_.
  kernels::cuda_fused_slanext_pre(
      page, rect_x, rect_y, rect_w, rect_h, d_image_.get(), stream);

  // Single-input — fixed shape (1, 3, 488, 488).
  nvinfer1::Dims4 img_dims{1, 3, kInputSize, kInputSize};
  if (!engine_->set_input_shape(name_image_, img_dims)) return false;
  if (!engine_->execute(stream)) {
    std::cerr << "[slanext] TRT execute failed\n";
    return false;
  }

  pending_stream_ = stream;
  CUDA_CHECK(cudaEventRecord(d2h_event_, stream));
  return true;
}

StructureResult SlaNeXt::collect() {
  StructureResult empty{};
  if (!engine_ || !d2h_event_) return empty;

  CUDA_CHECK(cudaEventSynchronize(d2h_event_));

  // T is fixed for the bundled-AR unroll. The CPU walk early-terminates at
  // the first <eos>. Some exports may produce <T=501; query to be safe.
  auto loc_dims   = engine_->tensor_shape(name_loc_);
  auto probs_dims = engine_->tensor_shape(name_probs_);
  int t = 0;
  if (probs_dims.nbDims >= 2) t = probs_dims.d[probs_dims.nbDims - 2];
  if (t <= 0) t = kMaxTokens;
  if (t > kMaxTokens) t = kMaxTokens;
  (void)loc_dims;

  const std::size_t loc_elems   = static_cast<std::size_t>(t) * 8;
  const std::size_t probs_elems = static_cast<std::size_t>(t) * kVocab;

  CUDA_CHECK(cudaMemcpyAsync(h_loc_.get(),   d_loc_.get(),
                              sizeof(float) * loc_elems,
                              cudaMemcpyDeviceToHost, pending_stream_));
  CUDA_CHECK(cudaMemcpyAsync(h_probs_.get(), d_probs_.get(),
                              sizeof(float) * probs_elems,
                              cudaMemcpyDeviceToHost, pending_stream_));
  CUDA_CHECK(cudaStreamSynchronize(pending_stream_));

  return decode_structure(h_probs_.get(), h_loc_.get(),
                          static_cast<std::size_t>(t),
                          static_cast<std::size_t>(kVocab),
                          dict_,
                          kInputSize, kInputSize,
                          pending_rect_w_, pending_rect_h_);
}

StructureResult SlaNeXt::infer(const GpuImage& page,
                              const Box& region,
                              cudaStream_t stream) {
  auto bb = aabb(region);
  const int rx = bb[0];
  const int ry = bb[1];
  const int rw = bb[2] - rx;
  const int rh = bb[3] - ry;
  if (!enqueue(page, rx, ry, rw, rh, stream)) return StructureResult{};
  return collect();
}

} // namespace turbo_ocr::table
