#include "turbo_ocr/table/nemotron_table_struct.h"

#include <algorithm>
#include <iostream>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/kernels/kernels.h"

namespace turbo_ocr::table {

using engine::TrtEngine;

NemotronTableStruct::~NemotronTableStruct() noexcept {
  if (d2h_event_) cudaEventDestroy(d2h_event_);
}

bool NemotronTableStruct::discover_tensor_names() {
  const auto& ins = engine_->input_names();
  if (ins.empty()) {
    std::cerr << "[nemotron] no input tensors\n";
    return false;
  }
  input_name_ = ins.front();
  const auto& outs = engine_->output_names();
  if (outs.empty()) {
    std::cerr << "[nemotron] no output tensors\n";
    return false;
  }
  output_name_ = outs.front();
  return true;
}

bool NemotronTableStruct::init_buffers() {
  d_image_.reset(static_cast<std::size_t>(1) * 3 * kInputSize * kInputSize);
  d_preds_.reset(static_cast<std::size_t>(1) * kNumQueries * kStride);
  h_preds_.reset(static_cast<std::size_t>(1) * kNumQueries * kStride);

  engine_->set_tensor_address(input_name_,  d_image_.get());
  engine_->set_tensor_address(output_name_, d_preds_.get());

  CUDA_CHECK(cudaEventCreateWithFlags(&d2h_event_, cudaEventDisableTiming));
  return true;
}

bool NemotronTableStruct::load_model(const std::string& trt_path) {
  engine_ = std::make_unique<TrtEngine>(trt_path);
  if (!engine_->load()) {
    std::cerr << "[nemotron] failed to load TRT engine: " << trt_path << '\n';
    return false;
  }
  if (!discover_tensor_names()) return false;
  if (!init_buffers()) return false;
  return true;
}

NemotronDecode NemotronTableStruct::infer(const GpuImage& page,
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

  kernels::cuda_fused_nemotron_pre(
      page, rx, ry, rw, rh, d_image_.get(), stream);

  nvinfer1::Dims4 in_dims{1, 3, kInputSize, kInputSize};
  if (!engine_->set_input_shape(input_name_, in_dims)) return empty;
  if (!engine_->execute(stream)) {
    std::cerr << "[nemotron] TRT execute failed\n";
    return empty;
  }

  CUDA_CHECK(cudaMemcpyAsync(h_preds_.get(), d_preds_.get(),
                              sizeof(float) * kNumQueries * kStride,
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(d2h_event_, stream));
  CUDA_CHECK(cudaEventSynchronize(d2h_event_));

  return decode_nemotron(h_preds_.get(), kNumQueries, rw, rh);
}

}  // namespace turbo_ocr::table
