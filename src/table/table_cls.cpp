#include "turbo_ocr/table/table_cls.h"

#include <algorithm>
#include <iostream>

#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/kernels/kernels.h"

namespace turbo_ocr::table {

using turbo_ocr::engine::TrtEngine;

bool TableCls::load_model(const std::string& trt_path) {
  engine_ = std::make_unique<TrtEngine>(trt_path);
  if (!engine_->load()) {
    std::cerr << "[table_cls] failed to load TRT engine: " << trt_path << '\n';
    return false;
  }
  return true;
}

void TableCls::allocate_buffers() {
  if (buffers_allocated_) return;

  const std::size_t in_elems  = static_cast<std::size_t>(kMaxBatch) * 3 * kInputSize * kInputSize;
  const std::size_t out_elems = static_cast<std::size_t>(kMaxBatch) * 2;
  d_input_  = CudaPtr<float>(in_elems);
  d_output_ = CudaPtr<float>(out_elems);
  h_output_ = CudaHostPtr<float>(out_elems);

  // Single-IO model: legacy bind_io path is sufficient.
  engine_->bind_io(d_input_.get(), d_output_.get());
  buffers_allocated_ = true;
}

std::vector<TableVariant>
TableCls::classify_batch(const GpuImage& page,
                         const std::vector<Box>& regions,
                         cudaStream_t stream) {
  std::vector<TableVariant> out;
  if (regions.empty()) return out;
  if (!engine_) return out;

  allocate_buffers();

  out.resize(regions.size(), TableVariant::Wired);
  const int total = static_cast<int>(regions.size());

  for (int beg = 0; beg < total; beg += kMaxBatch) {
    const int cur = std::min(kMaxBatch, total - beg);
    const std::size_t per_image = static_cast<std::size_t>(3) * kInputSize * kInputSize;

    for (int j = 0; j < cur; ++j) {
      auto bb = aabb(regions[beg + j]);
      const int rx = std::clamp(bb[0], 0, page.cols - 1);
      const int ry = std::clamp(bb[1], 0, page.rows - 1);
      const int rw = std::clamp(bb[2] - rx, 1, page.cols - rx);
      const int rh = std::clamp(bb[3] - ry, 1, page.rows - ry);
      kernels::cuda_fused_table_cls_pre(
          page, rx, ry, rw, rh,
          d_input_.get() + j * per_image,
          stream);
    }

    nvinfer1::Dims4 dims{cur, 3, kInputSize, kInputSize};
    if (!engine_->infer_dynamic(dims, stream)) {
      throw turbo_ocr::InferenceError("TableCls TRT inference failed");
    }

    CUDA_CHECK(cudaMemcpyAsync(h_output_.get(), d_output_.get(),
                                static_cast<std::size_t>(cur) * 2 * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int j = 0; j < cur; ++j) {
      const float s0 = h_output_.get()[j * 2 + 0];
      const float s1 = h_output_.get()[j * 2 + 1];
      out[beg + j] = (s1 > s0) ? TableVariant::Wireless : TableVariant::Wired;
    }
  }

  return out;
}

} // namespace turbo_ocr::table
