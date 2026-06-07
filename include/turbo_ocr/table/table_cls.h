#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_ptr.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt_engine.h"
#include "turbo_ocr/table/table_types.h"

namespace turbo_ocr::table {

// PP-LCNet_x1_0_table_cls — predicts wired vs wireless variant for each
// detected table region. Input (N, 3, 224, 224), output (N, 2). Profile
// MIN/OPT/MAX = 1/4/8. Preprocess uses cuda_fused_table_cls_pre (sub-rect of
// page GpuImage; resize-short(256) + center-crop(224) + ImageNet + CHW).
class TableCls {
public:
  static constexpr int kInputSize = 224;
  static constexpr int kMaxBatch  = 8;

  TableCls() = default;
  ~TableCls() noexcept = default;

  TableCls(const TableCls&) = delete;
  TableCls& operator=(const TableCls&) = delete;

  // Load a serialized TRT engine (built from PP-LCNet_x1_0_table_cls).
  [[nodiscard]] bool load_model(const std::string& trt_path);

  // Classify N regions on the page. The page GpuImage is sub-rect-sliced on
  // GPU; no host crop or extra H2D. Returns one variant per input box.
  // Synchronizes `stream` once at the end for the D2H readback.
  [[nodiscard]] std::vector<TableVariant>
  classify_batch(const GpuImage& page,
                 const std::vector<Box>& regions,
                 cudaStream_t stream);

private:
  void allocate_buffers();

  std::unique_ptr<engine::TrtEngine> engine_;

  CudaPtr<float> d_input_;   // [kMaxBatch, 3, 224, 224]
  CudaPtr<float> d_output_;  // [kMaxBatch, 2]
  CudaHostPtr<float> h_output_;
  bool buffers_allocated_ = false;
};

} // namespace turbo_ocr::table
