#pragma once

#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_ptr.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt_engine.h"
#include "turbo_ocr/table/nemotron_postprocess.h"

namespace turbo_ocr::table {

// NVIDIA Nemotron Table Structure v1 (YOLOX, Oct 2025) — fixed input
// 1×3×1024×1024 letterbox, output preds[1, 21504, 10]. NMS happens in
// nemotron_postprocess.cpp; this class just owns the TRT engine, the
// pinned/device buffers, and the host-side decode call.
class NemotronTableStruct {
 public:
  static constexpr int kInputSize = kNemotronCanvas;
  static constexpr int kNumQueries = kNemotronNumQueries;
  static constexpr int kStride = kNemotronStride;

  NemotronTableStruct() = default;
  ~NemotronTableStruct() noexcept;

  NemotronTableStruct(const NemotronTableStruct&) = delete;
  NemotronTableStruct& operator=(const NemotronTableStruct&) = delete;

  [[nodiscard]] bool load_model(const std::string& trt_path);

  // Run the engine on a sub-rect, blocking on the stream for D2H. Returns
  // a decoded {cells, rows, columns} triplet already in original-region
  // pixel coordinates (no further scaling needed).
  [[nodiscard]] NemotronDecode infer(const GpuImage& page,
                                      const Box& region,
                                      cudaStream_t stream);

 private:
  [[nodiscard]] bool discover_tensor_names();
  [[nodiscard]] bool init_buffers();

  std::unique_ptr<engine::TrtEngine> engine_;

  cudaEvent_t d2h_event_ = nullptr;

  CudaPtr<float>     d_image_;
  CudaPtr<float>     d_preds_;
  CudaHostPtr<float> h_preds_;

  std::string input_name_;
  std::string output_name_;
};

}  // namespace turbo_ocr::table
