#pragma once

#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_ptr.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt_engine.h"
#include "turbo_ocr/table/tatr_postprocess.h"

namespace turbo_ocr::table {

// Microsoft Table Transformer (DETR-family) backend. Fixed 800×800
// letterbox input, head returns (logits[125,7], pred_boxes[125,4]).
// Postprocess + grid assembly reuse NemotronAssembly via decode_tatr's
// class-id remap.
class TatrTableStruct {
 public:
  static constexpr int kInputSize  = kTatrCanvas;
  static constexpr int kNumQueries = kTatrNumQueries;
  static constexpr int kNumClasses = kTatrNumClasses;

  TatrTableStruct() = default;
  ~TatrTableStruct() noexcept;

  TatrTableStruct(const TatrTableStruct&) = delete;
  TatrTableStruct& operator=(const TatrTableStruct&) = delete;

  [[nodiscard]] bool load_model(const std::string& trt_path);

  // Run the engine on a sub-rect, blocking on the stream for D2H. Returns a
  // NemotronDecode (rows/cols/cells/headers) in original-region pixels.
  [[nodiscard]] NemotronDecode infer(const GpuImage& page,
                                      const Box& region,
                                      cudaStream_t stream);

 private:
  [[nodiscard]] bool discover_tensor_names();
  [[nodiscard]] bool init_buffers();

  std::unique_ptr<engine::TrtEngine> engine_;

  cudaEvent_t d2h_event_ = nullptr;

  CudaPtr<float>     d_image_;
  CudaPtr<float>     d_logits_;
  CudaPtr<float>     d_boxes_;
  CudaHostPtr<float> h_logits_;
  CudaHostPtr<float> h_boxes_;

  std::string input_name_;
  std::string logits_name_;
  std::string boxes_name_;
};

}  // namespace turbo_ocr::table
