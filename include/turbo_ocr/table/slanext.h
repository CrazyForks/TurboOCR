#pragma once

#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_ptr.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt_engine.h"
#include "turbo_ocr/table/slanext_dict.h"
#include "turbo_ocr/table/table_types.h"

namespace turbo_ocr::table {

// SLANeXt structure decoder. Single TRT execute on (1, 3, 488, 488) emits two
// outputs from a bundled-AR static unroll (T = 501):
//
//   loc_preds       — (T, 8)   or (T, 4, 2)   normalized quad corners
//   structure_probs — (T, V=50)               softmax over HTML token vocab
//
// CPU postprocess (decode_structure) walks the T rows, argmaxes vocab,
// terminates at first <eos>, and emits structure tokens + td quads.
//
// One instance per variant (wired / wireless). The umbrella owns two.
class SlaNeXt {
public:
  static constexpr int kInputSize  = 488;
  static constexpr int kMaxTokens  = 501;
  static constexpr int kVocab      = static_cast<int>(SLANEXT_VOCAB);  // 50

  SlaNeXt() = default;
  ~SlaNeXt() noexcept;

  SlaNeXt(const SlaNeXt&) = delete;
  SlaNeXt& operator=(const SlaNeXt&) = delete;

  // Load TRT engine. `dict` lets callers swap the bundled dict (default if
  // empty path); otherwise reads via default_dict().
  [[nodiscard]] bool load_model(const std::string& trt_path);
  void set_dict(CharDict dict) { dict_ = std::move(dict); has_dict_ = true; }

  // Enqueue preprocess + TRT execute on `stream`. Records an event so
  // collect() can wait on it without stalling the calling thread.
  // Sub-rect is in PAGE pixels.
  [[nodiscard]] bool enqueue(const GpuImage& page,
                             int rect_x, int rect_y,
                             int rect_w, int rect_h,
                             cudaStream_t stream);

  // Block on the recorded event, D2H both outputs, run decode_structure.
  [[nodiscard]] StructureResult collect();

  // Convenience: enqueue + sync + collect. Returns directly.
  [[nodiscard]] StructureResult infer(const GpuImage& page,
                                      const Box& region,
                                      cudaStream_t stream);

private:
  [[nodiscard]] bool discover_tensor_names();
  [[nodiscard]] bool init_buffers();

  std::unique_ptr<engine::TrtEngine> engine_;

  cudaEvent_t d2h_event_ = nullptr;
  cudaStream_t pending_stream_ = nullptr;
  int pending_rect_w_ = 0;
  int pending_rect_h_ = 0;

  CudaPtr<float> d_image_;     // [1, 3, 488, 488]
  CudaPtr<float> d_loc_;       // [1, T, 8] (or T*4*2 — same total)
  CudaPtr<float> d_probs_;     // [1, T, V]
  CudaHostPtr<float> h_loc_;
  CudaHostPtr<float> h_probs_;

  std::string name_image_;
  std::string name_loc_;       // loc_preds tensor
  std::string name_probs_;     // structure_probs tensor

  CharDict dict_;
  bool has_dict_ = false;
};

} // namespace turbo_ocr::table
