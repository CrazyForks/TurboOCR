#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/formula/formula_recognizer.h"
#include "turbo_ocr/formula/formula_tokenizer.h"
#include "turbo_ocr/formula/ort_session.h"

namespace turbo_ocr::formula {

// Pure in-process PP-FormulaNet-S recognizer running ONNX Runtime (no TensorRT
// engines, no Python sidecar, no socket/GIL). It loads .onnx models only.
//
// Three modes, selected at load:
//   * GPU FAST (default): ORT-CUDA-13 encoder.onnx -> cross-KV prep.onnx -> a
//     static-KV single-step decoder (step_batched.onnx, 1056-token KV buffer)
//     driven by a host AR loop with on-GPU argmax + KV ping-pong. Matches the
//     fused reference EXACTLY (CDM 0.811) and is ~8x faster than the fused Loop.
//   * GPU EXACT (PPFNS_EXACT=1): the fused graph inference_trt.onnx (encoder +
//     growable-KV AR Loop) run batched on ORT-CUDA-13. The literal reference, kept
//     as a fallback; slower (Loop-bound).
//   * CPU (FORMULA_DEVICE=cpu): the fused graph on ORT's CPUExecutionProvider with
//     host tensors only -- no CUDA decode buffers, no kernels, no Python. (The FAST
//     host loop needs CUDA kernels, so CPU uses the fused graph.)
//
// FAST/EXACT models live in <model_parent>/fast/ (encoder.onnx, prep.onnx,
// step_batched.onnx) and <model_parent>/inference_trt.onnx (the fused graph).
// The fast/ split graphs are locally generated (scratchpad/fastdec/batched_step.py)
// and may be absent from a fresh deploy: if FAST is selected but any fast/ file is
// missing/unloadable, the loader falls back to the fused graph (fast_=false) with a
// LOUD warning rather than aborting boot. See docs/models/formula.md.
class PPFormulaNetOrt final : public IFormulaRecognizer {
public:
  PPFormulaNetOrt();
  ~PPFormulaNetOrt() noexcept override;

  [[nodiscard]] bool load_model_dir(const std::string &model_dir) override;
  [[nodiscard]] bool load_tokenizer(const std::string &path) override;

  [[nodiscard]] std::vector<FormulaEngineResult>
  run(const GpuImage &page, const std::vector<Box> &boxes,
      cudaStream_t stream) override;

  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return "ppformulanet_s";
  }

private:
  bool alloc_buffers();
  void free_buffers() noexcept;
  // GPU FAST path: lockstep-batched static-KV step host-loop (encoder+prep+step).
  // Returns false when a step/argmax/CUDA error truncated the decode.
  bool decode_chunk(int B, std::vector<std::vector<int64_t>> &out);

  OrtSession fused_;                 // EXACT (GPU) + CPU path
  OrtSession enc_, prep_, step_;     // GPU FAST path
  bool fast_ = false;                // GPU host-loop (default on GPU)
  bool cpu_ = false;                 // ORT CPUExecutionProvider (FORMULA_DEVICE=cpu)
  std::optional<FormulaTokenizer> tok_;
  bool ready_ = false;
  std::string fast_dir_;
  cudaStream_t stream_ = nullptr;    // nullptr on the CPU path

  float *d_x_ = nullptr;             // [MAX_B,1,384,384] device crops (GPU paths only)
  // FAST-path device buffers (allocated only when fast_).
  float *d_mem_ = nullptr, *d_ck_ = nullptr, *d_cv_ = nullptr, *d_log_ = nullptr;
  float *kA_ = nullptr, *kB_ = nullptr, *vA_ = nullptr, *vB_ = nullptr;
  int64_t *d_tok_ = nullptr, *d_next_ = nullptr, *d_pos_ = nullptr, *d_all_ = nullptr;
  std::vector<uint8_t> host_page_;   // page D2H scratch
  std::vector<float> host_in_;       // [MAX_B,1,384,384] preprocessed crops
};

}  // namespace turbo_ocr::formula
