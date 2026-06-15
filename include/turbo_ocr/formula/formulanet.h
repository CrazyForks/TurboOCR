#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_ptr.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt_engine.h"
#include "turbo_ocr/formula/formula_kernels.h"
#include "turbo_ocr/formula/formula_tokenizer.h"

namespace turbo_ocr::formula {

// Engine-level result emitted per input box (aligned with the input order).
// Distinct from `router::FormulaResult` (which carries layout_id + box and is
// the pipeline-output shape). The orchestrator maps engine results to router
// results when it assembles the page result.
struct FormulaEngineResult {
  std::string latex;
  std::size_t token_count = 0;
  bool        hit_eos = false;
};

// Plan 03 §5.2 + §11: sub-batch crops at kFormulaBatchMax for the encoder /
// decoder Loop. Anything larger is chunked internally.
inline constexpr int kFormulaBatchMax = 4;

// Plan 03 §4.2: data-dependent output buffer is sized for B_max * (1024 + K + 8).
//   1024 : worst-case content length emitted by the MTP loop.
//   K=3  : MTP head width.
//   +8   : BOS prefix + EOS + a few specials.
inline constexpr int kFormulaMaxOutputTokens = 1024 + 3 + 8;

// PP-FormulaNet-S TRT engine wrapper.
//   - Single dynamic-batch profile (B=1..8 from onnx_to_trt.cpp::"formula").
//   - Fused MTP K=3 decoder lives inside the engine's `Loop` op (plan 03 §3).
//   - One execute + one mandatory cudaStreamSynchronize per sub-batch:
//     the Loop's output `[B, T]` has runtime-resolved `T` so we must sync
//     before reading the shape.
class FormulaNet {
public:
  FormulaNet();
  ~FormulaNet() noexcept = default;

  FormulaNet(const FormulaNet &) = delete;
  FormulaNet &operator=(const FormulaNet &) = delete;
  FormulaNet(FormulaNet &&) = default;
  FormulaNet &operator=(FormulaNet &&) = default;

  // Build (or reuse from cache) the TRT engine for the given ONNX file, then
  // load it. Internally calls `engine::ensure_trt_engine(onnx_path, "formula")`.
  [[nodiscard]] bool load_model(const std::string &onnx_path);

  // Load the byte-level BPE tokenizer JSON used to decode the output token
  // stream into LaTeX. Returns false if the file is missing or malformed.
  [[nodiscard]] bool load_tokenizer(const std::string &tokenizer_json_path);

  // Run the engine over the formula crops described by `boxes` (sub-rects of
  // `page`). Returns one `FormulaEngineResult` per input box, aligned with
  // `boxes`. Batches up to `kFormulaBatchMax` crops per engine execute and
  // chunks if more are given.
  //
  // `stream` is reused for H2D, kernels, engine execute, and D2H. The method
  // blocks on a `cudaStreamSynchronize` after each sub-batch (mandatory).
  [[nodiscard]] std::vector<FormulaEngineResult>
  run(const GpuImage &page, const std::vector<Box> &boxes, cudaStream_t stream);

  // Exposed for tests / orchestrator wiring.
  [[nodiscard]] bool is_ready() const noexcept {
    return engine_ != nullptr && tokenizer_ != nullptr && buffers_allocated_;
  }

private:
  bool discover_tensor_names();
  bool allocate_buffers();
  // One sub-batch (size <= kFormulaBatchMax). Appends results in input order
  // to `out` starting at `out_offset`.
  bool run_sub_batch(const GpuImage &page,
                     const std::vector<Box> &boxes,
                     int begin, int end,
                     cudaStream_t stream,
                     std::vector<FormulaEngineResult> &out);

  std::unique_ptr<engine::TrtEngine>  engine_;
  std::unique_ptr<FormulaTokenizer>   tokenizer_;

  // I/O tensor names (plan 03 §3: input `x`, output `fetch_name_0`).
  std::string input_name_;
  std::string output_name_;

  // Device buffers (all sized for kFormulaBatchMax).
  CudaPtr<kernels::FormulaSubRect> d_subrects_;
  CudaPtr<int32_t>                 d_bbox_;     // per-crop crop_margin bbox scratch
  CudaPtr<float>                   d_input_;    // [B, 1, 384, 384]
  CudaPtr<int64_t>                 d_output_;   // [B, kFormulaMaxOutputTokens]
  // Pinned host mirrors for low-latency staging / readback.
  CudaHostPtr<kernels::FormulaSubRect> h_subrects_;
  CudaHostPtr<int64_t>                 h_output_;

  bool buffers_allocated_ = false;
};

} // namespace turbo_ocr::formula
