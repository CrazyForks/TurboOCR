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
  // backend_label names the engine in logs/routing; force_fused makes load_model_dir
  // always take the fused EXACT path (no FAST host-loop). Defaults reproduce the
  // original PP-FormulaNet-S behavior exactly. PP-FormulaNet_plus-M reuses this class
  // via ("ppformulanet_plus_m", force_fused=FALSE): it ships its OWN split graphs
  // (encoder.onnx + prep.onnx + decoder_step.onnx in the model dir) and runs the plus-M
  // 6-layer MBart FAST host-loop (decode_chunk_plusm / decode_continuous_plusm). NOTE:
  // force_fused=true would set fast_=false -> plusm_=false and disable the plus-M fast
  // path entirely, so plus-M MUST be constructed with force_fused=false.
  explicit PPFormulaNetOrt(std::string backend_label = "ppformulanet_s",
                           bool force_fused = false);
  ~PPFormulaNetOrt() noexcept override;

  [[nodiscard]] bool load_model_dir(const std::string &model_dir) override;
  [[nodiscard]] bool load_tokenizer(const std::string &path) override;

  [[nodiscard]] std::vector<FormulaEngineResult>
  run(const GpuImage &page, const std::vector<Box> &boxes,
      cudaStream_t stream) override;

  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return label_;
  }

  // Dev/CI harness (NOT used by the server): decode the 30 gate crops in <gate_dir>
  // (en15_crops.bin + zh15_crops.bin, preprocessed [15,1,384,384] each) through this
  // backend's FAST host loop and write /tmp/cpp_<label>_tokens.json (lockstep), the
  // continuous tokens (plus-M), and /tmp/cpp_<label>_latex.json, plus print lockstep
  // crops/s. Driven only by tools/plusm_selftest.cpp so no bench code lives in the
  // production load/run path. Returns false if the gate crops or model are unavailable.
  [[nodiscard]] bool gate_bench(const std::string &gate_dir);

private:
  bool alloc_buffers();
  void free_buffers() noexcept;
  // GPU FAST path: lockstep-batched static-KV step host-loop (encoder+prep+step).
  // Returns false when a step/argmax/CUDA error truncated the decode.
  bool decode_chunk(int B, std::vector<std::vector<int64_t>> &out);
  // PP-FormulaNet_plus-M FAST path: same host-loop shape as decode_chunk, but the
  // plus-M MBart decoder is 6-layer / Dh=32 / single-token (pos[B] per-seq) and the
  // step graph emits next_token in-graph (no argmax kernel). Reads encoder.onnx +
  // prep.onnx + decoder_step.onnx from the model dir (not a fast/ subdir).
  bool decode_chunk_plusm(int B, std::vector<std::vector<int64_t>> &out);
  // PP-FormulaNet_plus-M continuous (iteration-level) batched decode: Bslots slots
  // process a queue of N crops whose cross-KV is pre-computed in ck_all_/cv_all_. A
  // slot that emits EOS is evicted (its sequence saved to out[crop]) and refilled with
  // the next queued crop (cross-KV copied into the slot, pos->0, token->START); the
  // static-KV step overwrites each KV slot before reading it, so no per-slot KV reset
  // is needed. Keeps the batch full -> removes the lockstep long-tail stall.
  // `step`/`kA..vB`/`maxlen` select the KV bucket (the small 384-window step_short_ for the
  // common case, or the 1056 step_ for the long tail). `queue` is the crop indices (into
  // ck_all_, of which n_total were prepped) to decode; out[crop] gets each finished sequence,
  // and a crop that hits `maxlen` tokens without EOS is appended to `overflow` (its partial
  // out[] cleared) for the caller to re-decode in a larger bucket.
  bool decode_continuous_plusm(OrtSession &step, float *kA, float *kB, float *vA, float *vB,
                               int maxlen, int Bslots, int n_total,
                               const std::vector<int> &queue,
                               std::vector<std::vector<int64_t>> &out,
                               std::vector<int> &overflow, bool final_bucket);
  // Production plus-M path: preprocess+encode ALL N formula crops of a page into the
  // all-crop buffers (in MAX_B-sized preprocessing batches, then batched cross-KV prep),
  // and decode them with the continuous host loop (Bslots = min(N, PM_MAX_BATCH)) — the
  // 384-KV bucket first, the long tail re-decoded in the 1056 bucket. Replaces the per-chunk
  // lockstep path so short crops never stall behind long ones. host_page_ must already hold
  // the D2H'd page.
  void decode_plusm_page(int N, const std::vector<Box> &boxes, const GpuImage &page,
                         bool drop_collapse, std::vector<FormulaEngineResult> &out);

  OrtSession fused_;                 // EXACT (GPU) + CPU path
  OrtSession enc_, prep_, step_;     // GPU FAST path
  OrtSession step_short_;            // plus-M length-bucket: 384-KV-window step (common case)
  bool fast_ = false;                // GPU host-loop (default on GPU)
  bool plusm_ = false;               // PP-FormulaNet_plus-M (6-layer MBart fast host-loop)
  bool cpu_ = false;                 // ORT CPUExecutionProvider (FORMULA_DEVICE=cpu)
  std::string label_ = "ppformulanet_s";   // backend_name() (engine label)
  bool force_fused_ = false;         // skip FAST host-loop, always fused EXACT
  std::optional<FormulaTokenizer> tok_;
  bool ready_ = false;
  std::string fast_dir_;
  cudaStream_t stream_ = nullptr;    // nullptr on the CPU path

  float *d_x_ = nullptr;             // [MAX_B,1,384,384] device crops (GPU paths only)
  // FAST-path device buffers (allocated only when fast_).
  float *d_mem_ = nullptr, *d_ck_ = nullptr, *d_cv_ = nullptr, *d_log_ = nullptr;
  float *kA_ = nullptr, *kB_ = nullptr, *vA_ = nullptr, *vB_ = nullptr;
  // plus-M length-bucket: 384-KV-window self-attention buffers (the common-case fast path).
  float *kA384_ = nullptr, *kB384_ = nullptr, *vA384_ = nullptr, *vB384_ = nullptr;
  int64_t *d_tok_ = nullptr, *d_next_ = nullptr, *d_pos_ = nullptr, *d_all_ = nullptr;
  // plus-M continuous-batch scratch: all-crop encoder memory + pre-computed cross-KV.
  float *d_mem_all_ = nullptr, *ck_all_ = nullptr, *cv_all_ = nullptr;
  std::vector<uint8_t> host_page_;   // page D2H scratch
  std::vector<float> host_in_;       // [MAX_B,1,384,384] preprocessed crops
};

}  // namespace turbo_ocr::formula
