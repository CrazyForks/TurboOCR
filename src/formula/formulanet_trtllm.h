#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

namespace turbo_ocr::formula {

// Drives the TRT-LLM formula decoder engine via a dlopen-ed plugin .so plus a
// dlopen-ed libtensorrt_llm.so for the packed-mask kernel symbol. Replaces the
// vanilla-TRT decoder path (which produces context-blind garbage on real
// images) with the engine validated bit-exact against Python EncDecModelRunner.
class FormulaNetTrtLlmDecoder {
 public:
  FormulaNetTrtLlmDecoder() = default;
  ~FormulaNetTrtLlmDecoder() noexcept;

  FormulaNetTrtLlmDecoder(const FormulaNetTrtLlmDecoder &) = delete;
  FormulaNetTrtLlmDecoder &operator=(const FormulaNetTrtLlmDecoder &) = delete;

  // Resolve the plugin .so + kernel .so symbols, deserialize the engine, and
  // pre-allocate all 36 IO buffers at engine max shapes. Idempotent.
  // engine_path defaults to /tmp/variant_a/engine_b128_seq512/decoder/rank0.engine
  // and plugin_path / runtime_path default to the build-venv locations used by
  // the validated cpp_smoke harness.
  [[nodiscard]] bool initialize(const std::string &engine_path,
                                const std::string &plugin_path,
                                const std::string &runtime_path);

  [[nodiscard]] bool is_ready() const noexcept { return ready_; }

  // Greedy decode for a batch. encoder_ctx_fp32 lives on device in shape
  // [B, S_enc, H=256] (float). The driver converts to FP16 internally. Tokens
  // for each lane are written into out_tokens[i] (starts with BOS, ends at EOS
  // or at engine max_seq_len=512). Returns false on any hard failure.
  [[nodiscard]] bool decode(const float *encoder_ctx_fp32,
                            int B, int S_enc, int max_new_tokens,
                            int bos_id, int eos_id,
                            cudaStream_t stream,
                            std::vector<std::vector<int32_t>> &out_tokens);

 private:
  bool dlopen_plugin_and_kernel(const std::string &plugin_path,
                                const std::string &runtime_path);
  bool deserialize_engine(const std::string &engine_path);
  bool allocate_buffers();

  // Set static-shape inputs and addresses that don't change across steps
  // (cache_indirection zeros, runtime_perf_knobs, sink_token_length, etc).
  void bind_static_inputs();

  // Compute and stage the packed mask + cross_attention_mask for one phase.
  // is_context: true on step 0 (Q-length=1 per stream, request_types=0),
  // false on subsequent gen steps (Q-length=1, request_types=1).
  bool stage_dynamic_inputs(int B, int S_enc, int step_idx, bool is_context,
                            cudaStream_t stream);

  // Stage encoder_output (packed [B*S_enc, H] fp16) from a device-side fp32
  // source. Also fills the encoder_max_input_length carrier whose only meaning
  // is the size of dim 0.
  bool stage_encoder_output(const float *enc_fp32_dev, int B, int S_enc,
                            cudaStream_t stream);

  // Execution helpers.
  bool set_phase_shapes(int B, int S_enc, int step_idx, bool is_context);
  bool enqueue_step(cudaStream_t stream);

  // Copy logits row for each lane back to the host.
  bool copy_logits_host(int B, cudaStream_t stream, std::vector<float> &out);

  // dlopen handles + resolved symbols.
  void *plugin_handle_  = nullptr;
  void *runtime_handle_ = nullptr;
  // invokeBuildPackedMask<bool>(PackedMaskParams<bool> const&, cudaStream_t)
  // Demangled signature; called by pointer.
  using BuildPackedMaskFn = void (*)(const void *params, cudaStream_t stream);
  BuildPackedMaskFn invoke_build_packed_mask_ = nullptr;

  std::unique_ptr<nvinfer1::ILogger>          logger_;
  // Shared, process-global engine (deserialized once per .trt path via the
  // engine cache). ctx_ below stays per-instance/per-thread.
  std::shared_ptr<nvinfer1::ICudaEngine>       engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;

  // Device + pinned-host buffers, keyed by tensor name. Owns the allocations.
  struct Buf {
    void *ptr        = nullptr;
    std::size_t size = 0;
    bool        host = false;  // pinned host vs device alloc
    bool        owned = true;
  };
  std::unordered_map<std::string, Buf> bufs_;

  // Logits dump area (host pinned, sized for B_max * V).
  void *h_logits_ = nullptr;

  // Static engine constants pulled from config.json at build time.
  // Pipeline batches ≤ kFormulaBatchMax=4 per call; sized to 16 for headroom.
  // Engine's max_batch_size is 128 so the engine still accepts any B ≤ 16.
  static constexpr int kBMax       = 16;
  static constexpr int kSEncMax    = 512;
  static constexpr int kTDecMax    = 512;
  static constexpr int kHidden     = 256;
  static constexpr int kHeads      = 8;
  static constexpr int kHeadSize   = 64;
  static constexpr int kKvHeads    = 8;
  static constexpr int kLayers     = 4;
  static constexpr int kVocab      = 8000;

  bool ready_ = false;

  // Total decoder slots reserved for this generation (set at decode() start
  // to max_new_tokens+1). Used as the fixed kv-dim of past_key_value_l for
  // every call within one generation, matching Python's EncDecModelRunner.
  int kv_slots_ = 0;

  // Serializes decode() calls. The driver owns a single IExecutionContext
  // and shared device + pinned-host buffers, so concurrent calls would
  // corrupt cross-stream state (host_request_types, sequence_length, etc).
  std::mutex decode_mu_;
};

} // namespace turbo_ocr::formula
