#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// CPU inference engine using ONNX Runtime.
// Loads ONNX models and runs them on CPU with all optimizations enabled.
// Drop-in replacement for TrtEngine when no GPU is available.

namespace turbo_ocr::engine {

/// ONNX Runtime CPU inference engine (drop-in replacement for TrtEngine).
class CpuEngine {
public:
  /// Construct with path to an ONNX model file (.onnx).
  explicit CpuEngine(const std::string &model_path);
  ~CpuEngine() noexcept = default;

  /// Load the ONNX model and create an inference session.
  [[nodiscard]] bool load();

  // Run inference. Input is a flat float buffer with given shape.
  // Returns output as a flat float vector + output shape.
  struct InferResult {
    std::vector<float> data;
    std::vector<int64_t> shape;

    [[nodiscard]] bool empty() const noexcept { return data.empty(); }
  };

  [[nodiscard]] InferResult infer(const float *input_data,
                                  const std::vector<int64_t> &input_shape);

  // Batched inference. input_shape is {B,3,H,W}; the returned tensor is the
  // full {B,seq,classes} output so the caller can slice each row. Identical
  // mechanics to infer() — kept separate for call-site clarity.
  [[nodiscard]] InferResult infer_batch(const float *input_data,
                                        const std::vector<int64_t> &input_shape);

  // Zero-copy batched inference. Same input contract as infer_batch, but
  // returns a view into an engine-owned output buffer instead of copying the
  // tensor out. The view (and its data pointer) is valid until the next
  // infer*/probe call on THIS engine. Used by the rec hot path, where the
  // {B,seq,classes} tensor is large and the caller fully consumes one batch
  // before issuing the next Run.
  struct InferView {
    const float *data = nullptr;
    std::vector<int64_t> shape;

    [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  };

  [[nodiscard]] InferView
  infer_batch_view(const float *input_data,
                   const std::vector<int64_t> &input_shape);

  // Probe output dims for a given input shape (runs a dummy inference)
  void probe_output_dims(const std::vector<int64_t> &input_shape,
                         int &out_dim1, int &out_dim2);

private:
  std::string model_path_;
  // When ORT_SHARED_POOL=1, sessions draw threads from one process-wide pool
  // (see process_env()) instead of each owning its own intra-op threadpool.
  bool use_shared_pool_ = false;
  // Alternate execution provider from env ORT_EP ("xnnpack"/"dnnl"; empty/"cpu"
  // = default MLAS). Applied in load() so a missing provider fails cleanly.
  std::string ort_ep_;
  std::unique_ptr<Ort::Session> session_;
  Ort::SessionOptions session_options_;

  // Configure session_options_ with the ORT_EP provider (throws Ort::Exception
  // if the requested provider isn't available in this onnxruntime build).
  void apply_execution_provider();

  // The single process-wide ORT Env, lazily created on first use. ORT fixes an
  // Env's threadpool config at first creation, so we must NOT eagerly construct
  // a plain Env per engine: when ORT_SHARED_POOL=1 this builds the Env with
  // global threadpools (thread count from ORT_GLOBAL_THREADS); otherwise a
  // plain Env with per-session threadpools (today's behavior).
  static Ort::Env &process_env();

  std::string input_name_;
  std::string output_name_;
  Ort::AllocatorWithDefaultOptions allocator_;

  // Owns the output tensor from the most recent infer_batch_view call so its
  // buffer outlives the returned view (until the next inference call).
  Ort::Value last_output_{nullptr};
};

} // namespace turbo_ocr::engine
