#include "turbo_ocr/formula/ort_session.h"

#include <onnxruntime_cxx_api.h>

#include <iostream>

namespace turbo_ocr::formula {

struct OrtSession::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "ppfns_ort"};
  std::unique_ptr<Ort::Session> sess;
  Ort::MemoryInfo mem{nullptr};
  int device_id = 0;
  bool ready = false;
  // Persistent binding for run_graph() (CUDA-graph replay): built once, reused.
  std::unique_ptr<Ort::IoBinding> binding;
  std::vector<Ort::Value> gvals;  // own the Value wrappers across replays
};

OrtSession::OrtSession() : p_(std::make_unique<Impl>()) {}
OrtSession::~OrtSession() = default;

#if !defined(TURBO_CPU_ONLY)
bool OrtSession::load(const std::string &onnx_path, int device_id, void *cuda_stream,
                      bool do_copy_default_stream, bool enable_cuda_graph) {
  try {
    p_->device_id = device_id;
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // V2 CUDA provider options so we can optionally hand ORT our cudaStream_t.
    // do_copy_in_default_stream=1 is REQUIRED by ORT's CUDA Loop op (the fused
    // PP-FormulaNet-S AR decoder is a Loop) — a dedicated copy stream is rejected.
    // enable_cuda_graph=1 captures the (static-shape, fixed-address) step graph on the
    // first run_graph() and replays it after — used only for the plus-M decoder step.
    const OrtApi &api = Ort::GetApi();
    OrtCUDAProviderOptionsV2 *cuda = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
    std::string dev = std::to_string(device_id);
    const char *keys[] = {"device_id", "do_copy_in_default_stream", "enable_cuda_graph"};
    const char *vals[] = {dev.c_str(), do_copy_default_stream ? "1" : "0",
                          enable_cuda_graph ? "1" : "0"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda, keys, vals, 3));
    if (cuda_stream)
      Ort::ThrowOnError(
          api.UpdateCUDAProviderOptionsWithValue(cuda, "user_compute_stream", cuda_stream));
    opts.AppendExecutionProvider_CUDA_V2(*cuda);
    api.ReleaseCUDAProviderOptions(cuda);

    p_->sess = std::make_unique<Ort::Session>(p_->env, onnx_path.c_str(), opts);
    p_->mem = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, device_id, OrtMemTypeDefault);
    p_->ready = true;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[OrtSession] FATAL load failed (" << onnx_path << "): " << e.what() << '\n';
    return false;
  }
}
#endif  // !TURBO_CPU_ONLY

bool OrtSession::load_cpu(const std::string &onnx_path) {
  try {
    p_->device_id = -1;
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // No execution provider appended => ORT's default CPUExecutionProvider.
    p_->sess = std::make_unique<Ort::Session>(p_->env, onnx_path.c_str(), opts);
    p_->mem = Ort::MemoryInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
    p_->ready = true;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[OrtSession] FATAL CPU load failed (" << onnx_path << "): " << e.what() << '\n';
    return false;
  }
}

bool OrtSession::ready() const { return p_->ready; }

std::vector<std::string> OrtSession::input_names() const {
  std::vector<std::string> names;
  if (!p_->sess) return names;
  Ort::AllocatorWithDefaultOptions alloc;
  size_t n = p_->sess->GetInputCount();
  names.reserve(n);
  for (size_t i = 0; i < n; ++i)
    names.emplace_back(p_->sess->GetInputNameAllocated(i, alloc).get());
  return names;
}

std::vector<std::string> OrtSession::output_names() const {
  std::vector<std::string> names;
  if (!p_->sess) return names;
  Ort::AllocatorWithDefaultOptions alloc;
  size_t n = p_->sess->GetOutputCount();
  names.reserve(n);
  for (size_t i = 0; i < n; ++i)
    names.emplace_back(p_->sess->GetOutputNameAllocated(i, alloc).get());
  return names;
}

bool OrtSession::run(const std::vector<OrtTensor> &inputs,
                     const std::vector<OrtTensor> &outputs) {
  try {
    auto numel = [](const std::vector<int64_t> &s) {
      size_t n = 1; for (int64_t d : s) n *= static_cast<size_t>(d); return n;
    };
    Ort::IoBinding binding(*p_->sess);
    std::vector<Ort::Value> vals;  // own the Value wrappers for the Run() lifetime
    vals.reserve(inputs.size() + outputs.size());
    auto make = [&](const OrtTensor &t) -> Ort::Value & {
      size_t n = numel(t.shape);
      vals.push_back(t.i64
          ? Ort::Value::CreateTensor<int64_t>(p_->mem, static_cast<int64_t *>(t.data), n,
                                              t.shape.data(), t.shape.size())
          : Ort::Value::CreateTensor<float>(p_->mem, static_cast<float *>(t.data), n,
                                            t.shape.data(), t.shape.size()));
      return vals.back();
    };
    for (const auto &t : inputs) binding.BindInput(t.name, make(t));
    for (const auto &t : outputs) binding.BindOutput(t.name, make(t));
    p_->sess->Run(Ort::RunOptions{nullptr}, binding);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[OrtSession] run failed: " << e.what() << '\n';
    return false;
  }
}

bool OrtSession::run_graph(const std::vector<OrtTensor> &inputs,
                           const std::vector<OrtTensor> &outputs) {
  try {
    if (!p_->binding) {  // first call: build + cache the binding (fixed buffers)
      auto numel = [](const std::vector<int64_t> &s) {
        size_t n = 1; for (int64_t d : s) n *= static_cast<size_t>(d); return n;
      };
      p_->binding = std::make_unique<Ort::IoBinding>(*p_->sess);
      p_->gvals.clear();
      p_->gvals.reserve(inputs.size() + outputs.size());
      auto make = [&](const OrtTensor &t) -> Ort::Value & {
        size_t n = numel(t.shape);
        p_->gvals.push_back(t.i64
            ? Ort::Value::CreateTensor<int64_t>(p_->mem, static_cast<int64_t *>(t.data), n,
                                                t.shape.data(), t.shape.size())
            : Ort::Value::CreateTensor<float>(p_->mem, static_cast<float *>(t.data), n,
                                              t.shape.data(), t.shape.size()));
        return p_->gvals.back();
      };
      for (const auto &t : inputs) p_->binding->BindInput(t.name, make(t));
      for (const auto &t : outputs) p_->binding->BindOutput(t.name, make(t));
    }
    p_->sess->Run(Ort::RunOptions{nullptr}, *p_->binding);  // captures on 1st call, replays after
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[OrtSession] run_graph failed: " << e.what() << '\n';
    return false;
  }
}

void OrtSession::reset_graph() {
  p_->binding.reset();
  p_->gvals.clear();
}

bool OrtSession::run_tokens(const char *in_name, const char *out_name, const float *d_x,
                            int64_t B, std::vector<int64_t> &tokens, int64_t &L) {
  try {
    Ort::IoBinding binding(*p_->sess);
    int64_t shp[4] = {B, 1, 384, 384};
    Ort::Value xv = Ort::Value::CreateTensor<float>(
        p_->mem, const_cast<float *>(d_x), (size_t)B * 1 * 384 * 384, shp, 4);
    binding.BindInput(in_name, xv);
    // Let ORT allocate the dynamic [B,L] token output directly in host memory.
    Ort::MemoryInfo cpu("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    binding.BindOutput(out_name, cpu);
    binding.SynchronizeInputs();
    p_->sess->Run(Ort::RunOptions{nullptr}, binding);
    binding.SynchronizeOutputs();
    std::vector<Ort::Value> outs = binding.GetOutputValues();
    auto shape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    L = shape.size() >= 2 ? shape[1] : 0;
    const int64_t *d = outs[0].GetTensorData<int64_t>();
    tokens.assign(d, d + (size_t)B * L);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[OrtSession] run_tokens failed: " << e.what() << '\n';
    return false;
  }
}

}  // namespace turbo_ocr::formula
