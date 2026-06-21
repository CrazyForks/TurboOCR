#include "turbo_ocr/engine/trt_engine.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/errors.h"

#include <cstdlib>
#include <format>
#include <fstream>
#include <mutex>
#include <vector>

using namespace turbo_ocr::engine;

TrtEngine::TrtEngine(const std::string &model_path) : model_path_(model_path) {}

TrtEngine::~TrtEngine() noexcept { destroy_graphs(); }

bool TrtEngine::graphs_enabled() {
  // Default OFF: on this GPU-compute-bound OCR stack a clean A/B showed baked
  // graphs are neutral even at concurrency 1 (the eliminated launch overhead
  // is lost under GPU-compute + sync time) and cost ~0.5 GiB/pipeline. Kept
  // as an opt-in (TURBO_OCR_CUDA_GRAPHS=1) — they pay off only if the pipeline
  // ever becomes launch-bound (e.g. after a compute cut that frees the GPU).
  static const bool v = [] {
    const char *env = std::getenv("TURBO_OCR_CUDA_GRAPHS");
    return env && std::atoi(env) != 0;
  }();
  return v;
}

void TrtEngine::destroy_graphs() noexcept {
  for (auto &g : baked_) {
    if (g.exec)
      cudaGraphExecDestroy(g.exec);
    g.ctx.reset();
    if (g.arena)
      cudaFree(g.arena);
  }
  baked_.clear();
}

int TrtEngine::bake_graph(int profile_idx, const nvinfer1::Dims &dims,
                          void *input, void *output, cudaStream_t stream) {
  if (!engine_ || !graphs_enabled())
    return -1;
  // Pipelines warm up concurrently; captures are globally serialized so no
  // other CUDA activity from this process interleaves with a recording.
  static std::mutex bake_mu;
  std::lock_guard<std::mutex> lock(bake_mu);

  auto ctx = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext(
          nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if (!ctx)
    return -1;
  if (profile_idx != 0) {
    ctx->setOptimizationProfileAsync(profile_idx, stream);
    cudaStreamSynchronize(stream);
  }
  ctx->setTensorAddress(input_name_.c_str(), input);
  ctx->setTensorAddress(output_name_.c_str(), output);
  if (!ctx->setInputShape(input_name_.c_str(), dims) ||
      !ctx->allInputDimensionsSpecified())
    return -1;

  const int64_t arena_size =
      engine_->getDeviceMemorySizeForProfileV2(profile_idx);
  void *arena = nullptr;
  if (arena_size <= 0 ||
      cudaMalloc(&arena, static_cast<size_t>(arena_size)) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  ctx->setDeviceMemoryV2(arena, arena_size);

  auto fail = [&] {
    cudaGetLastError();
    ctx.reset(); // before the arena it points into
    cudaFree(arena);
    return -1;
  };

  // Two real executions then a synced capture — the trtexec recipe. Myelin
  // finalizes per-context device state during the first runs; capturing any
  // earlier records a graph bound to unfinalized state.
  for (int k = 0; k < 2; ++k)
    if (!ctx->enqueueV3(stream))
      return fail();
  if (cudaStreamSynchronize(stream) != cudaSuccess)
    return fail();

  cudaGraph_t graph = nullptr;
  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) !=
      cudaSuccess)
    return fail();
  const bool enq_ok = ctx->enqueueV3(stream);
  const cudaError_t cap = cudaStreamEndCapture(stream, &graph);
  if (!enq_ok || cap != cudaSuccess || !graph) {
    if (graph)
      cudaGraphDestroy(graph);
    std::cerr << std::format(
        "[TRT] graph capture failed for {} profile {} — plain enqueue\n",
        model_path_, profile_idx);
    return fail();
  }
  cudaGraphExec_t exec = nullptr;
  if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
    cudaGraphDestroy(graph);
    return fail();
  }
  cudaGraphDestroy(graph);

  // Smoke-replay once so a broken graph surfaces at warmup, not in traffic.
  if (cudaGraphLaunch(exec, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaGraphExecDestroy(exec);
    return fail();
  }

  BakedGraph g;
  g.exec = exec;
  g.ctx = std::move(ctx);
  g.arena = arena;
  g.out_dims = g.ctx->getTensorShape(output_name_.c_str());
  baked_.push_back(std::move(g));
  return static_cast<int>(baked_.size()) - 1;
}

bool TrtEngine::launch_baked(int slot, cudaStream_t stream) {
  return slot >= 0 && static_cast<size_t>(slot) < baked_.size() &&
         cudaGraphLaunch(baked_[static_cast<size_t>(slot)].exec, stream) ==
             cudaSuccess;
}

bool TrtEngine::load() {
  std::ifstream file(model_path_, std::ios::binary);
  if (!file.good()) [[unlikely]] {
    std::cerr << std::format("[TRT] Error loading engine file: {}", model_path_) << '\n';
    return false;
  }

  file.seekg(0, file.end);
  auto pos = file.tellg();
  if (pos < 0) [[unlikely]] {
    std::cerr << std::format("[TRT] Error reading engine file size: {}", model_path_) << '\n';
    return false;
  }
  auto size = static_cast<size_t>(pos);
  file.seekg(0, file.beg);

  std::vector<char> trtModelStream(size);
  file.read(trtModelStream.data(), size);

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) [[unlikely]] {
    std::cerr << std::format("[TRT] Failed to create runtime for: {}", model_path_) << '\n';
    return false;
  }

  engine_.reset(runtime_->deserializeCudaEngine(trtModelStream.data(), size));
  if (!engine_) [[unlikely]] {
    std::cerr << std::format("[TRT] Failed to deserialize engine: {}", model_path_) << '\n';
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) [[unlikely]] {
    std::cerr << std::format("[TRT] Failed to create execution context: {}", model_path_) << '\n';
    return false;
  }

  auto nbIO = engine_->getNbIOTensors();
  for (int i = 0; i < nbIO; ++i) {
    const char *name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
      input_names_.emplace_back(name);
    else
      output_names_.emplace_back(name);
  }
  // Single-IO convenience pointers (first of each). Multi-IO callers use
  // input_names() / output_names() directly.
  if (!input_names_.empty())  input_name_  = input_names_[0];
  if (!output_names_.empty()) output_name_ = output_names_[0];

  return true;
}

void TrtEngine::bind_io(void *input, void *output) {
  if (!context_) [[unlikely]]
    return;
  bound_input_ = input;
  bound_output_ = output;
  context_->setTensorAddress(input_name_.c_str(), input);
  context_->setTensorAddress(output_name_.c_str(), output);
}

static constexpr bool dims_equal(const nvinfer1::Dims &a, const nvinfer1::Dims &b) {
  if (a.nbDims != b.nbDims) return false;
  for (int i = 0; i < a.nbDims; ++i)
    if (a.d[i] != b.d[i]) return false;
  return true;
}

bool TrtEngine::infer_dynamic(const nvinfer1::Dims &input_dims,
                              cudaStream_t stream) {
  if (!context_) [[unlikely]]
    return false;
  if (!dims_equal(input_dims, last_input_dims_)) {
    // TRT 10.14: all profiles share the same base tensor names.
    // setInputShape applies to whichever profile is currently active.
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) [[unlikely]] {
      std::cerr << std::format("[TRT] setInputShape FAILED for input=({},{},{},{}) profile={} on {}",
                               input_dims.d[0], input_dims.d[1], input_dims.d[2], input_dims.d[3],
                               current_profile_, model_path_) << '\n';
      return false;
    }
    last_input_dims_ = input_dims;
  }
  // PP-DocLayoutV3 (and any future multi-input model) has more than one
  // dynamic-shape input. If a caller forgets to set one, enqueueV3 silently
  // produces garbage output with no error, so guard the dispatch.
  if (!context_->allInputDimensionsSpecified()) [[unlikely]]
    throw turbo_ocr::InferenceError(
        "input dimensions not all set before enqueueV3 (" + model_path_ + ")");
  if (!context_->enqueueV3(stream)) [[unlikely]] {
    // A sticky fault here (illegal address, ECC, launch failure, …) has
    // poisoned the context for every pipeline in the process — fail-fast
    // rather than serve garbage. Recoverable faults fall through to false.
    turbo_ocr::abort_on_sticky_cuda_fault("TrtEngine::infer_dynamic enqueueV3");
    return false;
  }
  return true;
}

void TrtEngine::select_profile(int profile_idx, cudaStream_t stream) {
  if (profile_idx == current_profile_)
    return;

  if (profile_idx < 0 || profile_idx >= engine_->getNbOptimizationProfiles()) {
    std::cerr << std::format("[TRT] Invalid profile index {} (engine has {})",
                             profile_idx, engine_->getNbOptimizationProfiles())
              << '\n';
    return;
  }

  context_->setOptimizationProfileAsync(profile_idx, stream);
  current_profile_ = profile_idx;
  // Invalidate cached dims — new profile requires setInputShape again
  last_input_dims_ = {};

  // Re-bind I/O addresses for the new profile (TRT clears them on profile switch)
  if (bound_input_) context_->setTensorAddress(input_name_.c_str(), bound_input_);
  if (bound_output_) context_->setTensorAddress(output_name_.c_str(), bound_output_);
}

int TrtEngine::num_profiles() const noexcept {
  return engine_ ? engine_->getNbOptimizationProfiles() : 1;
}

nvinfer1::Dims TrtEngine::get_output_dims() const noexcept {
  if (!context_) [[unlikely]]
    return {};
  return context_->getTensorShape(output_name_.c_str());
}

void TrtEngine::set_tensor_address(const std::string &name, void *ptr) {
  if (!context_) [[unlikely]]
    return;
  context_->setTensorAddress(name.c_str(), ptr);
}

bool TrtEngine::set_input_shape(const std::string &name,
                                const nvinfer1::Dims &dims) {
  if (!context_) [[unlikely]]
    return false;
  if (!context_->setInputShape(name.c_str(), dims)) [[unlikely]] {
    std::cerr << std::format("[TRT] setInputShape FAILED for input={} on {}",
                             name, model_path_) << '\n';
    return false;
  }
  return true;
}

bool TrtEngine::execute(cudaStream_t stream) {
  if (!context_) [[unlikely]]
    return false;
  // PP-DocLayoutV3 (and any future multi-input model) has more than one
  // dynamic-shape input. If a caller forgets to set one, enqueueV3 silently
  // produces garbage output with no error, so guard the dispatch.
  if (!context_->allInputDimensionsSpecified()) [[unlikely]]
    throw turbo_ocr::InferenceError(
        "input dimensions not all set before enqueueV3 (" + model_path_ + ")");
  if (!context_->enqueueV3(stream)) [[unlikely]] {
    // A sticky fault here (illegal address, ECC, launch failure, …) has
    // poisoned the context for every pipeline in the process — fail-fast
    // rather than serve garbage. Recoverable faults fall through to false.
    turbo_ocr::abort_on_sticky_cuda_fault("TrtEngine::execute enqueueV3");
    return false;
  }
  return true;
}

nvinfer1::Dims TrtEngine::tensor_shape(const std::string &name) const {
  if (!context_) [[unlikely]]
    return {};
  return context_->getTensorShape(name.c_str());
}

void TrtEngine::probe_output_dims(const nvinfer1::Dims &input_dims,
                                   int &out_seq_len, int &out_num_classes) {
  if (!context_) [[unlikely]]
    return;
  if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
    std::cerr << "[TRT] probe_output_dims: setInputShape failed\n";
    return;
  }
  nvinfer1::Dims od = context_->getTensorShape(output_name_.c_str());
  if (od.nbDims >= 3) {
    out_seq_len = od.d[1];
    out_num_classes = od.d[2];
  }
  // Invalidate cached dims so the next infer_dynamic() call will re-set the
  // actual inference shape (probe may have used a different shape).
  last_input_dims_ = {};
}
