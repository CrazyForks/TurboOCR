#include "formulanet_trtllm.h"
#include "turbo_ocr/formula/formula_kernels.h"

#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace turbo_ocr::formula {

namespace {

constexpr int kMaskMAlign = 128;
constexpr int kMaskNAlign = 256;
constexpr int kMaskBits   = 32;

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::fprintf(stderr, "[TRT-LLM] %s\n", msg);
  }
};

#define CUDA_OK(x)                                                             \
  do {                                                                         \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "[FormulaTRTLLM] CUDA err %s @ %s:%d\n",            \
                   cudaGetErrorString(_e), __FILE__, __LINE__);                \
      return false;                                                            \
    }                                                                          \
  } while (0)

std::vector<char> slurp(const std::string &p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f.good()) return {};
  auto n = f.tellg();
  f.seekg(0);
  std::vector<char> b(static_cast<std::size_t>(n));
  f.read(b.data(), n);
  return b;
}

int round_up(int v, int a) { return (v + a - 1) / a * a; }

int argmax_row(const float *row, int vocab) {
  int best = -1;
  float best_v = -std::numeric_limits<float>::infinity();
  for (int v = 0; v < vocab; ++v) {
    if (std::isnan(row[v])) continue;
    if (row[v] > best_v) { best_v = row[v]; best = v; }
  }
  if (best < 0) {
    std::fprintf(stderr,
                 "[FormulaTRTLLM] argmax_row: every logit was NaN (V=%d) — "
                 "decoder output corrupt, aborting AR step\n",
                 vocab);
  }
  return best;
}

// Mirror of tensorrt_llm::kernels::PackedMaskParams<MaskInputDataType>. We do
// not include the header (which transitively drags TRT-LLM common.h and
// torch-coupled bits) — we re-declare the POD exactly. Field order and types
// must match fmhaPackedMask.h for ABI compatibility.
template <typename T>
struct PackedMaskParamsLayout {
  const T *maskInput            = nullptr;
  int     *cuQSeqLens           = nullptr;
  uint32_t *packedMask          = nullptr;
  int     *cuMaskRows           = nullptr;
  const int *actualQSeqLens     = nullptr;
  const int *actualKvSeqLens    = nullptr;
  // ContextAttentionMaskType (enum class, 4 bytes) — 4 == CUSTOM_MASK
  int attentionMaskType         = 4;
  int batchSize                 = 0;
  int maxQSeqLen                = 0;
  int maxKvSeqLen               = 0;
  int slidingWindowSize         = 0;
  T   validPosVal               = T{1};
};

} // namespace

FormulaNetTrtLlmDecoder::~FormulaNetTrtLlmDecoder() noexcept {
  ctx_.reset();
  engine_.reset();
  runtime_.reset();
  for (auto &[name, b] : bufs_) {
    if (!b.owned || b.ptr == nullptr) continue;
    if (b.host) cudaFreeHost(b.ptr);
    else        cudaFree(b.ptr);
  }
  if (h_logits_) cudaFreeHost(h_logits_);
  if (plugin_handle_)  dlclose(plugin_handle_);
  if (runtime_handle_) dlclose(runtime_handle_);
}

bool FormulaNetTrtLlmDecoder::dlopen_plugin_and_kernel(
    const std::string &plugin_path, const std::string &runtime_path) {
  plugin_handle_ = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!plugin_handle_) {
    std::fprintf(stderr, "[FormulaTRTLLM] dlopen plugin %s failed: %s\n",
                 plugin_path.c_str(), dlerror());
    return false;
  }
  using InitFn = bool (*)(void *, const char *);
  auto init_fn = reinterpret_cast<InitFn>(dlsym(plugin_handle_, "initTrtLlmPlugins"));
  if (!init_fn) {
    std::fprintf(stderr, "[FormulaTRTLLM] dlsym initTrtLlmPlugins failed: %s\n", dlerror());
    return false;
  }
  if (!init_fn(logger_.get(), "tensorrt_llm")) {
    std::fprintf(stderr, "[FormulaTRTLLM] initTrtLlmPlugins returned false\n");
    return false;
  }

  runtime_handle_ = dlopen(runtime_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!runtime_handle_) {
    std::fprintf(stderr, "[FormulaTRTLLM] dlopen runtime %s failed: %s\n",
                 runtime_path.c_str(), dlerror());
    return false;
  }
  // Mangled symbol for invokeBuildPackedMask<bool>:
  //   _ZN12tensorrt_llm3_v17kernels21invokeBuildPackedMaskIbEEvRKNS1_16PackedMaskParamsIT_EEP11CUstream_st
  const char *sym =
      "_ZN12tensorrt_llm3_v17kernels21invokeBuildPackedMaskIbEEvRKNS1_"
      "16PackedMaskParamsIT_EEP11CUstream_st";
  invoke_build_packed_mask_ =
      reinterpret_cast<BuildPackedMaskFn>(dlsym(runtime_handle_, sym));
  if (!invoke_build_packed_mask_) {
    std::fprintf(stderr,
                 "[FormulaTRTLLM] dlsym invokeBuildPackedMask<bool> failed: %s\n",
                 dlerror());
    return false;
  }
  return true;
}

bool FormulaNetTrtLlmDecoder::deserialize_engine(const std::string &engine_path) {
  auto blob = slurp(engine_path);
  if (blob.empty()) {
    std::fprintf(stderr, "[FormulaTRTLLM] failed to read engine: %s\n",
                 engine_path.c_str());
    return false;
  }
  runtime_.reset(nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) return false;
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) {
    std::fprintf(stderr, "[FormulaTRTLLM] deserializeCudaEngine failed\n");
    return false;
  }
  ctx_.reset(engine_->createExecutionContext());
  if (!ctx_) return false;
  return true;
}

bool FormulaNetTrtLlmDecoder::allocate_buffers() {
  // 36 tensors. Sizes are the engine max for ctx phase (largest) so the same
  // buffer covers both ctx and gen phases.
  const int B   = kBMax;
  const int Se  = kSEncMax;
  const int Td  = kTDecMax;
  const int H   = kHidden;
  const int Ne  = kHeads;
  const int Hs  = kHeadSize;
  const int L   = kLayers;
  const int V   = kVocab;
  const std::size_t i4 = sizeof(int32_t);
  const std::size_t i8 = sizeof(int64_t);

  auto alloc_dev = [&](const std::string &name, std::size_t bytes) -> bool {
    Buf b; b.host = false; b.size = bytes;
    if (cudaMalloc(&b.ptr, std::max<std::size_t>(bytes, 1)) != cudaSuccess) {
      std::fprintf(stderr, "[FormulaTRTLLM] cudaMalloc %s (%zu) failed\n",
                   name.c_str(), bytes);
      return false;
    }
    bufs_[name] = b;
    return true;
  };
  auto alloc_host = [&](const std::string &name, std::size_t bytes) -> bool {
    Buf b; b.host = true; b.size = bytes;
    if (cudaMallocHost(&b.ptr, std::max<std::size_t>(bytes, 1)) != cudaSuccess) {
      std::fprintf(stderr, "[FormulaTRTLLM] cudaMallocHost %s (%zu) failed\n",
                   name.c_str(), bytes);
      return false;
    }
    bufs_[name] = b;
    return true;
  };

  // Packed inputs (per-stream batch B).
  if (!alloc_dev("input_ids",                    B * i4)) return false;
  if (!alloc_dev("position_ids",                 B * i4)) return false;
  if (!alloc_dev("encoder_input_lengths",        B * i4)) return false;
  if (!alloc_dev("sequence_length",              B * i4)) return false;
  if (!alloc_dev("context_lengths",              B * i4)) return false;
  if (!alloc_dev("last_token_ids",               B * i4)) return false;
  if (!alloc_host("host_past_key_value_lengths", B * i4)) return false;
  if (!alloc_host("host_context_lengths",        B * i4)) return false;
  if (!alloc_host("host_request_types",          B * i4)) return false;
  if (!alloc_host("host_runtime_perf_knobs",     16 * i8)) return false;
  if (!alloc_host("host_context_progress",       1 * i8))  return false;
  if (!alloc_host("host_max_attention_window_sizes", L * i4)) return false;
  if (!alloc_host("host_sink_token_length",      1 * i4)) return false;

  // encoder_max_input_length is a shape-carrier (only dim0 size used).
  if (!alloc_dev("encoder_max_input_length",     Se * i4)) return false;

  // encoder_output packed (B * S_enc, H) fp16.
  if (!alloc_dev("encoder_output",
                 static_cast<std::size_t>(B) * Se * H * sizeof(__half))) return false;

  // Cross attention masks. Boolean and packed int32. Sized for engine max.
  if (!alloc_dev("cross_attention_mask",
                 static_cast<std::size_t>(B) * Se * sizeof(bool))) return false;
  // Packed mask shape worst case: total_aligned_rows = B * 128, cols = ceil(Se/256)*256/32
  std::size_t pm_rows = static_cast<std::size_t>(B) * kMaskMAlign;
  std::size_t pm_cols = round_up(Se, kMaskNAlign) / kMaskBits;
  if (!alloc_dev("cross_attention_packed_mask", pm_rows * pm_cols * i4)) return false;

  // cache_indirection: [B, 1, max_seq_len] int32 — never used (no beam), zeros.
  if (!alloc_dev("cache_indirection",
                 static_cast<std::size_t>(B) * 1 * Td * i4)) return false;

  if (!alloc_dev("cross_kv_cache_gen", 1 * sizeof(bool))) return false;

  // Layer-wise self KV: per layer [B, 2, num_kv_heads, max_seq_len, head_size] fp16.
  const std::size_t kv_self_per_layer =
      static_cast<std::size_t>(B) * 2 * kKvHeads * Td * Hs * sizeof(__half);
  // Layer-wise cross KV: [B, 2, num_kv_heads, max_enc_len, head_size] fp16.
  const std::size_t kv_cross_per_layer =
      static_cast<std::size_t>(B) * 2 * kKvHeads * Se * Hs * sizeof(__half);
  for (int l = 0; l < L; ++l) {
    if (!alloc_dev("past_key_value_" + std::to_string(l),       kv_self_per_layer)) return false;
    if (!alloc_dev("cross_past_key_value_" + std::to_string(l), kv_cross_per_layer)) return false;
  }

  // Output: logits [B, V] fp32.
  if (!alloc_dev("logits",
                 static_cast<std::size_t>(B) * V * sizeof(float))) return false;

  // present_* aliases past_* (in-place KV write). Use owned=false so dtor
  // doesn't double-free.
  for (int l = 0; l < L; ++l) {
    const auto suffix = std::to_string(l);
    Buf p = bufs_["past_key_value_" + suffix];
    Buf c = bufs_["cross_past_key_value_" + suffix];
    Buf pa; pa.ptr = p.ptr; pa.size = p.size; pa.host = p.host; pa.owned = false;
    Buf ca; ca.ptr = c.ptr; ca.size = c.size; ca.host = c.host; ca.owned = false;
    bufs_["present_key_value_" + suffix]       = pa;
    bufs_["cross_present_key_value_" + suffix] = ca;
  }

  // Host pinned scratch for logits readback.
  if (cudaMallocHost(&h_logits_,
                     static_cast<std::size_t>(B) * V * sizeof(float)) != cudaSuccess) {
    std::fprintf(stderr, "[FormulaTRTLLM] cudaMallocHost h_logits failed\n");
    return false;
  }
  return true;
}

void FormulaNetTrtLlmDecoder::bind_static_inputs() {
  // host_max_attention_window_sizes is updated each step (it carries the
  // current seq-total-kv-length, not the max). Set initial to 1.
  int32_t *win = static_cast<int32_t *>(bufs_["host_max_attention_window_sizes"].ptr);
  for (int l = 0; l < kLayers; ++l) win[l] = 1;

  // host_sink_token_length = 0
  *static_cast<int32_t *>(bufs_["host_sink_token_length"].ptr) = 0;

  // host_runtime_perf_knobs = [1, -1, -1, ...] per captured Python runner.
  int64_t *knobs = static_cast<int64_t *>(bufs_["host_runtime_perf_knobs"].ptr);
  knobs[0] = 1;
  for (int i = 1; i < 16; ++i) knobs[i] = -1;

  // host_context_progress = 0
  *static_cast<int64_t *>(bufs_["host_context_progress"].ptr) = 0;

  // cache_indirection zeros — done once via cudaMemset.
  cudaMemset(bufs_["cache_indirection"].ptr, 0, bufs_["cache_indirection"].size);

  // cross_kv_cache_gen — set per-phase in stage_dynamic_inputs.
}

bool FormulaNetTrtLlmDecoder::initialize(const std::string &engine_path,
                                        const std::string &plugin_path,
                                        const std::string &runtime_path) {
  if (ready_) return true;
  logger_ = std::make_unique<TrtLogger>();
  if (!dlopen_plugin_and_kernel(plugin_path, runtime_path)) return false;
  if (!deserialize_engine(engine_path)) return false;
  if (!allocate_buffers()) return false;
  bind_static_inputs();
  ready_ = true;
  return true;
}

bool FormulaNetTrtLlmDecoder::stage_encoder_output(const float *enc_fp32_dev,
                                                  int B, int S_enc,
                                                  cudaStream_t stream) {
  // On-device fp32 -> fp16 cast directly into the engine's encoder_output
  // buffer. (Earlier host-bounce path was suspected of shifting first-token
  // argmax via differing round behavior; this path matches torch's .half().)
  const std::size_t total =
      static_cast<std::size_t>(B) * S_enc * kHidden;
  kernels::cuda_encoder_cast_f32_to_f16(
      enc_fp32_dev, bufs_["encoder_output"].ptr, total, stream);
  return true;
}

bool FormulaNetTrtLlmDecoder::stage_dynamic_inputs(int B, int S_enc,
                                                   int step_idx,
                                                   bool is_context,
                                                   cudaStream_t stream) {
  // Captured Python-runner ground truth for B=1, S_enc=305, MAX_NEW=4:
  //   call0 (ctx):  input_ids=[1]  position_ids=[0]  request_types=[0]
  //                 host_past_kv=[1]  host_ctx_len=[1]  seq_len=[1]
  //                 last_token_ids=[1]
  //   call1 (gen1): input_ids=[61] position_ids=[1]  request_types=[1]
  //                 host_past_kv=[1]  host_ctx_len=[1]  seq_len=[2]
  //                 last_token_ids=[1]
  //   call2 (gen2): position_ids=[2] host_past_kv=[2]  seq_len=[3]
  //   host_max_attention_window_sizes = [max_new_tokens+1]*4 (FIXED, not
  //   per-step) — sized for the whole generation, not the current step.
  // Convention: step_idx counts ctx as 0, first gen as 1, etc.

  int32_t rt = is_context ? 0 : 1;
  int32_t *h_rt = static_cast<int32_t *>(bufs_["host_request_types"].ptr);
  for (int i = 0; i < B; ++i) h_rt[i] = rt;

  // host_past_key_value_lengths:
  //   ctx (step_idx=0):     1
  //   gen-step k:           k (gen1=1, gen2=2, ...)
  int32_t *h_pkv = static_cast<int32_t *>(bufs_["host_past_key_value_lengths"].ptr);
  int32_t pkv_val = is_context ? 1 : step_idx;
  for (int i = 0; i < B; ++i) h_pkv[i] = pkv_val;

  // host_context_lengths: always 1 (per captured bins for both phases).
  int32_t *h_cl = static_cast<int32_t *>(bufs_["host_context_lengths"].ptr);
  for (int i = 0; i < B; ++i) h_cl[i] = 1;

  // host_max_attention_window_sizes is set once in bind_static_inputs() to the
  // total generation budget (max_new_tokens+1). Do NOT update per-step.

  // sequence_length: ctx=1, gen-step k = k+1.
  std::vector<int32_t> seq(B);
  std::vector<int32_t> ctxl(B, 1);
  std::vector<int32_t> encil(B, S_enc);
  std::vector<int32_t> ltok(B);
  std::vector<int32_t> pos(B);
  int32_t seq_val = is_context ? 1 : (step_idx + 1);
  for (int i = 0; i < B; ++i) {
    seq[i]  = seq_val;
    ltok[i] = i + 1;          // prefix-sum of packed lengths (1 per stream)
    pos[i]  = is_context ? 0 : step_idx;
  }
  CUDA_OK(cudaMemcpyAsync(bufs_["sequence_length"].ptr, seq.data(),
                          B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
  CUDA_OK(cudaMemcpyAsync(bufs_["context_lengths"].ptr, ctxl.data(),
                          B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
  CUDA_OK(cudaMemcpyAsync(bufs_["encoder_input_lengths"].ptr, encil.data(),
                          B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
  CUDA_OK(cudaMemcpyAsync(bufs_["last_token_ids"].ptr, ltok.data(),
                          B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
  CUDA_OK(cudaMemcpyAsync(bufs_["position_ids"].ptr, pos.data(),
                          B * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

  // cross_attention_mask: [B, S_enc] all true (no encoder padding).
  std::vector<uint8_t> mask_u8(static_cast<std::size_t>(B) * S_enc, 1);
  CUDA_OK(cudaMemcpyAsync(bufs_["cross_attention_mask"].ptr, mask_u8.data(),
                          mask_u8.size() * sizeof(uint8_t),
                          cudaMemcpyHostToDevice, stream));

  // cross_kv_cache_gen: True only on ctx phase (compute cross-KV once); False
  // on every gen step (reuse cached cross-KV).
  uint8_t kv_gen = is_context ? 1 : 0;
  CUDA_OK(cudaMemcpyAsync(bufs_["cross_kv_cache_gen"].ptr, &kv_gen, 1,
                          cudaMemcpyHostToDevice, stream));
  return true;
}

bool FormulaNetTrtLlmDecoder::set_phase_shapes(int B, int S_enc, int step_idx,
                                               bool is_context) {
  using Dims = nvinfer1::Dims;
  auto set1 = [&](const char *name, int d0) -> bool {
    Dims d{}; d.nbDims = 1; d.d[0] = d0;
    return ctx_->setInputShape(name, d);
  };
  auto set2 = [&](const char *name, int d0, int d1) -> bool {
    Dims d{}; d.nbDims = 2; d.d[0] = d0; d.d[1] = d1;
    return ctx_->setInputShape(name, d);
  };
  auto set3 = [&](const char *name, int d0, int d1, int d2) -> bool {
    Dims d{}; d.nbDims = 3; d.d[0] = d0; d.d[1] = d1; d.d[2] = d2;
    return ctx_->setInputShape(name, d);
  };
  auto set5 = [&](const char *name, int d0, int d1, int d2, int d3, int d4) -> bool {
    Dims d{}; d.nbDims = 5;
    d.d[0] = d0; d.d[1] = d1; d.d[2] = d2; d.d[3] = d3; d.d[4] = d4;
    return ctx_->setInputShape(name, d);
  };

  // Per Python-runner ground truth (call0=ctx, call1=gen-step1 of B=1, S_enc=305):
  //   encoder_output: ctx=(S_enc, 256) -> we pass (B*S_enc, 256). gen=(0, 256).
  //   cross_attention_packed_mask: ctx=(B*128, packed_cols). gen=(B, 1).
  //   cache_indirection: (B, 1, kv_total) where kv_total = step_idx+1 for gen,
  //                       and 1 for ctx (just produced BOS slot).
  //   past_key_value_l: (B, 2, num_kv_heads, kv_total, head_size).

  const int q_total = B;  // one new token per stream per call
  if (!set1("input_ids",             q_total)) return false;
  if (!set1("position_ids",          q_total)) return false;
  if (!set1("encoder_input_lengths", B))       return false;
  if (!set1("encoder_max_input_length", S_enc)) return false;
  if (!set1("sequence_length",       B))       return false;
  if (!set1("context_lengths",       B))       return false;
  if (!set1("last_token_ids",        B))       return false;
  if (!set1("host_past_key_value_lengths", B)) return false;
  if (!set1("host_context_lengths",  B))       return false;
  if (!set1("host_request_types",    B))       return false;
  if (!set1("host_runtime_perf_knobs", 16))    return false;
  if (!set1("host_context_progress",  1))      return false;
  if (!set1("host_max_attention_window_sizes", kLayers)) return false;
  if (!set1("host_sink_token_length", 1))      return false;
  if (!set1("cross_kv_cache_gen", 1))           return false;

  // encoder_output: only the ctx call carries encoder data; gen passes (0, H).
  if (is_context) {
    if (!set2("encoder_output", B * S_enc, kHidden)) return false;
  } else {
    if (!set2("encoder_output", 0, kHidden)) return false;
  }
  if (!set2("cross_attention_mask", B, S_enc)) return false;

  // Packed mask: ctx (B*kMaskMAlign, packed_cols); gen (B, 1).
  if (is_context) {
    const int packed_cols = round_up(S_enc, kMaskNAlign) / kMaskBits;
    if (!set2("cross_attention_packed_mask", B * kMaskMAlign, packed_cols)) return false;
  } else {
    if (!set2("cross_attention_packed_mask", B, 1)) return false;
  }

  // Per Python EncDecModelRunner, past_kv shape stays FIXED across the whole
  // generation at (B, 2, num_kv_heads, max_decoder_seq, head_size) where
  // max_decoder_seq = max_new_tokens + 1. cache_indirection last dim matches.
  const int kv_dim = kv_slots_;
  if (!set3("cache_indirection", B, 1, kv_dim)) return false;

  for (int l = 0; l < kLayers; ++l) {
    auto s_name = "past_key_value_" + std::to_string(l);
    auto c_name = "cross_past_key_value_" + std::to_string(l);
    if (!set5(s_name.c_str(), B, 2, kKvHeads, kv_dim, kHeadSize)) return false;
    if (!set5(c_name.c_str(), B, 2, kKvHeads, S_enc, kHeadSize)) return false;
  }
  (void)step_idx;
  return true;
}

bool FormulaNetTrtLlmDecoder::enqueue_step(cudaStream_t stream) {
  // Bind every tensor (TRT requires a valid address even for outputs).
  for (const auto &[name, b] : bufs_) {
    if (!ctx_->setTensorAddress(name.c_str(), b.ptr)) {
      std::fprintf(stderr, "[FormulaTRTLLM] setTensorAddress(%s) failed\n",
                   name.c_str());
      return false;
    }
  }
  if (!ctx_->allInputDimensionsSpecified() ||
      !ctx_->allInputShapesSpecified()) {
    std::fprintf(stderr, "[FormulaTRTLLM] input shapes/dims not fully set\n");
    return false;
  }
  if (!ctx_->enqueueV3(stream)) {
    std::fprintf(stderr, "[FormulaTRTLLM] enqueueV3 failed\n");
    return false;
  }
  return true;
}

bool FormulaNetTrtLlmDecoder::copy_logits_host(int B, cudaStream_t stream,
                                               std::vector<float> &out) {
  out.resize(static_cast<std::size_t>(B) * kVocab);
  CUDA_OK(cudaMemcpyAsync(out.data(), bufs_["logits"].ptr,
                          out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
  CUDA_OK(cudaStreamSynchronize(stream));
  return true;
}

bool FormulaNetTrtLlmDecoder::decode(const float *encoder_ctx_fp32, int B,
                                     int S_enc, int max_new_tokens,
                                     int bos_id, int eos_id,
                                     cudaStream_t stream,
                                     std::vector<std::vector<int32_t>> &out_tokens) {
  if (!ready_) return false;
  // Singleton driver — serialize concurrent decode() calls.
  std::lock_guard<std::mutex> lock(decode_mu_);
  if (B <= 0 || B > kBMax) {
    std::fprintf(stderr, "[FormulaTRTLLM] B=%d out of range\n", B); return false;
  }
  if (S_enc <= 0 || S_enc > kSEncMax) {
    std::fprintf(stderr, "[FormulaTRTLLM] S_enc=%d out of range\n", S_enc); return false;
  }
  max_new_tokens = std::min(max_new_tokens, kTDecMax - 1);
  kv_slots_ = max_new_tokens + 1;  // fixed kv-dim across all calls

  // Update host_max_attention_window_sizes once per decode (= kv_slots_).
  int32_t *win = static_cast<int32_t *>(bufs_["host_max_attention_window_sizes"].ptr);
  for (int l = 0; l < kLayers; ++l) win[l] = kv_slots_;

  out_tokens.assign(B, {});
  for (int i = 0; i < B; ++i) {
    out_tokens[i].reserve(max_new_tokens + 1);
    out_tokens[i].push_back(bos_id);
  }

  if (!stage_encoder_output(encoder_ctx_fp32, B, S_enc, stream)) return false;

  std::vector<bool> done(B, false);
  std::vector<int32_t> next_ids(B, bos_id);
  std::vector<float> logits;

  // Step 0: context phase (Q-len=1 per stream; total Q-tokens = B).
  // input_ids = [BOS] * B
  {
    std::vector<int32_t> ids(B, bos_id);
    if (cudaMemcpyAsync(bufs_["input_ids"].ptr, ids.data(),
                        B * sizeof(int32_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) return false;
  }
  if (!stage_dynamic_inputs(B, S_enc, /*step_idx=*/0,
                             /*is_context=*/true, stream)) return false;

  // Build packed mask on-device via the dlopen'd kernel symbol.
  // q_total = B (one new token per stream). max_q = round_up(1, 128) = 128.
  // packed_rows = B * 128.
  std::vector<int32_t> q_seqlens(B, 1);
  std::vector<int32_t> kv_seqlens(B, S_enc);
  // Stash q/kv seqlens to device for the kernel.
  Buf &qbuf = bufs_["cross_attention_mask"];  // reuse not safe; allocate scratch
  (void)qbuf;
  // Use small dedicated scratch buffers; allocate on the fly (tiny, ~B*4).
  int32_t *d_qseq = nullptr, *d_kvseq = nullptr,
          *d_cuq = nullptr,  *d_curows = nullptr;
  CUDA_OK(cudaMalloc(&d_qseq,   B * sizeof(int32_t)));
  CUDA_OK(cudaMalloc(&d_kvseq,  B * sizeof(int32_t)));
  CUDA_OK(cudaMalloc(&d_cuq,    (B + 1) * sizeof(int32_t)));
  CUDA_OK(cudaMalloc(&d_curows, (B + 1) * sizeof(int32_t)));
  CUDA_OK(cudaMemcpyAsync(d_qseq, q_seqlens.data(), B * sizeof(int32_t),
                          cudaMemcpyHostToDevice, stream));
  CUDA_OK(cudaMemcpyAsync(d_kvseq, kv_seqlens.data(), B * sizeof(int32_t),
                          cudaMemcpyHostToDevice, stream));

  auto build_packed_mask = [&](int batch, int max_q, int max_kv,
                               int packed_rows) -> bool {
    PackedMaskParamsLayout<bool> p{};
    p.maskInput          = static_cast<const bool *>(bufs_["cross_attention_mask"].ptr);
    p.cuQSeqLens         = nullptr;  // 3D mask layout (we pass [B, max_q=1, max_kv])
    p.packedMask         = static_cast<uint32_t *>(bufs_["cross_attention_packed_mask"].ptr);
    p.cuMaskRows         = d_curows;
    p.actualQSeqLens     = d_qseq;
    p.actualKvSeqLens    = d_kvseq;
    p.attentionMaskType  = 4;  // CUSTOM_MASK
    p.batchSize          = batch;
    p.maxQSeqLen         = max_q;
    p.maxKvSeqLen        = max_kv;
    p.slidingWindowSize  = 0;
    p.validPosVal        = true;
    invoke_build_packed_mask_(&p, stream);
    return true;
  };

  // Build the ctx-phase packed mask (Q=1 per stream, kv=S_enc per stream).
  if (!build_packed_mask(B, 1, S_enc, B * kMaskMAlign)) return false;

  // Debug: read back first 16 int32s of packed_mask and dump to stderr.
  // Expected pattern (from Python capture): [0x33333333, 0x33333333,
  //   0x33333333, 0x33333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  if (std::getenv("TURBO_OCR_TRTLLM_DEBUG")) {
    cudaStreamSynchronize(stream);
    int32_t mask_dbg[16] = {0};
    cudaMemcpy(mask_dbg, bufs_["cross_attention_packed_mask"].ptr,
               sizeof(mask_dbg), cudaMemcpyDeviceToHost);
    std::fprintf(stderr, "[FormulaTRTLLM] ctx packed_mask row0: ");
    for (int k = 0; k < 16; ++k) std::fprintf(stderr, "%08x ", mask_dbg[k]);
    std::fprintf(stderr, "\n");
  }

  if (!set_phase_shapes(B, S_enc, /*step_idx=*/0, /*is_context=*/true)) return false;
  if (!enqueue_step(stream)) return false;
  if (!copy_logits_host(B, stream, logits)) return false;

  for (int i = 0; i < B; ++i) {
    const float *row = logits.data() + static_cast<std::size_t>(i) * kVocab;
    int tok = argmax_row(row, kVocab);
    out_tokens[i].push_back(tok);
    next_ids[i] = tok;
    if (tok == eos_id) done[i] = true;
  }
  if (std::getenv("TURBO_OCR_TRTLLM_DEBUG")) {
    std::fprintf(stderr, "[FormulaTRTLLM] B=%d S_enc=%d ctx top-5: ",
                 B, S_enc);
    // Top-5 of slot 0
    const float *row = logits.data();
    int top5[5] = {0,1,2,3,4};
    for (int v = 5; v < kVocab; ++v) {
      // Insertion: maintain top5 sorted descending
      for (int k = 0; k < 5; ++k) {
        if (row[v] > row[top5[k]]) {
          for (int m = 4; m > k; --m) top5[m] = top5[m-1];
          top5[k] = v; break;
        }
      }
    }
    for (int k = 0; k < 5; ++k)
      std::fprintf(stderr, "%d(%.3f) ", top5[k], row[top5[k]]);
    std::fprintf(stderr, "\n");
  }

  for (int step = 1; step < max_new_tokens; ++step) {
    bool all_done = std::all_of(done.begin(), done.end(),
                                [](bool d){ return d; });
    if (all_done) break;

    // Stage gen-phase: input_ids = next_ids per stream.
    if (cudaMemcpyAsync(bufs_["input_ids"].ptr, next_ids.data(),
                        B * sizeof(int32_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      cudaFree(d_qseq); cudaFree(d_kvseq); cudaFree(d_cuq); cudaFree(d_curows);
      return false;
    }
    if (!stage_dynamic_inputs(B, S_enc, /*step_idx=*/step,
                               /*is_context=*/false, stream)) {
      cudaFree(d_qseq); cudaFree(d_kvseq); cudaFree(d_cuq); cudaFree(d_curows);
      return false;
    }
    // Gen-phase packed-mask is a tiny (B, 1) placeholder per the captured
    // runner; the kernel is not invoked here (cross-KV is already cached
    // from the ctx phase, and gen attends to it without rebuilding the mask).
    if (!set_phase_shapes(B, S_enc, /*step_idx=*/step, /*is_context=*/false)) {
      cudaFree(d_qseq); cudaFree(d_kvseq); cudaFree(d_cuq); cudaFree(d_curows);
      return false;
    }
    if (!enqueue_step(stream)) {
      cudaFree(d_qseq); cudaFree(d_kvseq); cudaFree(d_cuq); cudaFree(d_curows);
      return false;
    }
    if (!copy_logits_host(B, stream, logits)) {
      cudaFree(d_qseq); cudaFree(d_kvseq); cudaFree(d_cuq); cudaFree(d_curows);
      return false;
    }
    for (int i = 0; i < B; ++i) {
      if (done[i]) continue;
      const float *row = logits.data() + static_cast<std::size_t>(i) * kVocab;
      int tok = argmax_row(row, kVocab);
      out_tokens[i].push_back(tok);
      next_ids[i] = tok;
      if (tok == eos_id) done[i] = true;
    }
  }

  cudaFree(d_qseq); cudaFree(d_kvseq); cudaFree(d_cuq); cudaFree(d_curows);
  return true;
}

} // namespace turbo_ocr::formula
