#include "turbo_ocr/formula/ppformulanet_ort.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/formula/ppformulanet_preprocess.h"

namespace fs = std::filesystem;

namespace turbo_ocr::formula {

namespace {
constexpr int MAX_B = 32, S = kFormulaInputSize;
// MAXLEN = the static self-attention KV buffer (must match step_batched.onnx's 1056).
// The model's learned positional embedding is 1029 long -> caps at 1026 tokens; the old
// 512 buffer corrupted long formulas (>510 tok) — the sole cause of the FAST-path gap.
// MAXIT*3 = 1026 tokens covers the full range.
constexpr int H = 16, Dh = 24, CTX = 144, VOCAB = 50000, MAXLEN = 1056, MAXIT = 342, CHECK = 16;
static_assert(MAXIT * 3 <= MAXLEN,
              "decode writes 3 KV slots per step at pos=it*3; MAXIT*3 must fit MAXLEN");

// argmax over VOCAB per FP32 logit row. The strided per-thread scan + strict-greater
// tree reduction biases toward lower indices on exact ties, but does not guarantee the
// strict lowest-index tie-break a serial torch/ORT argmax gives; exact FP32 ties are
// vanishingly rare here, so the distinction is immaterial in practice.
__global__ void argmax_kernel(const float *lg, int64_t *out, int rows, int V) {
  int row = blockIdx.x; if (row >= rows) return;
  const float *p = lg + static_cast<size_t>(row) * V;
  __shared__ float sm[256]; __shared__ int si[256];
  float m = -1e30f; int mi = 0;
  for (int i = threadIdx.x; i < V; i += blockDim.x) { float v = p[i]; if (v > m) { m = v; mi = i; } }
  sm[threadIdx.x] = m; si[threadIdx.x] = mi; __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s && sm[threadIdx.x + s] > sm[threadIdx.x]) {
      sm[threadIdx.x] = sm[threadIdx.x + s]; si[threadIdx.x] = si[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = si[0];
}
__global__ void fill_pos(int64_t *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = static_cast<int64_t>(i) * 3;
}
}  // namespace

PPFormulaNetOrt::PPFormulaNetOrt() = default;
PPFormulaNetOrt::~PPFormulaNetOrt() noexcept {
  free_buffers();
  if (stream_) cudaStreamDestroy(stream_);
}

bool PPFormulaNetOrt::alloc_buffers() {
  host_in_.assign((size_t)MAX_B * S * S, 0.0f);
  if (cpu_) return true;  // CPU path: host tensors only, no device buffers/stream
  auto m = [](void **p, size_t n) { return cudaMalloc(p, n) == cudaSuccess; };
  if (!m((void **)&d_x_, (size_t)MAX_B * S * S * sizeof(float))) return false;
  if (fast_) {  // static-KV host-loop scratch
    size_t kv = (size_t)2 * MAX_B * H * MAXLEN * Dh, cr = (size_t)2 * MAX_B * H * CTX * Dh;
    bool ok = m((void **)&d_mem_, (size_t)MAX_B * CTX * 2048 * sizeof(float))
        && m((void **)&d_ck_, cr * sizeof(float)) && m((void **)&d_cv_, cr * sizeof(float))
        && m((void **)&d_log_, (size_t)MAX_B * 3 * VOCAB * sizeof(float))
        && m((void **)&kA_, kv * sizeof(float)) && m((void **)&kB_, kv * sizeof(float))
        && m((void **)&vA_, kv * sizeof(float)) && m((void **)&vB_, kv * sizeof(float))
        && m((void **)&d_tok_, (size_t)MAX_B * 3 * sizeof(int64_t))
        && m((void **)&d_next_, (size_t)MAX_B * 3 * sizeof(int64_t))
        && m((void **)&d_pos_, (size_t)MAXIT * sizeof(int64_t))
        && m((void **)&d_all_, (size_t)MAXIT * MAX_B * 3 * sizeof(int64_t));
    if (!ok) return false;
    fill_pos<<<(MAXIT + 63) / 64, 64>>>(d_pos_, MAXIT);
    if (cudaGetLastError() != cudaSuccess) return false;
  }
  return cudaDeviceSynchronize() == cudaSuccess;
}

void PPFormulaNetOrt::free_buffers() noexcept {
  for (void *p : {(void *)d_x_, (void *)d_mem_, (void *)d_ck_, (void *)d_cv_, (void *)d_log_,
                  (void *)kA_, (void *)kB_, (void *)vA_, (void *)vB_,
                  (void *)d_tok_, (void *)d_next_, (void *)d_pos_, (void *)d_all_})
    if (p) cudaFree(p);
}

bool PPFormulaNetOrt::load_model_dir(const std::string &model_dir) {
  fs::path mp(model_dir);
  fs::path base = fs::is_directory(mp) ? mp : mp.parent_path();
  fs::path fast = base / "fast";
  // Device select: FORMULA_DEVICE=cpu -> ORT CPUExecutionProvider (no CUDA, no Python).
  // Otherwise GPU, where FAST (host-loop) is the default and matches the fused CDM
  // exactly (0.811) at ~8x speed; PPFNS_EXACT=1 forces the fused graph on GPU.
  const char *dev = std::getenv("FORMULA_DEVICE");
  std::string devl = dev ? dev : "";
  std::transform(devl.begin(), devl.end(), devl.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  cpu_ = (devl == "cpu");
  fast_ = !cpu_ && std::getenv("PPFNS_EXACT") == nullptr;
  fast_dir_ = fast.string();
  if (!cpu_ && cudaStreamCreate(&stream_) != cudaSuccess) {
    std::cerr << "[PPFormulaNetOrt] FATAL: CUDA stream create failed\n";
    return false;
  }
  if (!alloc_buffers()) {
    std::cerr << "[PPFormulaNetOrt] FATAL: buffer alloc failed\n";
    return false;
  }
  bool ok;
  if (cpu_) {
    // CPU: the fused graph on ORT-CPU with host tensors (the FAST host loop needs CUDA).
    ok = fused_.load_cpu((base / "inference_trt.onnx").string());
    std::cerr << "[PPFormulaNetOrt] CPU decode path (fused graph, ORT CPUExecutionProvider)\n";
  } else if (fast_) {
    // FAST: encoder + cross-KV prep + static-KV step, all ORT-CUDA-13 on our stream,
    // driven by a host AR loop. The encoder is bit-exact to the fused in-graph encoder.
    // These split graphs are locally generated (scratchpad/fastdec/batched_step.py) and
    // are NOT in the committed bundle, so a fresh deploy may ship only the fused
    // inference_trt.onnx. If any fast/ file is missing or fails to load, fall back to the
    // fused graph (slower) with a LOUD warning rather than hard-failing boot.
    const fs::path enc_p = fast / "encoder.onnx", prep_p = fast / "prep.onnx",
                   step_p = fast / "step_batched.onnx";
    const bool have_all =
        fs::exists(enc_p) && fs::exists(prep_p) && fs::exists(step_p);
    const bool fast_loaded =
        have_all && enc_.load(enc_p.string(), 0, stream_, false)
        && prep_.load(prep_p.string(), 0, stream_, false)
        && step_.load(step_p.string(), 0, stream_, false);
    if (fast_loaded) {
      ok = true;
      std::cerr << "[PPFormulaNetOrt] FAST decode path (encoder+prep+step host-loop)\n";
    } else {
      std::cerr << "[PPFormulaNetOrt] WARNING: FAST artifacts missing in " << fast.string()
                << " — falling back to fused graph (slower); regenerate with "
                   "scratchpad/fastdec/batched_step.py\n";
      fast_ = false;
      ok = fused_.load((base / "inference_trt.onnx").string(), 0, stream_);
    }
  } else {
    // EXACT (PPFNS_EXACT=1): the fused graph (encoder + AR Loop) batched on ORT-CUDA.
    ok = fused_.load((base / "inference_trt.onnx").string(), 0, stream_);
  }
  if (!ok) { std::cerr << "[PPFormulaNetOrt] FATAL: formula model load failed under " << base << '\n'; return false; }
  ready_ = static_cast<bool>(tok_);  // ready once both model+tokenizer loaded
  if (fast_ && std::getenv("PPFNS_GOLDEN_SELFTEST")) {
    const int B = 12;
    std::ifstream f(fast_dir_ + "/golden3_crops.bin", std::ios::binary);
    if (f) {
      std::vector<float> crops((size_t)B * S * S);
      f.read(reinterpret_cast<char *>(crops.data()), crops.size() * sizeof(float));
      cudaMemcpyAsync(d_x_, crops.data(), crops.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
      bool e = true;
      for (int b = 0; b < B && e; ++b) {
        OrtTensor ex{"x", d_x_ + (size_t)b * S * S, {1, 1, S, S}, false};
        OrtTensor em{"p2o.pd_op.transpose.0.0", d_mem_ + (size_t)b * CTX * 2048, {1, CTX, 2048}, false};
        e = enc_.run({ex}, {em});
      }
      cudaStreamSynchronize(stream_);
      { std::vector<float> mh((size_t)B * CTX * 2048);
        cudaMemcpy(mh.data(), d_mem_, mh.size() * sizeof(float), cudaMemcpyDeviceToHost);
        std::ofstream mf("/tmp/cpp_mem.bin", std::ios::binary);
        mf.write(reinterpret_cast<char *>(mh.data()), mh.size() * sizeof(float)); }
      const int64_t Bi = B;
      OrtTensor pin{"memory", d_mem_, {Bi, CTX, 2048}, false};
      OrtTensor pck{"ck", d_ck_, {2, Bi, H, CTX, Dh}, false}, pcv{"cv", d_cv_, {2, Bi, H, CTX, Dh}, false};
      prep_.run({pin}, {pck, pcv});
      std::vector<std::vector<int64_t>> seqs; decode_chunk(B, seqs);
      std::ofstream o("/tmp/cpp_fast_tokens.json"); o << '[';
      for (int b = 0; b < B; ++b) {
        o << (b ? ",[" : "["); for (size_t j = 0; j < seqs[b].size(); ++j) o << (j ? "," : "") << seqs[b][j]; o << ']';
        std::cerr << "[golden-fast] crop" << b << ": " << seqs[b].size() << " tok\n";
      }
      o << "]\n"; std::cerr << "[golden-fast] wrote /tmp/cpp_fast_tokens.json\n";
    }
  }
  return true;
}

bool PPFormulaNetOrt::load_tokenizer(const std::string &path) {
  tok_ = FormulaTokenizer::load(path);
  if (!tok_) { std::cerr << "[PPFormulaNetOrt] tokenizer load failed: " << path << '\n'; return false; }
  if (fast_ ? step_.ready() : fused_.ready()) ready_ = true;
  return true;
}

// FAST path: lockstep-batched static-KV decode. Build the step bindings once and only
// retarget the ping-pong KV + pos pointers each iteration; all async on stream_ with a
// deferred EOS check every CHECK steps. Fills clean content token rows (BOS/pad/EOS
// stripped) per crop. Returns false if a step/argmax/CUDA error truncated the decode
// (the partial rows are still returned, but the caller must mark the crops failed).
bool PPFormulaNetOrt::decode_chunk(int B, std::vector<std::vector<int64_t>> &out) {
  const int64_t EOS = tok_->eos_id(), Bi = B;
  size_t kvB = (size_t)2 * B * H * MAXLEN * Dh * sizeof(float);
  cudaMemsetAsync(kA_, 0, kvB, stream_); cudaMemsetAsync(vA_, 0, kvB, stream_);
  cudaMemsetAsync(d_tok_, 0, (size_t)B * 3 * sizeof(int64_t), stream_);
  float *kin = kA_, *kout = kB_, *vin = vA_, *vout = vB_;
  std::vector<OrtTensor> ins = {
      {"tokens", d_tok_, {Bi, 3}, true}, {"pos", d_pos_, {1}, true},
      {"kb", kin, {2, Bi, H, MAXLEN, Dh}, false}, {"vb", vin, {2, Bi, H, MAXLEN, Dh}, false},
      {"ck", d_ck_, {2, Bi, H, CTX, Dh}, false}, {"cv", d_cv_, {2, Bi, H, CTX, Dh}, false}};
  std::vector<OrtTensor> outs = {
      {"logits", d_log_, {Bi, 3, VOCAB}, false}, {"kb_out", kout, {2, Bi, H, MAXLEN, Dh}, false},
      {"vb_out", vout, {2, Bi, H, MAXLEN, Dh}, false}};
  // d_all_ is never zeroed, so `last` MUST bound the steps actually written this
  // call. On either error break only steps 0..it-1 are valid -> set last=it; the
  // clean all-done break sets last=it+1. ok=false on any error so the caller can
  // mark the chunk's crops failed instead of decoding stale tokens as success.
  std::vector<char> done(B, 0); std::vector<int64_t> hall; int last = MAXIT; bool ok = true;
  for (int it = 0; it < MAXIT; ++it) {
    ins[1].data = d_pos_ + it; ins[2].data = kin; ins[3].data = vin;
    outs[1].data = kout; outs[2].data = vout;
    if (!step_.run(ins, outs)) {
      std::cerr << "[PPFormulaNetOrt] step run failed it=" << it << '\n';
      last = it; ok = false; break;
    }
    argmax_kernel<<<B * 3, 256, 0, stream_>>>(d_log_, d_next_, B * 3, VOCAB);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
      std::cerr << "[PPFormulaNetOrt] argmax launch failed it=" << it << ": " << cudaGetErrorString(e) << '\n';
      last = it; ok = false; break;
    }
    cudaMemcpyAsync(d_all_ + (size_t)it * B * 3, d_next_, (size_t)B * 3 * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(d_tok_, d_next_, (size_t)B * 3 * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream_);
    std::swap(kin, kout); std::swap(vin, vout);
    if ((it + 1) % CHECK == 0) {
      cudaStreamSynchronize(stream_);
      size_t n = (size_t)(it + 1) * B * 3; hall.resize(n);
      cudaMemcpy(hall.data(), d_all_, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
      int nd = 0;
      for (int b = 0; b < B; ++b) {
        if (done[b]) { ++nd; continue; }
        bool e = false;
        for (int s = 0; s <= it && !e; ++s)
          for (int j = 0; j < 3; ++j) if (hall[(size_t)s * B * 3 + (size_t)b * 3 + j] == EOS) { e = true; break; }
        if (e) { done[b] = 1; ++nd; }
      }
      if (nd == B) { last = it + 1; break; }
    }
  }
  cudaStreamSynchronize(stream_);
  // The async memset/memcpy launches in the AR loop don't check their returns;
  // surface any error the sync just exposed (e.g. an illegal access) rather than
  // decoding from a corrupt d_all_.
  if (cudaError_t e = cudaPeekAtLastError(); e != cudaSuccess) {
    std::cerr << "[PPFormulaNetOrt] decode_chunk CUDA error: " << cudaGetErrorString(e) << '\n';
    ok = false;
  }
  size_t n = (size_t)last * B * 3; hall.resize(n);
  cudaMemcpy(hall.data(), d_all_, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
  out.assign(B, {});
  for (int b = 0; b < B; ++b) {
    bool stop = false;
    for (int s = 0; s < last && !stop; ++s)
      for (int j = 0; j < 3; ++j) {
        int64_t t = hall[(size_t)s * B * 3 + (size_t)b * 3 + j];
        if (t == EOS) { stop = true; break; }
        if (t != 0 && t != 1) out[b].push_back(t);
      }
  }
  return ok;
}

std::vector<FormulaEngineResult>
PPFormulaNetOrt::run(const GpuImage &page, const std::vector<Box> &boxes, cudaStream_t stream) {
  std::vector<FormulaEngineResult> out;
  if (boxes.empty()) return out;
  out.resize(boxes.size());
  if (!ready_ || page.empty()) { for (auto &r : out) r.ok = false; return out; }

  size_t need = (size_t)page.rows * page.step;
  if (host_page_.size() < need) host_page_.resize(need);
  cudaMemcpyAsync(host_page_.data(), page.data, need, cudaMemcpyDeviceToHost, stream);
  if (cudaStreamSynchronize(stream) != cudaSuccess) { for (auto &r : out) r.ok = false; return out; }

  const int N = (int)boxes.size();
  static const bool drop_collapse = std::getenv("PPFNS_DROP_COLLAPSE") != nullptr;
  // Decode chunk size. The fused encoder's batched cuDNN conv drifts slightly from
  // single-sample at large batches (flips a few near-tie tokens); small batches match
  // the per-crop reference while still amortizing the AR Loop. Tunable for testing.
  static const int chunk = []{ const char *e = std::getenv("PPFNS_CHUNK");
    int c = e ? std::atoi(e) : 8; return c < 1 ? 1 : (c > MAX_B ? MAX_B : c); }();
  for (int s0 = 0; s0 < N; s0 += chunk) {
    const int B = std::min(chunk, N - s0);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < B; ++i) {
      auto cr = clamped_crop_rect(boxes[s0 + i], page.cols, page.rows);
      const int x0 = cr[0], y0 = cr[1], w = cr[2], h = cr[3];
      std::vector<uint8_t> tmp((size_t)std::max(1, w) * std::max(1, h) * 3);
      const uint8_t *sp = host_page_.data() + (size_t)y0 * page.step + (size_t)x0 * 3;
      for (int r = 0; r < h; ++r)
        std::memcpy(tmp.data() + (size_t)r * w * 3, sp + (size_t)r * page.step, (size_t)w * 3);
      formula_preprocess_one(tmp.data(), w, h, host_in_.data() + (size_t)i * S * S);
    }
    if (!cpu_)
      cudaMemcpyAsync(d_x_, host_in_.data(), (size_t)B * S * S * sizeof(float), cudaMemcpyHostToDevice, stream_);
    std::vector<std::vector<int64_t>> seqs(B);   // clean content tokens per crop
    bool chunk_ok = true;                        // false -> mark this chunk's crops failed
    if (fast_) {
      // Encode per crop (batch=1 matches the reference) -> d_mem_, cross-KV prep -> ck/cv,
      // then the static-KV host AR loop. All on stream_, deferred sync inside decode_chunk.
      bool eok = true;
      for (int b = 0; b < B && eok; ++b) {
        OrtTensor ex{"x", d_x_ + (size_t)b * S * S, {1, 1, S, S}, false};
        OrtTensor em{"p2o.pd_op.transpose.0.0", d_mem_ + (size_t)b * CTX * 2048, {1, CTX, 2048}, false};
        eok = enc_.run({ex}, {em});
      }
      const int64_t Bi = B;
      OrtTensor pin{"memory", d_mem_, {Bi, CTX, 2048}, false};
      OrtTensor pck{"ck", d_ck_, {2, Bi, H, CTX, Dh}, false};
      OrtTensor pcv{"cv", d_cv_, {2, Bi, H, CTX, Dh}, false};
      if (!eok || !prep_.run({pin}, {pck, pcv})) {
        std::cerr << "[PPFormulaNetOrt] fast encode/prep failed\n";
        for (int i = 0; i < B; ++i) out[s0 + i].ok = false; continue;
      }
      chunk_ok = decode_chunk(B, seqs);
    } else {
      // EXACT (GPU) or CPU: batched fused graph. CPU binds the HOST crops directly;
      // GPU syncs the H2D first then binds d_x_.
      const float *xin;
      if (cpu_) {
        xin = host_in_.data();
      } else {
        cudaStreamSynchronize(stream_);  // fused reads d_x_
        xin = d_x_;
      }
      std::vector<int64_t> flat; int64_t L = 0;
      if (!fused_.run_tokens("x", "fetch_name_0", xin, B, flat, L)) {
        std::cerr << "[PPFormulaNetOrt] fused decode failed\n";
        for (int i = 0; i < B; ++i) out[s0 + i].ok = false; continue;
      }
      const int64_t EOS = tok_->eos_id();
      for (int i = 0; i < B; ++i)
        for (int64_t j = 0; j < L; ++j) {
          int64_t t = flat[(size_t)i * L + j];
          if (t == EOS) break;
          if (t != 0 && t != 1) seqs[i].push_back(t);  // drop BOS(0)/pad(1)
        }
    }
    for (int i = 0; i < B; ++i) {
      std::string latex = tok_->decode(seqs[i], /*post_process=*/false);
      // Dropping collapsed formulas to empty measured WORSE (it also drops legit long
      // matrices). Emit-everything is better; keep the detector behind an opt-in env.
      out[s0 + i].latex = (drop_collapse && formula_is_mode_collapsed(seqs[i], latex)) ? std::string() : latex;
      out[s0 + i].token_count = seqs[i].size();
      out[s0 + i].hit_eos = !seqs[i].empty();  // EOS-stripped seq -> non-empty == normal stop
      out[s0 + i].ok = chunk_ok;               // false -> decode_chunk truncated on error
    }
  }
  return out;
}

}  // namespace turbo_ocr::formula
