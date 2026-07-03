#include "turbo_ocr/formula/ppformulanet_ort.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
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

// Reduce (val,idx) across a warp keeping the LOWEST lane index on exact-value ties
// (strict-greater replaces, so the lower lane survives). __shfl_down_sync keeps the
// whole reduction in registers — no shared traffic, no per-level __syncthreads.
__device__ __forceinline__ void warp_argmax(float &v, int &i) {
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) {
    float ov = __shfl_down_sync(0xffffffffu, v, s);
    int oi = __shfl_down_sync(0xffffffffu, i, s);
    if (ov > v) { v = ov; i = oi; }
  }
}
// argmax over VOCAB per FP32 logit row. Coalesced strided scan (consecutive lanes read
// consecutive logits) -> per-lane max, then a two-level warp-shuffle reduction (intra-warp,
// then across the <=32 warp winners in warp 0). For every input with a STRICT row max
// (i.e. real model logits) this returns exactly that max's index. Tie-break caveat: on
// an EXACT FP32 tie the survivor is the tied index with the lowest THREAD id (lowest
// i % 256), NOT the lowest vocab index a serial numpy/paddle argmax returns — e.g. a
// tie between logits[100] and logits[257] emits 257 (thread 1), not 100. Exact cross-
// stripe FP32 ties do not occur in real logits, so decoded output is unchanged in
// practice (empirically bit-identical formula CDM) — but "bit-exact to reference
// argmax" is guaranteed only for strict maxima.
__global__ void argmax_kernel(const float *lg, int64_t *out, int rows, int V) {
  int row = blockIdx.x; if (row >= rows) return;
  const float *p = lg + static_cast<size_t>(row) * V;
  float m = -1e30f; int mi = 0;
  for (int i = threadIdx.x; i < V; i += blockDim.x) { float v = p[i]; if (v > m) { m = v; mi = i; } }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  warp_argmax(m, mi);  // lane 0 of each warp now holds that warp's winner
  __shared__ float sm[32]; __shared__ int si[32];
  if (lane == 0) { sm[warp] = m; si[warp] = mi; }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = (blockDim.x + 31) >> 5;
    m = lane < nwarps ? sm[lane] : -1e30f;
    mi = lane < nwarps ? si[lane] : 0;
    warp_argmax(m, mi);
    if (lane == 0) out[row] = mi;
  }
}
// On-device EOS termination: scan the newly-decoded steps [s0,s1) of the flat token
// history (all[step*B*Ks + b*Ks + j]) and set a STICKY per-row done flag when any of the
// row's Ks slots equals EOS. The host then D2Hs only the B flags (not the growing token
// history) and does no per-token rescan — it just reads the flags to decide termination.
__global__ void scan_eos_range(const int64_t *all, unsigned char *done, int B, int Ks,
                               int64_t EOS, int s0, int s1) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B || done[b]) return;
  for (int s = s0; s < s1; ++s) {
    const int64_t *row = all + ((size_t)s * B + b) * Ks;
    for (int j = 0; j < Ks; ++j)
      if (row[j] == EOS) { done[b] = 1; return; }
  }
}
__global__ void fill_pos(int64_t *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = static_cast<int64_t>(i) * 3;
}
// PP-FormulaNet_plus-M: 6-layer MBart decoder, Dh=32, one greedy token/step, per-seq
// pos[B], 1056-cap static KV. decoder_step.onnx emits next_token in-graph (greedy
// argmax) so the host loop threads next_token->tokens device-to-device. Decoder start
// token = </s> (id 2, the MBart eos==decoder_start convention); decode stops on out id 2.
constexpr int PM_LAYERS = 6, PM_Dh = 32, PM_MAXIT = 1056, PM_START = 2;
constexpr int PM_MAX_N = 128;  // continuous-batch queue capacity (crops pre-encoded)
// ORT-CUDA reliably runs the plus-M step at batch <= 30 but throws at exactly MAX_B=32
// (the step graph's batch dim is dynamic, so this is a runtime quirk, not a graph cap);
// clamp the active batch here so production never hits it.
constexpr int PM_MAX_BATCH = 30;
// Length-bucketing: most formulas decode in well under 384 tokens, and the static-KV step
// attends over the FULL KV window every step, so a 384-wide bucket cuts per-step attention
// + the KV-output write ~2.75x vs the 1056 window. Crops that exceed 384 tokens re-decode
// in the 1056 buffers. 384 covers the OmniDocBench isolated-formula gate (max 331 tokens).
constexpr int PM_MAXLEN_S = 384;
__global__ void fill_val(int64_t *p, int64_t v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}
// Accumulate the just-written self-KV slot at per-row position pos[b] from src (kb_out)
// into dst (kb), for every (layer,row,head,dim). Replaces the full-buffer ping-pong swap
// so the step's KV input/output addresses stay FIXED across iterations — the precondition
// for ORT CUDA-graph capture/replay. Only LY*B*H*Dh elements move (one position), so it is
// far cheaper than copying the whole 1056-wide buffer.
__global__ void accum_kv_slot(float *dst, const float *src, const int64_t *pos,
                              int LY, int B, int Hh, int MAXL, int DHh) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = LY * B * Hh * DHh;
  if (idx >= total) return;
  int d = idx % DHh; int t = idx / DHh;
  int h = t % Hh; t /= Hh;
  int b = t % B; t /= B;
  int l = t;
  const int64_t p = pos[b];
  const size_t off = ((((size_t)l * B + b) * Hh + h) * (size_t)MAXL + (size_t)p) * DHh + d;
  dst[off] = src[off];
}
}  // namespace

PPFormulaNetOrt::PPFormulaNetOrt(std::string backend_label)
    : label_(std::move(backend_label)) {}
PPFormulaNetOrt::~PPFormulaNetOrt() noexcept {
  free_buffers();
  if (stream_) cudaStreamDestroy(stream_);
}

bool PPFormulaNetOrt::alloc_buffers() {
  host_in_.assign((size_t)MAX_B * S * S, 0.0f);
  auto m = [](void **p, size_t n) { return cudaMalloc(p, n) == cudaSuccess; };
  if (!m((void **)&d_x_, (size_t)MAX_B * S * S * sizeof(float))) return false;
  if (fast_) {  // static-KV host-loop scratch
    // plus-M: 6 layers / Dh=32 / 1 KV slot per step / pos[B].  PP-FormulaNet-S:
    // 2 layers / Dh=24 / 3 slots per step (MTP) / pos[MAXIT] precomputed (it*3).
    const int LY = plusm_ ? PM_LAYERS : 2, DH = plusm_ ? PM_Dh : Dh;
    const int MIT = plusm_ ? PM_MAXIT : MAXIT, KS = plusm_ ? 1 : 3;
    size_t kv = (size_t)LY * MAX_B * H * MAXLEN * DH, cr = (size_t)LY * MAX_B * H * CTX * DH;
    bool ok = m((void **)&d_mem_, (size_t)MAX_B * CTX * 2048 * sizeof(float))
        && m((void **)&d_ck_, cr * sizeof(float)) && m((void **)&d_cv_, cr * sizeof(float))
        && m((void **)&d_log_, (size_t)MAX_B * 3 * VOCAB * sizeof(float))
        && m((void **)&kA_, kv * sizeof(float)) && m((void **)&kB_, kv * sizeof(float))
        && m((void **)&vA_, kv * sizeof(float)) && m((void **)&vB_, kv * sizeof(float))
        && m((void **)&d_tok_, (size_t)MAX_B * 3 * sizeof(int64_t))
        && m((void **)&d_next_, (size_t)MAX_B * 3 * sizeof(int64_t))
        && m((void **)&d_pos_, (size_t)std::max(MAXIT, MAX_B) * sizeof(int64_t))
        && m((void **)&d_all_, (size_t)MIT * MAX_B * KS * sizeof(int64_t))
        && m((void **)&d_done_, (size_t)MAX_B * sizeof(unsigned char));
    if (!ok) return false;
    if (plusm_) {  // continuous-batch: all-crop encoder memory + pre-computed cross-KV
      size_t cra = (size_t)LY * PM_MAX_N * H * CTX * DH;
      size_t kvs = (size_t)LY * MAX_B * H * PM_MAXLEN_S * DH;  // 384-window self-KV bucket
      if (!m((void **)&d_mem_all_, (size_t)PM_MAX_N * CTX * 2048 * sizeof(float))
          || !m((void **)&ck_all_, cra * sizeof(float))
          || !m((void **)&cv_all_, cra * sizeof(float))
          || !m((void **)&kA384_, kvs * sizeof(float)) || !m((void **)&kB384_, kvs * sizeof(float))
          || !m((void **)&vA384_, kvs * sizeof(float)) || !m((void **)&vB384_, kvs * sizeof(float)))
        return false;
    } else {  // -S precomputes pos=it*3; plus-M fills pos=it per step in-loop
      fill_pos<<<(MAXIT + 63) / 64, 64, 0, stream_>>>(d_pos_, MAXIT);
      if (cudaGetLastError() != cudaSuccess) return false;
    }
  }
  // Sync OUR stream only. cudaDeviceSynchronize would be illegal while any
  // other pipeline's CUDA-graph capture is in flight (device-wide sync against
  // an active capture fails regardless of capture mode) — and pipelines warm up
  // concurrently, so a formula load racing another pipeline's rec-graph bake
  // would spuriously fail. Launching fill_pos on stream_ (not the legacy
  // stream) likewise avoids invalidating a concurrent capture through an
  // implicit legacy-stream dependency.
  return cudaStreamSynchronize(stream_) == cudaSuccess;
}

void PPFormulaNetOrt::free_buffers() noexcept {
  for (void *p : {(void *)d_x_, (void *)d_mem_, (void *)d_ck_, (void *)d_cv_, (void *)d_log_,
                  (void *)kA_, (void *)kB_, (void *)vA_, (void *)vB_,
                  (void *)d_tok_, (void *)d_next_, (void *)d_pos_, (void *)d_all_, (void *)d_done_,
                  (void *)d_mem_all_, (void *)ck_all_, (void *)cv_all_,
                  (void *)kA384_, (void *)kB384_, (void *)vA384_, (void *)vB384_})
    if (p) cudaFree(p);
}

bool PPFormulaNetOrt::load_model_dir(const std::string &model_dir) {
  fs::path mp(model_dir);
  fs::path base = fs::is_directory(mp) ? mp : mp.parent_path();
  fs::path fast = base / "fast";
  // GPU FAST is the only path: the host-loop matches the fused CDM exactly (0.811)
  // at ~8x speed. There is no fused EXACT mode and no FORMULA_DEVICE=cpu here — CPU
  // formula runs through CpuFormulaRecognizer instead.
  fast_ = true;
  plusm_ = (label_ == "ppformulanet_plus_m");  // 6-layer MBart fast host-loop
  fast_dir_ = fast.string();
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    std::cerr << "[PPFormulaNetOrt] FATAL: CUDA stream create failed\n";
    return false;
  }
  if (!alloc_buffers()) {
    std::cerr << "[PPFormulaNetOrt] FATAL: buffer alloc failed\n";
    return false;
  }
  // FAST: encoder + cross-KV prep + static-KV step, all ORT-CUDA-13 on our stream,
  // driven by a host AR loop. The encoder is bit-exact to the fused in-graph encoder.
  // plus-M ships its split graphs in the model dir (encoder/prep/decoder_step.onnx);
  // PP-FormulaNet-S keeps them in a fast/ subdir (encoder/prep/step_batched.onnx).
  const fs::path enc_p  = plusm_ ? base / "encoder.onnx"      : fast / "encoder.onnx";
  const fs::path prep_p = plusm_ ? base / "prep.onnx"         : fast / "prep.onnx";
  const fs::path step_p = plusm_ ? base / "decoder_step.onnx" : fast / "step_batched.onnx";
  const bool have_all =
      fs::exists(enc_p) && fs::exists(prep_p) && fs::exists(step_p);
  const bool fast_loaded =
      have_all && enc_.load(enc_p.string(), 0, stream_, false)
      && prep_.load(prep_p.string(), 0, stream_, false)
      // NOTE: ORT CUDA-graph capture was tried for the plus-M step (enable_cuda_graph)
      // and does NOT work — it freezes the step's pos-dependent KV scatter at the first
      // step's position, so every replay writes to the wrong slot (0/30 correct), and it
      // gave no speedup anyway (the step is compute/memory-bound, not launch-bound). We
      // keep the persistent-binding run_graph() path (correct + slightly faster) but do
      // NOT enable cuda-graph. The real per-step lever is length-bucketing the KV window.
      && step_.load(step_p.string(), 0, stream_, false, /*enable_cuda_graph=*/false);
  if (!fast_loaded) {
    // The FAST split graphs are REQUIRED — there is no fused fallback. Missing/unloadable
    // graphs mean an incomplete deploy (or FORMULA_ONNX pointing at the wrong model dir);
    // fail loudly rather than silently serving the wrong/slower model (no-silent-failure).
    std::cerr << "[PPFormulaNetOrt] FATAL: FAST graphs missing/unloadable under "
              << (plusm_ ? base.string() : fast.string())
              << " (need encoder.onnx + prep.onnx + "
              << (plusm_ ? "decoder_step.onnx" : "step_batched.onnx")
              << "). The fast/ bundle must ship with the model.\n";
    return false;
  }
  // plus-M length-bucket: load the optional 384-KV-window step. Absent (e.g. fresh
  // deploy) -> decode_continuous_plusm transparently uses the 1056 window for all crops.
  if (plusm_) {
    const fs::path short_p = base / "decoder_step_384.onnx";
    if (fs::exists(short_p) && step_short_.load(short_p.string(), 0, stream_, false))
      std::cerr << "[PPFormulaNetOrt] plus-M length-bucket: 384-KV step loaded\n";
    else
      std::cerr << "[PPFormulaNetOrt] plus-M length-bucket: decoder_step_384.onnx absent — "
                   "1056 window for all crops\n";
  }
  std::cerr << "[PPFormulaNetOrt] FAST decode path ("
            << (plusm_ ? "plus-M 6-layer MBart" : "PP-FormulaNet-S")
            << " encoder+prep+step host-loop)\n";
  ready_ = static_cast<bool>(tok_);  // ready once both model+tokenizer loaded
  return true;
}

bool PPFormulaNetOrt::load_tokenizer(const std::string &path) {
  tok_ = FormulaTokenizer::load(path);
  if (!tok_) { std::cerr << "[PPFormulaNetOrt] tokenizer load failed: " << path << '\n'; return false; }
  if (step_.ready()) ready_ = true;
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
  cudaMemsetAsync(d_done_, 0, (size_t)B, stream_);
  std::vector<char> done(B, 0); std::vector<int64_t> hall; int last = MAXIT; bool ok = true;
  int checked = 0;  // steps already scanned for EOS on device (sticky d_done_)
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
      // On-device EOS scan of the new steps sets sticky per-row flags; only B bytes cross
      // the bus (vs the whole growing token history) and the host does no per-token rescan.
      scan_eos_range<<<(B + 63) / 64, 64, 0, stream_>>>(d_all_, d_done_, B, 3, EOS, checked, it + 1);
      cudaStreamSynchronize(stream_);
      if (cudaError_t e = cudaMemcpy(done.data(), d_done_, (size_t)B, cudaMemcpyDeviceToHost);
          e != cudaSuccess) {
        std::cerr << "[PPFormulaNetOrt] periodic done D2H failed it=" << it
                  << ": " << cudaGetErrorString(e) << '\n';
        last = it + 1; ok = false; break;  // don't early-stop on stale tokens
      }
      checked = it + 1;
      int nd = 0; for (int b = 0; b < B; ++b) nd += done[b] ? 1 : 0;
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
  if (cudaError_t e = cudaMemcpy(hall.data(), d_all_, n * sizeof(int64_t),
                                 cudaMemcpyDeviceToHost); e != cudaSuccess) {
    std::cerr << "[PPFormulaNetOrt] final D2H copy failed: " << cudaGetErrorString(e)
              << " — failing chunk instead of decoding stale tokens\n";
    out.assign(B, {});  // empty seqs + return false -> caller marks crops failed
    return false;
  }
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

// plus-M FAST path: lockstep static-KV decode on the 6-layer MBart step graph.
// decoder_start = </s> (PM_START=2); decoder_step.onnx emits next_token (greedy argmax)
// in-graph, so we thread next_token->tokens device-to-device and only D2H the tiny [B]
// token row every CHECK steps for the EOS-eviction check. pos is shared per step
// (lockstep -> all rows at pos=it). Returns false on a truncating step/CUDA error.
bool PPFormulaNetOrt::decode_chunk_plusm(int B, std::vector<std::vector<int64_t>> &out) {
  const int64_t EOS = tok_->eos_id(), Bi = B;
  const int LY = PM_LAYERS, DH = PM_Dh;
  size_t kvB = (size_t)LY * B * H * MAXLEN * DH * sizeof(float);
  cudaMemsetAsync(kA_, 0, kvB, stream_);
  cudaMemsetAsync(vA_, 0, kvB, stream_);
  fill_val<<<(B + 63) / 64, 64, 0, stream_>>>(d_tok_, PM_START, B);  // decoder start = </s>
  float *kin = kA_, *kout = kB_, *vin = vA_, *vout = vB_;
  std::vector<OrtTensor> ins = {
      {"tokens", d_tok_, {Bi, 1}, true}, {"pos", d_pos_, {Bi}, true},
      {"kb", kin, {LY, Bi, H, MAXLEN, DH}, false}, {"vb", vin, {LY, Bi, H, MAXLEN, DH}, false},
      {"ck", d_ck_, {LY, Bi, H, CTX, DH}, false}, {"cv", d_cv_, {LY, Bi, H, CTX, DH}, false}};
  std::vector<OrtTensor> outs = {
      {"logits", d_log_, {Bi, 1, VOCAB}, false}, {"next_token", d_next_, {Bi, 1}, true},
      {"kb_out", kout, {LY, Bi, H, MAXLEN, DH}, false}, {"vb_out", vout, {LY, Bi, H, MAXLEN, DH}, false}};
  cudaMemsetAsync(d_done_, 0, (size_t)B, stream_);
  std::vector<char> done(B, 0); std::vector<int64_t> hall; int last = PM_MAXIT; bool ok = true;
  int checked = 0;  // steps already scanned for EOS on device (sticky d_done_)
  for (int it = 0; it < PM_MAXIT; ++it) {
    fill_val<<<(B + 63) / 64, 64, 0, stream_>>>(d_pos_, it, B);  // lockstep: all rows at pos=it
    ins[2].data = kin; ins[3].data = vin; outs[2].data = kout; outs[3].data = vout;
    if (!step_.run(ins, outs)) {
      std::cerr << "[PPFormulaNetOrt] plus-M step run failed it=" << it << '\n';
      last = it; ok = false; break;
    }
    cudaMemcpyAsync(d_all_ + (size_t)it * B, d_next_, (size_t)B * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(d_tok_, d_next_, (size_t)B * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream_);
    std::swap(kin, kout); std::swap(vin, vout);
    if ((it + 1) % CHECK == 0) {
      // On-device EOS scan (see decode_chunk): D2H only the B sticky flags, no host rescan.
      scan_eos_range<<<(B + 63) / 64, 64, 0, stream_>>>(d_all_, d_done_, B, 1, EOS, checked, it + 1);
      cudaStreamSynchronize(stream_);
      if (cudaMemcpy(done.data(), d_done_, (size_t)B, cudaMemcpyDeviceToHost) != cudaSuccess) {
        last = it + 1; ok = false; break;
      }
      checked = it + 1;
      int nd = 0; for (int b = 0; b < B; ++b) nd += done[b] ? 1 : 0;
      if (nd == B) { last = it + 1; break; }
    }
  }
  cudaStreamSynchronize(stream_);
  if (cudaPeekAtLastError() != cudaSuccess) ok = false;
  size_t n = (size_t)last * B; hall.resize(n);
  if (cudaMemcpy(hall.data(), d_all_, n * sizeof(int64_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
    out.assign(B, {});
    return false;
  }
  out.assign(B, {});
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < last; ++s) {
      int64_t t = hall[(size_t)s * B + b];
      if (t == EOS) break;
      if (t != 0 && t != 1) out[b].push_back(t);  // drop BOS(0)/pad(1)
    }
  return ok;
}

// plus-M continuous (iteration-level) batched decode over a chosen KV bucket. `step` +
// kA/kB/vA/vB + `maxlen` select the window (the 384 step_short_ for the common case, or the
// 1056 step_ for the long tail). `queue` is the crop indices (into ck_all_, of which n_total
// were prepped) to decode; out[crop] receives each finished sequence; a crop that reaches
// `maxlen` tokens without EOS is appended to `overflow` (its partial out[] cleared) for the
// caller to re-decode in a larger bucket. Fixed KV addresses (no ping-pong; accum_kv_slot
// writes back only the new pos slot). out is NOT resized here — the caller pre-sizes it.
bool PPFormulaNetOrt::decode_continuous_plusm(OrtSession &step, float *kA, float *kB,
                                              float *vA, float *vB, int maxlen, int Bslots,
                                              int n_total, const std::vector<int> &queue,
                                              std::vector<std::vector<int64_t>> &out,
                                              std::vector<int> &overflow, bool final_bucket) {
  const int64_t EOS = tok_->eos_id();
  const int LY = PM_LAYERS, DH = PM_Dh;
  if (Bslots > PM_MAX_BATCH) Bslots = PM_MAX_BATCH;  // ORT step throws at MAX_B=32
  const int QN = (int)queue.size();
  const int64_t Bi = Bslots;
  if (QN <= 0 || Bslots <= 0) return true;

  const size_t blk = (size_t)H * CTX * DH;  // one (layer,crop) cross-KV block
  auto load_cross = [&](int crop, int b) {  // ck_all_/cv_all_[*,crop] -> slot column b
    cudaMemcpy2DAsync(d_ck_ + (size_t)b * blk, (size_t)Bslots * blk * sizeof(float),
                      ck_all_ + (size_t)crop * blk, (size_t)n_total * blk * sizeof(float),
                      blk * sizeof(float), LY, cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpy2DAsync(d_cv_ + (size_t)b * blk, (size_t)Bslots * blk * sizeof(float),
                      cv_all_ + (size_t)crop * blk, (size_t)n_total * blk * sizeof(float),
                      blk * sizeof(float), LY, cudaMemcpyDeviceToDevice, stream_);
  };

  std::vector<int> slot_crop(Bslots, -1);
  std::vector<int64_t> hpos(Bslots, 0), htok(Bslots, PM_START), hnext(Bslots);
  int next_q = 0, active = 0;
  for (int b = 0; b < Bslots && next_q < QN; ++b) {
    slot_crop[b] = queue[next_q]; load_cross(queue[next_q], b); hpos[b] = 0; htok[b] = PM_START;
    ++next_q; ++active;
  }

  // Fixed KV addresses (no ping-pong): the step reads kb=kA/vb=vA and writes the full KV to
  // kb_out=kB/vb_out=vB; accum_kv_slot copies ONLY the new pos slot back into kA/vA.
  //
  // LOAD-BEARING INVARIANT (slot reuse): when a finished crop is evicted and a new one
  // starts in the same slot (hpos=0), the slot's self-KV is NOT cleared — positions
  // beyond the new crop's hpos still hold the PREVIOUS crop's K/V. This is correct only
  // because the exported step masks self-attention causally to [0, pos]: freshly written
  // slots shadow the stale ones before they can ever be attended. If a future export
  // attends the full maxlen window, slot refill MUST memset the slot's KV first.
  step.reset_graph();  // re-bind: Bslots + maxlen (shapes) differ across calls/buckets
  std::vector<OrtTensor> ins = {
      {"tokens", d_tok_, {Bi, 1}, true}, {"pos", d_pos_, {Bi}, true},
      {"kb", kA, {LY, Bi, H, maxlen, DH}, false}, {"vb", vA, {LY, Bi, H, maxlen, DH}, false},
      {"ck", d_ck_, {LY, Bi, H, CTX, DH}, false}, {"cv", d_cv_, {LY, Bi, H, CTX, DH}, false}};
  std::vector<OrtTensor> outs = {
      {"logits", d_log_, {Bi, 1, VOCAB}, false}, {"next_token", d_next_, {Bi, 1}, true},
      {"kb_out", kB, {LY, Bi, H, maxlen, DH}, false}, {"vb_out", vB, {LY, Bi, H, maxlen, DH}, false}};

  bool ok = true;
  const int kvt = LY * Bslots * H * DH;
  long guard = 0, guard_max = (long)QN * maxlen + maxlen;
  while (active > 0 && guard++ < guard_max) {
    cudaMemcpyAsync(d_pos_, hpos.data(), (size_t)Bslots * sizeof(int64_t), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_tok_, htok.data(), (size_t)Bslots * sizeof(int64_t), cudaMemcpyHostToDevice, stream_);
    if (!step.run_graph(ins, outs)) {
      std::cerr << "[plusm-cont] step.run_graph FAILED Bslots=" << Bslots << " maxlen=" << maxlen
                << " cuda=" << cudaGetErrorString(cudaGetLastError()) << '\n';
      ok = false; break;
    }
    accum_kv_slot<<<(kvt + 255) / 256, 256, 0, stream_>>>(kA, kB, d_pos_, LY, Bslots, H, maxlen, DH);
    accum_kv_slot<<<(kvt + 255) / 256, 256, 0, stream_>>>(vA, vB, d_pos_, LY, Bslots, H, maxlen, DH);
    cudaMemcpyAsync(hnext.data(), d_next_, (size_t)Bslots * sizeof(int64_t), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    for (int b = 0; b < Bslots; ++b) {
      if (slot_crop[b] < 0) { htok[b] = PM_START; continue; }
      const int crop = slot_crop[b];
      const int64_t t = hnext[b];
      const bool eos = (t == EOS);
      const bool over = !eos && (hpos[b] + 1 >= maxlen);  // hit the bucket cap without EOS
      // keep the token unless this is a non-final bucket that will re-decode the whole crop.
      if (!eos && t != 0 && t != 1 && (!over || final_bucket)) out[crop].push_back(t);
      if (eos || over) {
        if (over && !final_bucket) { out[crop].clear(); overflow.push_back(crop); }  // re-decode bigger
        --active;
        if (next_q < QN) {
          slot_crop[b] = queue[next_q]; load_cross(queue[next_q], b); hpos[b] = 0; htok[b] = PM_START;
          ++next_q; ++active;
        } else { slot_crop[b] = -1; hpos[b] = 0; htok[b] = PM_START; }
      } else { htok[b] = t; hpos[b] += 1; }
    }
  }
  cudaStreamSynchronize(stream_);
  if (cudaPeekAtLastError() != cudaSuccess) ok = false;
  if (active > 0) {
    // Guard tripped with crops still decoding: their outputs are truncated.
    // Never report that as success (no-silent-failure).
    std::cerr << "[plusm-cont] iteration guard tripped with " << active
              << " crop(s) still active — output truncated\n";
    ok = false;
  }
  return ok;
}

void PPFormulaNetOrt::decode_plusm_page(int N, const std::vector<Box> &boxes, const GpuImage &page,
                                        bool drop_collapse, std::vector<FormulaEngineResult> &out) {
  for (int g0 = 0; g0 < N; g0 += PM_MAX_N) {
    const int GN = std::min(PM_MAX_N, N - g0);
    bool eok = true;
    // Preprocess + encode the group's GN crops into d_mem_all_, in MAX_B-sized batches
    // (host_in_/d_x_ hold at most MAX_B crops). Encoder is per-crop (batch=1) to match ref.
    for (int sb = 0; sb < GN && eok; sb += MAX_B) {
      const int SB = std::min(MAX_B, GN - sb);
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < SB; ++i) {
        auto cr = clamped_crop_rect(boxes[g0 + sb + i], page.cols, page.rows);
        const int x0 = cr[0], y0 = cr[1], w = cr[2], h = cr[3];
        std::vector<uint8_t> tmp((size_t)std::max(1, w) * std::max(1, h) * 3);
        const uint8_t *sp = host_page_.data() + (size_t)y0 * page.step + (size_t)x0 * 3;
        for (int r = 0; r < h; ++r)
          std::memcpy(tmp.data() + (size_t)r * w * 3, sp + (size_t)r * page.step, (size_t)w * 3);
        formula_preprocess_one(tmp.data(), w, h, host_in_.data() + (size_t)i * S * S);
      }
      cudaMemcpyAsync(d_x_, host_in_.data(), (size_t)SB * S * S * sizeof(float),
                      cudaMemcpyHostToDevice, stream_);
      for (int b = 0; b < SB && eok; ++b) {
        OrtTensor ex{"x", d_x_ + (size_t)b * S * S, {1, 1, S, S}, false};
        OrtTensor em{"p2o.pd_op.transpose.0.0", d_mem_all_ + (size_t)(sb + b) * CTX * 2048,
                     {1, CTX, 2048}, false};
        eok = enc_.run({ex}, {em});
      }
      cudaStreamSynchronize(stream_);  // H2D + encode complete before host_in_ is reused
    }
    const int64_t GNi = GN;
    OrtTensor pin{"memory", d_mem_all_, {GNi, CTX, 2048}, false};
    OrtTensor pck{"ck", ck_all_, {PM_LAYERS, GNi, H, CTX, PM_Dh}, false};
    OrtTensor pcv{"cv", cv_all_, {PM_LAYERS, GNi, H, CTX, PM_Dh}, false};
    std::vector<std::vector<int64_t>> seqs(GN);
    std::vector<int> queue(GN), overflow, ovf2;
    for (int i = 0; i < GN; ++i) queue[i] = i;
    bool okg = eok && prep_.run({pin}, {pck, pcv});
    if (okg) {
      // Common case: decode in the 384-KV bucket; formulas that exceed 384 tokens overflow
      // and re-decode in the full 1056 bucket (the final cap). If the 384 graph is absent,
      // everything runs in the 1056 bucket directly.
      const bool have_short = step_short_.ready();
      okg = decode_continuous_plusm(have_short ? step_short_ : step_,
                                    have_short ? kA384_ : kA_, have_short ? kB384_ : kB_,
                                    have_short ? vA384_ : vA_, have_short ? vB384_ : vB_,
                                    have_short ? PM_MAXLEN_S : MAXLEN,
                                    std::min(GN, PM_MAX_BATCH), GN, queue, seqs, overflow,
                                    /*final_bucket=*/!have_short);
      if (okg && !overflow.empty())
        okg = decode_continuous_plusm(step_, kA_, kB_, vA_, vB_, MAXLEN,
                                      std::min((int)overflow.size(), PM_MAX_BATCH), GN, overflow,
                                      seqs, ovf2, /*final_bucket=*/true);
    }
    if (!okg) std::cerr << "[PPFormulaNetOrt] plus-M page decode failed at group " << g0 << '\n';
    for (int i = 0; i < GN; ++i) {
      if (!okg) { out[g0 + i].ok = false; continue; }
      std::string latex = tok_->decode(seqs[i], /*post_process=*/false);
      out[g0 + i].latex =
          (drop_collapse && formula_is_mode_collapsed(seqs[i], latex)) ? std::string() : latex;
      out[g0 + i].token_count = seqs[i].size();
      out[g0 + i].hit_eos = !seqs[i].empty();
      out[g0 + i].ok = true;
    }
  }
}

bool PPFormulaNetOrt::gate_bench(const std::string &gate_dir) {
  if (!fast_ || !step_.ready() || !tok_) {
    std::cerr << "[gate_bench] requires a loaded FAST backend + tokenizer\n";
    return false;
  }
  const fs::path eval(gate_dir);
  const int B = 30;
  std::vector<float> crops((size_t)B * S * S);
  std::ifstream fe((eval / "en15_crops.bin").string(), std::ios::binary);
  std::ifstream fz((eval / "zh15_crops.bin").string(), std::ios::binary);
  if (!fe || !fz) {
    std::cerr << "[gate_bench] gate crops not found under " << eval.string() << '\n';
    return false;
  }
  fe.read(reinterpret_cast<char *>(crops.data()), (size_t)15 * S * S * sizeof(float));
  fz.read(reinterpret_cast<char *>(crops.data() + (size_t)15 * S * S), (size_t)15 * S * S * sizeof(float));
  cudaMemcpyAsync(d_x_, crops.data(), crops.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
  const int64_t Bi = B, LY = plusm_ ? PM_LAYERS : 2, DH = plusm_ ? PM_Dh : Dh;
  auto enc_prep_decode = [&](std::vector<std::vector<int64_t>> &seqs) -> bool {
    bool eok = true;
    for (int b = 0; b < B && eok; ++b) {
      OrtTensor ex{"x", d_x_ + (size_t)b * S * S, {1, 1, S, S}, false};
      OrtTensor em{"p2o.pd_op.transpose.0.0", d_mem_ + (size_t)b * CTX * 2048, {1, CTX, 2048}, false};
      eok = enc_.run({ex}, {em});
    }
    OrtTensor pin{"memory", d_mem_, {Bi, CTX, 2048}, false};
    OrtTensor pck{"ck", d_ck_, {LY, Bi, H, CTX, DH}, false};
    OrtTensor pcv{"cv", d_cv_, {LY, Bi, H, CTX, DH}, false};
    if (!eok || !prep_.run({pin}, {pck, pcv})) return false;
    return plusm_ ? decode_chunk_plusm(B, seqs) : decode_chunk(B, seqs);
  };
  auto esc = [](const std::string &s) {
    std::string o = "\"";
    for (char c : s) {
      if (c == '\\' || c == '"') { o += '\\'; o += c; }
      else if (c == '\n') o += "\\n"; else if (c == '\t') o += "\\t"; else if (c == '\r') {}
      else o += c;
    }
    o += '"'; return o;
  };
  auto dump_tokens = [&](const std::string &path, const std::vector<std::vector<int64_t>> &s) {
    std::ofstream o(path); o << '[';
    for (int b = 0; b < (int)s.size(); ++b) { o << (b ? ",[" : "[");
      for (size_t j = 0; j < s[b].size(); ++j) o << (j ? "," : "") << s[b][j]; o << ']'; }
    o << "]\n";
  };
  std::vector<std::vector<int64_t>> seqs;
  if (!enc_prep_decode(seqs)) { std::cerr << "[gate_bench] enc/prep/decode failed\n"; return false; }
  const std::string base = "/tmp/cpp_" + label_;
  dump_tokens(base + "_tokens.json", seqs);
  { std::ofstream o(base + "_latex.json"); o << '[';
    for (int b = 0; b < B; ++b) o << (b ? "," : "") << esc(tok_->decode(seqs[b], /*post_process=*/false));
    o << "]\n"; }
  // plus-M: also validate the continuous (production) decode path WITH evict/refill (Bslots<B).
  if (plusm_) {
    cudaMemcpyAsync(d_mem_all_, d_mem_, (size_t)B * CTX * 2048 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    OrtTensor pin{"memory", d_mem_all_, {Bi, CTX, 2048}, false};
    OrtTensor pck{"ck", ck_all_, {PM_LAYERS, Bi, H, CTX, PM_Dh}, false};
    OrtTensor pcv{"cv", cv_all_, {PM_LAYERS, Bi, H, CTX, PM_Dh}, false};
    // Validate the continuous decode (WITH evict/refill, Bslots<B) in the production bucket
    // (384 step_short_ when present, else 1056).
    const bool hs = step_short_.ready();
    std::vector<std::vector<int64_t>> cseqs(B);
    std::vector<int> q(B), ovf; for (int i = 0; i < B; ++i) q[i] = i;
    if (prep_.run({pin}, {pck, pcv}) &&
        decode_continuous_plusm(hs ? step_short_ : step_, hs ? kA384_ : kA_, hs ? kB384_ : kB_,
                                hs ? vA384_ : vA_, hs ? vB384_ : vB_, hs ? PM_MAXLEN_S : MAXLEN,
                                16, B, q, cseqs, ovf, /*final_bucket=*/!hs))
      dump_tokens(base + "_cont_tokens.json", cseqs);
    // continuous throughput on an N=90 queue, full Bslots — 1056 window vs the 384 bucket.
    const int N = 3 * B;
    for (int r = 0; r < 3; ++r)
      cudaMemcpyAsync(d_mem_all_ + (size_t)r * B * CTX * 2048, d_mem_,
                      (size_t)B * CTX * 2048 * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
    OrtTensor pin2{"memory", d_mem_all_, {(int64_t)N, CTX, 2048}, false};
    OrtTensor pck2{"ck", ck_all_, {PM_LAYERS, (int64_t)N, H, CTX, PM_Dh}, false};
    OrtTensor pcv2{"cv", cv_all_, {PM_LAYERS, (int64_t)N, H, CTX, PM_Dh}, false};
    if (prep_.run({pin2}, {pck2, pcv2})) {
      std::vector<int> qn(N); for (int i = 0; i < N; ++i) qn[i] = i;
      auto bench = [&](OrtSession &st, float *ka, float *kb, float *va, float *vb, int ml,
                       const char *tag, bool fin) {
        std::vector<std::vector<int64_t>> sc(N); std::vector<int> ov;
        cudaStreamSynchronize(stream_);
        auto c0 = std::chrono::steady_clock::now();
        const int CR = 5;
        for (int r = 0; r < CR; ++r) { for (auto &s : sc) s.clear(); ov.clear();
          decode_continuous_plusm(st, ka, kb, va, vb, ml, PM_MAX_BATCH, N, qn, sc, ov, fin); }
        cudaStreamSynchronize(stream_);
        double cs = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
        std::cerr << "[gate_bench] " << label_ << " continuous N=" << N << " " << tag << ": "
                  << (CR * N / cs) << " crops/s\n";
      };
      bench(step_, kA_, kB_, vA_, vB_, MAXLEN, "[1056 window]", true);
      if (hs) bench(step_short_, kA384_, kB384_, vA384_, vB384_, PM_MAXLEN_S, "[384 bucket]", false);
    }
  }
  std::cerr << "[gate_bench] " << label_ << " wrote " << base << "_*.json (" << B << " crops)\n";
  const int REPS = 10;
  cudaStreamSynchronize(stream_);
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < REPS; ++r) { std::vector<std::vector<int64_t>> s2; enc_prep_decode(s2); }
  cudaStreamSynchronize(stream_);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cerr << "[gate_bench] " << label_ << " lockstep B=30: " << (REPS * B / sec) << " crops/s\n";
  return true;
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
  // plus-M decodes the whole page with continuous (iteration-level) batching — short crops
  // never stall behind long ones, and the batch stays full across the page's whole queue.
  // (-S / fused keep the simpler per-chunk lockstep path below.)
  if (fast_ && plusm_) {
    decode_plusm_page(N, boxes, page, drop_collapse, out);
    return out;
  }
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
    cudaMemcpyAsync(d_x_, host_in_.data(), (size_t)B * S * S * sizeof(float), cudaMemcpyHostToDevice, stream_);
    std::vector<std::vector<int64_t>> seqs(B);   // clean content tokens per crop
    bool chunk_ok = true;                        // false -> mark this chunk's crops failed
    if (fast_) {
      // PP-FormulaNet-S: encode per crop (batch=1 matches the reference) -> d_mem_, cross-KV
      // prep -> ck/cv, then the static-KV host AR loop. (plus-M took the whole-page continuous
      // path above and never reaches here.)
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
