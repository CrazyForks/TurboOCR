#include "turbo_ocr/table/slanext_enc_split.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/table/slanext_host_decode.h"
#include "turbo_ocr/table/slanext_postprocess.h"

namespace turbo_ocr::table {

using engine::TrtEngine;

// The GRU+attention host decoder is implemented twice (this GPU TU's inline
// loop in SlanextEncSplit::infer and the CPU sibling slanext_host_decode), each
// with its own copy of these arch constants. Bind the two sets so any drift
// fails the build instead of silently diverging GPU vs CPU table structure
// output. (The decode-loop duplication itself is tracked separately.)
static_assert(SlanextEncSplit::kTenc      == SlanextDecoderWeights::kTenc);
static_assert(SlanextEncSplit::kCtx       == SlanextDecoderWeights::kCtx);
static_assert(SlanextEncSplit::kHidden    == SlanextDecoderWeights::kHidden);
static_assert(SlanextEncSplit::kVocab     == SlanextDecoderWeights::kVocab);
static_assert(SlanextEncSplit::kLoc       == SlanextDecoderWeights::kLoc);
static_assert(SlanextEncSplit::kMaxTokens == SlanextDecoderWeights::kMaxTokens);
static_assert(SlanextEncSplit::kInputSize == SlanextDecoderWeights::kInputSize);

namespace {

// out[j] = sum_i x[i] * W[i*OUT + j] (+ b[j]); W is [IN, OUT] row-major.
// Inputs are dense activations (no zero-skip — it would just mispredict per
// element). The inner j-loop auto-vectorizes (contiguous wr[j]/out[j]).
inline void mv_io(const float* __restrict__ x, const float* __restrict__ W,
                  int IN, int OUT, const float* __restrict__ b,
                  float* __restrict__ out) {
  for (int j = 0; j < OUT; ++j) out[j] = b ? b[j] : 0.0f;
  for (int i = 0; i < IN; ++i) {
    const float xi = x[i];
    const float* wr = W + static_cast<std::size_t>(i) * OUT;
#pragma omp simd
    for (int j = 0; j < OUT; ++j) out[j] += xi * wr[j];
  }
}

// out[k] = sum_m x[m] * W[k*IN + m] (+ b[k]); W is [OUT, IN] row-major.
// The m-loop is a dot-product reduction — omp simd + the TU's -ffast-math let
// the compiler reassociate it into AVX2 FMA accumulators (the GRU hot path).
inline void mv_oi(const float* __restrict__ x, const float* __restrict__ W,
                  int OUT, int IN, const float* __restrict__ b,
                  float* __restrict__ out) {
  for (int k = 0; k < OUT; ++k) {
    const float* wr = W + static_cast<std::size_t>(k) * IN;
    float s = b ? b[k] : 0.0f;
#pragma omp simd reduction(+ : s)
    for (int m = 0; m < IN; ++m) s += x[m] * wr[m];
    out[k] = s;
  }
}

inline float sigmoidf(float z) { return 1.0f / (1.0f + std::exp(-z)); }

}  // namespace

SlanextEncSplit::~SlanextEncSplit() noexcept {
  if (d2h_event_) cudaEventDestroy(d2h_event_);
}

bool SlanextEncSplit::discover_tensor_names() {
  const auto& ins = engine_->input_names();
  const auto& outs = engine_->output_names();
  if (ins.empty() || outs.empty()) {
    std::cerr << "[slanext-split] missing input/output tensors\n";
    return false;
  }
  input_name_ = ins.front();
  output_name_ = outs.front();
  return true;
}

bool SlanextEncSplit::init_buffers() {
  d_image_.reset(static_cast<std::size_t>(1) * 3 * kInputSize * kInputSize);
  d_feat_.reset(static_cast<std::size_t>(1) * kTenc * kCtx);
  h_feat_.reset(static_cast<std::size_t>(1) * kTenc * kCtx);
  engine_->set_tensor_address(input_name_, d_image_.get());
  engine_->set_tensor_address(output_name_, d_feat_.get());
  CUDA_CHECK(cudaEventCreateWithFlags(&d2h_event_, cudaEventDisableTiming));
  return true;
}

bool SlanextEncSplit::load_decoder(const std::string& bin_path) {
  std::ifstream f(bin_path, std::ios::binary);
  if (!f) {
    std::cerr << "[slanext-split] cannot open decoder blob: " << bin_path << '\n';
    return false;
  }
  auto rd = [&](std::vector<float>& v, std::size_t n) -> bool {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    return static_cast<bool>(f);
  };
  // Fixed order (matches the Python extractor). Sizes use the named arch
  // constants so a blob/constant drift is caught at the read, not silently.
  constexpr int kGruGate = 3 * kHidden;        // 768 (reset|update|candidate)
  constexpr int kGruIn = kCtx + kVocab;        // 146 (ctx + onehot)
  bool ok = rd(lin0_, kCtx * kHidden) && rd(lin1w_, kHidden * kHidden) &&
            rd(lin1b_, kHidden) && rd(lin2_, kHidden * 1) &&
            rd(lin3w_, kHidden * kHidden) && rd(lin3b_, kHidden) &&
            rd(lin4w_, kHidden * kVocab) && rd(lin4b_, kVocab) &&
            rd(lin5w_, kHidden * kHidden) && rd(lin5b_, kHidden) &&
            rd(lin6w_, kHidden * kLoc) && rd(lin6b_, kLoc) &&
            rd(gw0_, kGruGate * kGruIn) && rd(gw1_, kGruGate * kHidden) &&
            rd(gb0_, kGruGate) && rd(gb1_, kGruGate);
  if (!ok) {
    std::cerr << "[slanext-split] decoder blob truncated: " << bin_path << '\n';
    return false;
  }
  // Reject a too-LARGE / mismatched blob (truncation is caught above): the
  // 16 reads must consume the whole file, else it's the wrong weights.
  if (f.peek() != std::ifstream::traits_type::eof()) {
    std::cerr << "[slanext-split] decoder blob has trailing bytes (wrong file?): "
              << bin_path << '\n';
    return false;
  }
  return true;
}

bool SlanextEncSplit::load_model(const std::string& encoder_trt_path,
                                 const std::string& decoder_bin_path,
                                 const std::string& dict_path) {
  engine_ = std::make_unique<TrtEngine>(encoder_trt_path);
  if (!engine_->load()) {
    std::cerr << "[slanext-split] failed to load encoder engine: "
              << encoder_trt_path << '\n';
    return false;
  }
  if (!discover_tensor_names()) return false;
  if (!init_buffers()) return false;
  if (!load_decoder(decoder_bin_path)) return false;
  try {
    dict_ = dict_path.empty() ? default_dict() : CharDict::from_path(dict_path);
  } catch (const std::exception& e) {
    std::cerr << "[slanext-split] dict load failed: " << e.what() << '\n';
    return false;
  }
  return true;
}

StructureResult SlanextEncSplit::infer(const GpuImage& page, const Box& region,
                                       cudaStream_t stream) {
  StructureResult empty{};
  if (!engine_) return empty;

  auto bb = aabb(region);
  const int rx = std::clamp(bb[0], 0, page.cols - 1);
  const int ry = std::clamp(bb[1], 0, page.rows - 1);
  const int rw = std::clamp(bb[2] - rx, 1, page.cols - rx);
  const int rh = std::clamp(bb[3] - ry, 1, page.rows - ry);
  if (rw <= 0 || rh <= 0) return empty;

  // Encoder: ResizeByLong-488 + ImageNet norm + pad (cuda_fused_slanext_pre),
  // then one TRT execute -> feature [1, 256, 96].
  kernels::cuda_fused_slanext_pre_rgb(page, rx, ry, rw, rh, d_image_.get(), stream);
  nvinfer1::Dims4 in_dims{1, 3, kInputSize, kInputSize};
  if (!engine_->set_input_shape(input_name_, in_dims)) {
    std::cerr << "[slanext-split] encoder set_input_shape failed for region ["
              << rx << ',' << ry << ',' << rw << 'x' << rh
              << "] — table region DROPPED\n";
    return empty;
  }
  if (!engine_->execute(stream)) {
    std::cerr << "[slanext-split] encoder execute failed for region ["
              << rx << ',' << ry << ',' << rw << 'x' << rh
              << "] — table region DROPPED\n";
    return empty;
  }
  CUDA_CHECK(cudaMemcpyAsync(h_feat_.get(), d_feat_.get(),
                             sizeof(float) * kTenc * kCtx,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(d2h_event_, stream));
  CUDA_CHECK(cudaEventSynchronize(d2h_event_));

  // Host AR decode (GRU + additive attention + structure/loc heads).
  const float* feat = h_feat_.get();  // [T_enc=256, C=96] row-major
  const int eos = static_cast<int>(dict_.eos_idx());

  // bHp[T_enc][hidden] = feat @ linear_0 (i2h, 96->256, no bias)
  std::vector<float> bHp(static_cast<std::size_t>(kTenc) * kHidden);
  for (int i = 0; i < kTenc; ++i)
    mv_io(feat + static_cast<std::size_t>(i) * kCtx, lin0_.data(), kCtx, kHidden,
          nullptr, bHp.data() + static_cast<std::size_t>(i) * kHidden);

  std::vector<float> logits;  // [T,50] RAW structure logits (softmaxed post-loop)
  std::vector<float> locs;    // [T,8]
  logits.reserve(static_cast<std::size_t>(kMaxTokens) * kVocab);
  locs.reserve(static_cast<std::size_t>(kMaxTokens) * kLoc);

  std::vector<float> h(kHidden, 0.0f), hp(kHidden), a(kTenc), ctx(kCtx),
      gin(3 * kHidden), ghn(3 * kHidden), s1(kHidden), st(kVocab), l1(kHidden),
      l8(kLoc);
  constexpr int kGruIn = kCtx + kVocab;  // GRU input width (ctx + onehot)
  // Degenerate-decode guard: cap how many times one structure token may repeat
  // back-to-back. Legit runs (a wide row of <td></td>) top out near max table
  // width (~30-40 cols), so 96 never fires on real tables but truncates a
  // runaway that would otherwise spin to the 501 cap on pathological input.
  constexpr int kMaxRunRepeat = 96;
  int prev = 0;  // sos
  int steps = 0;
  int run_tok = -1, run_len = 0;
  for (int t = 0; t < kMaxTokens; ++t) {
    mv_io(h.data(), lin1w_.data(), kHidden, kHidden, lin1b_.data(), hp.data());
    // additive attention energy a[i] = score(tanh(bHp[i] + hp)); softmax over T_enc
    for (int i = 0; i < kTenc; ++i) {
      const float* br = bHp.data() + static_cast<std::size_t>(i) * kHidden;
      float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
      for (int j = 0; j < kHidden; ++j) acc += std::tanh(br[j] + hp[j]) * lin2_[j];
      a[i] = acc;
    }
    float emax = a[0];
    for (int i = 1; i < kTenc; ++i) emax = std::max(emax, a[i]);
    float esum = 0.0f;
    for (int i = 0; i < kTenc; ++i) { a[i] = std::exp(a[i] - emax); esum += a[i]; }
    const float inv = 1.0f / esum;
    // ctx = sum_i a[i] * feat[i]
    std::fill(ctx.begin(), ctx.end(), 0.0f);
    for (int i = 0; i < kTenc; ++i) {
      const float ai = a[i] * inv;
      const float* fr = feat + static_cast<std::size_t>(i) * kCtx;
#pragma omp simd
      for (int c = 0; c < kCtx; ++c) ctx[c] += ai * fr[c];
    }
    // GRU cell, gate order (reset, update, candidate). The 146-wide input is
    // ctx(96, dense) ++ onehot(prev): fold the onehot to a single column add
    // instead of a 50-wide multiply-by-zeros.
    for (int k = 0; k < 3 * kHidden; ++k) {
      const float* wr = gw0_.data() + static_cast<std::size_t>(k) * kGruIn;
      float s = gb0_[k];
#pragma omp simd reduction(+ : s)
      for (int m = 0; m < kCtx; ++m) s += ctx[m] * wr[m];
      gin[k] = s + wr[kCtx + prev];  // onehot: the single nonzero input dim
    }
    mv_oi(h.data(), gw1_.data(), 3 * kHidden, kHidden, gb1_.data(), ghn.data());
    for (int j = 0; j < kHidden; ++j) {
      const float r = sigmoidf(gin[j] + ghn[j]);
      const float z = sigmoidf(gin[kHidden + j] + ghn[kHidden + j]);
      const float n =
          std::tanh(gin[2 * kHidden + j] + r * ghn[2 * kHidden + j]);
      h[j] = (1.0f - z) * n + z * h[j];
    }
    // structure head (no activation between the two linears) -> raw logits
    mv_io(h.data(), lin3w_.data(), kHidden, kHidden, lin3b_.data(), s1.data());
    mv_io(s1.data(), lin4w_.data(), kHidden, kVocab, lin4b_.data(), st.data());
    // argmax on RAW logits (softmax is monotone -> identical argmax); defer the
    // softmax out of the sequential critical path.
    int best = 0; float bv = st[0];
    for (int v = 1; v < kVocab; ++v) if (st[v] > bv) { bv = st[v]; best = v; }
    if (best == eos) break;  // EOS: stop before storing/loc-head (it's discarded)
    // Runaway guard: a token repeating back-to-back past any legit table width
    // means the decode has collapsed — truncate to the clean prefix already
    // emitted instead of spinning to the 501 cap on garbage.
    if (best == run_tok) {
      if (++run_len >= kMaxRunRepeat) break;
    } else {
      run_tok = best;
      run_len = 1;
    }
    logits.insert(logits.end(), st.begin(), st.end());
    // loc head
    mv_io(h.data(), lin5w_.data(), kHidden, kHidden, lin5b_.data(), l1.data());
    mv_io(l1.data(), lin6w_.data(), kHidden, kLoc, lin6b_.data(), l8.data());
    for (int k = 0; k < kLoc; ++k) locs.push_back(sigmoidf(l8[k]));
    ++steps;
    prev = best;
  }

  // Softmax the stored logits once (off the per-step critical path) so
  // decode_structure keeps an identical per-token confidence score.
  std::vector<float> probs(logits.size());
  for (int t = 0; t < steps; ++t) {
    const float* lr = logits.data() + static_cast<std::size_t>(t) * kVocab;
    float* pr = probs.data() + static_cast<std::size_t>(t) * kVocab;
    float mx = lr[0];
    for (int v = 1; v < kVocab; ++v) mx = std::max(mx, lr[v]);
    float sm = 0.0f;
    for (int v = 0; v < kVocab; ++v) { pr[v] = std::exp(lr[v] - mx); sm += pr[v]; }
    const float iv = 1.0f / sm;
    for (int v = 0; v < kVocab; ++v) pr[v] *= iv;
  }

  return decode_structure(probs.data(), locs.data(),
                          static_cast<std::size_t>(steps),
                          static_cast<std::size_t>(kVocab), dict_, kInputSize,
                          kInputSize, rw, rh);
}

}  // namespace turbo_ocr::table
