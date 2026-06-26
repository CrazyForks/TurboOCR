#include "turbo_ocr/table/slanext_host_decode.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include "turbo_ocr/table/slanext_postprocess.h"

namespace turbo_ocr::table {

namespace {

// out[j] = sum_i x[i] * W[i*OUT + j] (+ b[j]); W is [IN, OUT] row-major.
inline void mv_io(const float *__restrict__ x, const float *__restrict__ W,
                  int IN, int OUT, const float *__restrict__ b,
                  float *__restrict__ out) {
  for (int j = 0; j < OUT; ++j) out[j] = b ? b[j] : 0.0f;
  for (int i = 0; i < IN; ++i) {
    const float xi = x[i];
    const float *wr = W + static_cast<std::size_t>(i) * OUT;
#pragma omp simd
    for (int j = 0; j < OUT; ++j) out[j] += xi * wr[j];
  }
}

// out[k] = sum_m x[m] * W[k*IN + m] (+ b[k]); W is [OUT, IN] row-major.
inline void mv_oi(const float *__restrict__ x, const float *__restrict__ W,
                  int OUT, int IN, const float *__restrict__ b,
                  float *__restrict__ out) {
  for (int k = 0; k < OUT; ++k) {
    const float *wr = W + static_cast<std::size_t>(k) * IN;
    float s = b ? b[k] : 0.0f;
#pragma omp simd reduction(+ : s)
    for (int m = 0; m < IN; ++m) s += x[m] * wr[m];
    out[k] = s;
  }
}

inline float sigmoidf(float z) { return 1.0f / (1.0f + std::exp(-z)); }

}  // namespace

bool SlanextDecoderWeights::load(const std::string &bin_path) {
  std::ifstream f(bin_path, std::ios::binary);
  if (!f) {
    std::cerr << "[slanext-cpu] cannot open decoder blob: " << bin_path << '\n';
    return false;
  }
  auto rd = [&](std::vector<float> &v, std::size_t n) -> bool {
    v.resize(n);
    f.read(reinterpret_cast<char *>(v.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    return static_cast<bool>(f);
  };
  constexpr int kGruGate = 3 * kHidden;   // 768 (reset|update|candidate)
  constexpr int kGruIn = kCtx + kVocab;   // 146 (ctx + onehot)
  bool ok = rd(lin0_, kCtx * kHidden) && rd(lin1w_, kHidden * kHidden) &&
            rd(lin1b_, kHidden) && rd(lin2_, kHidden * 1) &&
            rd(lin3w_, kHidden * kHidden) && rd(lin3b_, kHidden) &&
            rd(lin4w_, kHidden * kVocab) && rd(lin4b_, kVocab) &&
            rd(lin5w_, kHidden * kHidden) && rd(lin5b_, kHidden) &&
            rd(lin6w_, kHidden * kLoc) && rd(lin6b_, kLoc) &&
            rd(gw0_, kGruGate * kGruIn) && rd(gw1_, kGruGate * kHidden) &&
            rd(gb0_, kGruGate) && rd(gb1_, kGruGate);
  if (!ok) {
    std::cerr << "[slanext-cpu] decoder blob truncated: " << bin_path << '\n';
    return false;
  }
  if (f.peek() != std::ifstream::traits_type::eof()) {
    std::cerr << "[slanext-cpu] decoder blob has trailing bytes (wrong file?): "
              << bin_path << '\n';
    return false;
  }
  return true;
}

StructureResult slanext_host_decode(const float *feat,
                                    const SlanextDecoderWeights &W,
                                    const CharDict &dict, int ori_w, int ori_h) {
  using w = SlanextDecoderWeights;
  const int eos = static_cast<int>(dict.eos_idx());

  std::vector<float> bHp(static_cast<std::size_t>(w::kTenc) * w::kHidden);
  for (int i = 0; i < w::kTenc; ++i)
    mv_io(feat + static_cast<std::size_t>(i) * w::kCtx, W.lin0_.data(), w::kCtx,
          w::kHidden, nullptr,
          bHp.data() + static_cast<std::size_t>(i) * w::kHidden);

  std::vector<float> logits, locs;
  logits.reserve(static_cast<std::size_t>(w::kMaxTokens) * w::kVocab);
  locs.reserve(static_cast<std::size_t>(w::kMaxTokens) * w::kLoc);

  std::vector<float> h(w::kHidden, 0.0f), hp(w::kHidden), a(w::kTenc),
      ctx(w::kCtx), gin(3 * w::kHidden), ghn(3 * w::kHidden), s1(w::kHidden),
      st(w::kVocab), l1(w::kHidden), l8(w::kLoc);
  constexpr int kGruIn = w::kCtx + w::kVocab;
  constexpr int kMaxRunRepeat = 96;
  int prev = 0;
  int steps = 0;
  int run_tok = -1, run_len = 0;
  for (int t = 0; t < w::kMaxTokens; ++t) {
    mv_io(h.data(), W.lin1w_.data(), w::kHidden, w::kHidden, W.lin1b_.data(),
          hp.data());
    for (int i = 0; i < w::kTenc; ++i) {
      const float *br = bHp.data() + static_cast<std::size_t>(i) * w::kHidden;
      float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
      for (int j = 0; j < w::kHidden; ++j) acc += std::tanh(br[j] + hp[j]) * W.lin2_[j];
      a[i] = acc;
    }
    float emax = a[0];
    for (int i = 1; i < w::kTenc; ++i) emax = std::max(emax, a[i]);
    float esum = 0.0f;
    for (int i = 0; i < w::kTenc; ++i) { a[i] = std::exp(a[i] - emax); esum += a[i]; }
    const float inv = 1.0f / esum;
    std::fill(ctx.begin(), ctx.end(), 0.0f);
    for (int i = 0; i < w::kTenc; ++i) {
      const float ai = a[i] * inv;
      const float *fr = feat + static_cast<std::size_t>(i) * w::kCtx;
#pragma omp simd
      for (int c = 0; c < w::kCtx; ++c) ctx[c] += ai * fr[c];
    }
    for (int k = 0; k < 3 * w::kHidden; ++k) {
      const float *wr = W.gw0_.data() + static_cast<std::size_t>(k) * kGruIn;
      float s = W.gb0_[k];
#pragma omp simd reduction(+ : s)
      for (int m = 0; m < w::kCtx; ++m) s += ctx[m] * wr[m];
      gin[k] = s + wr[w::kCtx + prev];
    }
    mv_oi(h.data(), W.gw1_.data(), 3 * w::kHidden, w::kHidden, W.gb1_.data(),
          ghn.data());
    for (int j = 0; j < w::kHidden; ++j) {
      const float r = sigmoidf(gin[j] + ghn[j]);
      const float z = sigmoidf(gin[w::kHidden + j] + ghn[w::kHidden + j]);
      const float n =
          std::tanh(gin[2 * w::kHidden + j] + r * ghn[2 * w::kHidden + j]);
      h[j] = (1.0f - z) * n + z * h[j];
    }
    mv_io(h.data(), W.lin3w_.data(), w::kHidden, w::kHidden, W.lin3b_.data(),
          s1.data());
    mv_io(s1.data(), W.lin4w_.data(), w::kHidden, w::kVocab, W.lin4b_.data(),
          st.data());
    int best = 0; float bv = st[0];
    for (int v = 1; v < w::kVocab; ++v) if (st[v] > bv) { bv = st[v]; best = v; }
    if (best == eos) break;
    if (best == run_tok) {
      if (++run_len >= kMaxRunRepeat) break;
    } else {
      run_tok = best;
      run_len = 1;
    }
    logits.insert(logits.end(), st.begin(), st.end());
    mv_io(h.data(), W.lin5w_.data(), w::kHidden, w::kHidden, W.lin5b_.data(),
          l1.data());
    mv_io(l1.data(), W.lin6w_.data(), w::kHidden, w::kLoc, W.lin6b_.data(),
          l8.data());
    for (int k = 0; k < w::kLoc; ++k) locs.push_back(sigmoidf(l8[k]));
    ++steps;
    prev = best;
  }

  std::vector<float> probs(logits.size());
  for (int t = 0; t < steps; ++t) {
    const float *lr = logits.data() + static_cast<std::size_t>(t) * w::kVocab;
    float *pr = probs.data() + static_cast<std::size_t>(t) * w::kVocab;
    float mx = lr[0];
    for (int v = 1; v < w::kVocab; ++v) mx = std::max(mx, lr[v]);
    float sm = 0.0f;
    for (int v = 0; v < w::kVocab; ++v) { pr[v] = std::exp(lr[v] - mx); sm += pr[v]; }
    const float iv = 1.0f / sm;
    for (int v = 0; v < w::kVocab; ++v) pr[v] *= iv;
  }

  return decode_structure(probs.data(), locs.data(),
                          static_cast<std::size_t>(steps),
                          static_cast<std::size_t>(w::kVocab), dict,
                          w::kInputSize, w::kInputSize, ori_w, ori_h);
}

}  // namespace turbo_ocr::table
