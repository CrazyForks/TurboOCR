#include "turbo_ocr/formula/ppformulanet_preprocess.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

namespace turbo_ocr::formula {

namespace {
constexpr int S = kFormulaInputSize;
constexpr float MEAN = kFormulaMean, STD = kFormulaStd, PAD_VAL = kFormulaPadVal;

// PIL.Image.resize(BICUBIC) — a SEPARABLE scaled-bicubic resample with antialiasing
// on downscale (filter support grows with the downsample factor). cv2.INTER_CUBIC
// does NOT antialias, so for the wide->short formula crops it diverges from the
// reference by ~1.6/255, wrecking the encoder input. This is a BIT-EXACT port of
// PIL's algorithm: double coeffs (a=-0.5, support 2.0, filterscale=max(1,in/out))
// quantized to 22-bit fixed point, integer convolution, two uint8 passes. Float
// weights drifted ~0.02/px on most crops and the greedy decode amplified it.
constexpr int PREC = 22;  // PIL's PRECISION_BITS = 32 - 8 - 2
inline double pil_cubic(double x) {
  const double a = -0.5; x = std::fabs(x);
  if (x < 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1.0;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}
// One separable pass `in_len`->`out_len`: emit (out idx, first src idx, int coeffs).
template <class Store>
void pil_axis(int out_len, int in_len, Store store) {
  double scale = (double)in_len / out_len, fscale = std::max(1.0, scale), support = 2.0 * fscale;
  std::vector<int64_t> kk;
  for (int xx = 0; xx < out_len; ++xx) {
    double center = (xx + 0.5) * scale;
    int lo = (int)(center - support + 0.5); if (lo < 0) lo = 0;
    int hi = (int)(center + support + 0.5); if (hi > in_len) hi = in_len;
    int n = hi - lo;
    std::vector<double> w(n); double sum = 0.0;
    for (int k = 0; k < n; ++k) { w[k] = pil_cubic((lo + k - center + 0.5) / fscale); sum += w[k]; }
    kk.resize(n);
    for (int k = 0; k < n; ++k) {                            // normalize then quantize
      double v = w[k] / sum;
      kk[k] = (int64_t)(v >= 0 ? v * (1 << PREC) + 0.5 : v * (1 << PREC) - 0.5);
    }
    store(xx, lo, kk);
  }
}
// PIL-bicubic resize of a BGR8 crop -> CV_8UC3, two separable uint8 passes with
// PIL's exact integer convolution (ss = 2^21 + Σ pixel*coeff; out = clip8(ss>>22)).
cv::Mat pil_resize(const cv::Mat &src, int nw, int nh) {
  const int64_t HALF = (int64_t)1 << (PREC - 1);
  auto clip8 = [](int64_t ss) -> uint8_t { ss >>= PREC; return (uint8_t)(ss < 0 ? 0 : ss > 255 ? 255 : ss); };
  cv::Mat tmp(src.rows, nw, CV_8UC3);  // horizontal pass -> uint8
  pil_axis(nw, src.cols, [&](int xx, int lo, const std::vector<int64_t> &w) {
    for (int y = 0; y < src.rows; ++y) {
      const uint8_t *row = src.ptr<uint8_t>(y); int64_t b = HALF, g = HALF, r = HALF;
      for (size_t k = 0; k < w.size(); ++k) { const uint8_t *p = row + (lo + (int)k) * 3; b += w[k] * p[0]; g += w[k] * p[1]; r += w[k] * p[2]; }
      uint8_t *o = tmp.ptr<uint8_t>(y) + xx * 3; o[0] = clip8(b); o[1] = clip8(g); o[2] = clip8(r);
    }
  });
  cv::Mat out(nh, nw, CV_8UC3);  // vertical pass -> uint8
  pil_axis(nh, src.rows, [&](int yy, int lo, const std::vector<int64_t> &w) {
    for (int x = 0; x < nw; ++x) {
      int64_t b = HALF, g = HALF, r = HALF;
      for (size_t k = 0; k < w.size(); ++k) { const uint8_t *p = tmp.ptr<uint8_t>(lo + (int)k) + x * 3; b += w[k] * p[0]; g += w[k] * p[1]; r += w[k] * p[2]; }
      uint8_t *o = out.ptr<uint8_t>(yy) + x * 3; o[0] = clip8(b); o[1] = clip8(g); o[2] = clip8(r);
    }
  });
  return out;
}
}  // namespace

// Mirror the Python sidecar preprocess: margin-crop -> AR-preserving PIL-bicubic
// resize (longer side 384) -> /255, (x-mean)/std on BGR floats -> BGR2GRAY ->
// center-pad to 384 with the normalized-zero pad value.
void formula_preprocess_one(const uint8_t *bgr, int w, int h, float *out) {
  if (w <= 0 || h <= 0) { std::fill(out, out + S * S, PAD_VAL); return; }
  cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t *>(bgr));
  cv::Mat gray; cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  double mn, mx; cv::minMaxLoc(gray, &mn, &mx);
  cv::Mat cropped = src;
  if (mx > mn) {
    cv::Mat dataf; gray.convertTo(dataf, CV_32F);
    dataf = (dataf - mn) / std::max(1.0, mx - mn) * 255.0;
    cv::Mat mask = dataf < 200.0f;  // CV_8U 255/0
    std::vector<cv::Point> nz; cv::findNonZero(mask, nz);
    if (!nz.empty()) {
      cv::Rect r = cv::boundingRect(nz);
      if (r.width > 0 && r.height > 0) cropped = src(r);
    }
  }
  // Python round() is banker's rounding (ties-to-even); std::lrint in the default
  // FE_TONEAREST mode matches it (std::lround would tie-away and shift a side by 1px).
  int W = cropped.cols, Hh = cropped.rows, nw, nh;
  if (W <= Hh) { nw = S; nh = std::max(1, (int)std::lrint((double)S * Hh / W)); }
  else         { nh = S; nw = std::max(1, (int)std::lrint((double)S * W / Hh)); }
  if (nh > S) { nw = std::max(1, (int)std::lrint((double)nw * S / nh)); nh = S; }
  if (nw > S) { nh = std::max(1, (int)std::lrint((double)nh * S / nw)); nw = S; }
  cv::Mat resized = pil_resize(cropped, nw, nh);  // CV_8UC3, PIL-bicubic (uint8 per pass)
  cv::Mat bgrf; resized.convertTo(bgrf, CV_32F, 1.0 / 255.0);
  bgrf = (bgrf - MEAN) / STD;
  cv::Mat grayf; cv::cvtColor(bgrf, grayf, cv::COLOR_BGR2GRAY);
  int dh = S - nh, dw = S - nw, top = dh / 2, bot = dh - top, left = dw / 2, right = dw - left;
  cv::Mat padded;
  cv::copyMakeBorder(grayf, padded, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(PAD_VAL));
  if (padded.isContinuous()) std::memcpy(out, padded.ptr<float>(), (size_t)S * S * sizeof(float));
  else for (int r = 0; r < S; ++r) std::memcpy(out + (size_t)r * S, padded.ptr<float>(r), (size_t)S * sizeof(float));
}

// Mirror the sidecar's _is_mode_collapsed: the 3-gram uniqueness test catches a
// repetitive runaway run without dropping a legitimate long matrix.
bool formula_is_mode_collapsed(const std::vector<int64_t> &toks, const std::string &latex) {
  if (toks.size() >= 240) return true;
  if (latex.size() >= 1500) return true;
  if (toks.size() >= 50) {
    std::set<std::array<int64_t, 3>> uniq;
    const int n = static_cast<int>(toks.size()) - 2;
    for (int i = 0; i < n; ++i) uniq.insert({toks[i], toks[i + 1], toks[i + 2]});
    if (n > 0 && static_cast<double>(uniq.size()) / n < 0.25) return true;
  }
  return false;
}

}  // namespace turbo_ocr::formula
