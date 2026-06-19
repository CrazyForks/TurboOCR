#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace turbo_ocr::detection {

// Bounds for the engine optimization-profile MAX side. Kept symmetric with the
// integer-overflow safeguard in paddle_det.cpp (`max_pixels = kMaxSideLen_²`).
inline constexpr int kDetMaxSideMin = 32;
inline constexpr int kDetMaxSideMax = 4096;

// Per-model detection resize policy (PaddleOCR DetResizeForTest, resize_image_type0).
// PaddleX OCR pipeline default for every PP-OCRv6 det tier (inference.yml has
// "DetResizeForTest: null"): limit_type="min", limit_side_len=64, max_side_limit=4000.
//   - "min": grow the SHORTER side up to limit_side_len.
//   - "max": shrink the LONGER side down to limit_side_len.
// Both clamp the final output so max(resize_h,resize_w) <= max_side_limit.
struct DetResizeParams {
  const char* limit_type;  // "min" or "max"
  int limit_side_len;
  int max_side_limit;
};

// PaddleOCR's official OCR pipeline default is {"min", 64, 4000}. The 4000 cap
// is sized for single-image CLI use; this server pre-allocates det buffers at
// max_side_limit² per pooled pipeline, so 4000²×pool OOMs a 32 GB card. 1280
// runs the vast majority of documents at native resolution (the part that
// matters) while fitting the pool. Raise via DET_MAX_SIDE_LIMIT/DET_MAX_SIDE.
inline constexpr DetResizeParams kDetResizeDefault{"min", 64, 1280};

// DB post-processing parameters (PP-OCRv6 detection defaults). Per-model: only
// box_thresh differs across tiers (0.45 medium/small, 0.40 tiny). Shared by both
// detector paths so the GPU and CPU detectors can never disagree.
struct DbParams {
  float thresh;        // pixel-map binarization threshold
  float box_thresh;    // per-box mean-score cutoff
  float unclip_ratio;  // polygon expansion ratio
};

inline constexpr DbParams kDbDefaults{0.2f, 0.45f, 1.4f};

// Round to the nearest multiple of 32 (det engine requires /32 input dims),
// with a floor of 32. Matches PaddleOCR resize_image_type0's rounding.
[[nodiscard]] inline int round32_floor(double v) {
  return std::max(static_cast<int>(std::round(v / 32.0) * 32), 32);
}

// Shared resize computation for all three call sites (CPU + GPU single + GPU
// batch). Implements PaddleOCR resize_image_type0 for both limit policies plus
// the max_side_limit cap, returning the /32-rounded output dims. Returns
// (resize_h, resize_w).
[[nodiscard]] inline std::pair<int, int> compute_det_resize(int h, int w,
                                                            const DetResizeParams& p) {
  const int L = p.limit_side_len;
  float ratio = 1.0f;
  if (p.limit_type[0] == 'm' && p.limit_type[1] == 'a') {  // "max"
    const int longest = std::max(h, w);
    if (longest > L) ratio = static_cast<float>(L) / longest;
  } else {  // "min"
    const int shortest = std::min(h, w);
    if (shortest < L) ratio = static_cast<float>(L) / shortest;
  }

  int resize_h = round32_floor(h * ratio);
  int resize_w = round32_floor(w * ratio);

  // Cap the longer output side at max_side_limit, then re-round to /32.
  const int longest_out = std::max(resize_h, resize_w);
  if (longest_out > p.max_side_limit) {
    const double rescale = static_cast<double>(p.max_side_limit) / longest_out;
    resize_h = round32_floor(resize_h * rescale);
    resize_w = round32_floor(resize_w * rescale);
  }
  return {resize_h, resize_w};
}

// Apply env overrides to a per-model resize base. Each field is overridden ONLY
// when its env var is present, so a per-model value is never clobbered by a
// default. DET_MAX_SIDE (the single-knob effective-engine-max override consumed
// by effective_det_max_side()) is also folded in here so the resize cap can
// never exceed the engine profile / pinned buffers.
//   DET_LIMIT_TYPE / DET_LIMIT_SIDE_LEN / DET_MAX_SIDE_LIMIT / DET_MAX_SIDE
[[nodiscard]] inline DetResizeParams read_det_resize(DetResizeParams base = kDetResizeDefault) {
  if (const char* env = std::getenv("DET_LIMIT_TYPE"))
    base.limit_type = (env[0] == 'm' && env[1] == 'a') ? "max" : "min";
  if (const char* env = std::getenv("DET_LIMIT_SIDE_LEN"); env && *env)
    base.limit_side_len = std::atoi(env);
  if (const char* env = std::getenv("DET_MAX_SIDE_LIMIT"); env && *env)
    base.max_side_limit = std::atoi(env);
  // DET_MAX_SIDE sizes the TRT profile MAX + pinned buffers via
  // effective_det_max_side(); it must ALSO cap the resize output, or a
  // DET_MAX_SIDE below max_side_limit (e.g. 640 < 1280) lets compute_det_resize
  // emit up to max_side_limit px into a buffer/profile sized for the smaller
  // value — overrun. Fold it in as a shrink-only min (enlarging is safe: the
  // buffer is sized to the larger effective max).
  if (const char* env = std::getenv("DET_MAX_SIDE"); env && *env)
    base.max_side_limit =
        std::min(base.max_side_limit, std::clamp(std::atoi(env), kDetMaxSideMin, kDetMaxSideMax));
  return base;
}

// Apply env overrides to a per-model DB base. Each field is overridden ONLY when
// its env var is present, so a per-model value is never clobbered by a default.
//   DET_DB_THRESH / DET_BOX_THRESH / DET_UNCLIP
[[nodiscard]] inline DbParams read_db_params(DbParams base = kDbDefaults) {
  if (const char* env = std::getenv("DET_DB_THRESH"); env && *env)
    base.thresh = static_cast<float>(std::atof(env));
  if (const char* env = std::getenv("DET_BOX_THRESH"); env && *env)
    base.box_thresh = static_cast<float>(std::atof(env));
  if (const char* env = std::getenv("DET_UNCLIP"); env && *env)
    base.unclip_ratio = static_cast<float>(std::atof(env));
  return base;
}

// Effective engine optimization-profile MAX side. The TRT profile MAX (and the
// buffers sized against it) must be >= the largest possible resize output, i.e.
// >= the model's max_side_limit. DET_MAX_SIDE, when set, overrides it (the
// single-knob override). Clamped to [kDetMaxSideMin, kDetMaxSideMax].
//
// Rounded UP to the next multiple of 32: compute_det_resize caps the longer
// side at max_side_limit then re-rounds to /32 (round-to-nearest), which can
// nudge the output above a non-/32 max_side_limit by up to 16px. Sizing the
// profile/buffers at the /32-ceil guarantees they always cover that output.
//
// Read by:
//   - paddle_det.cpp (GPU detector — sizes pinned input buffers)
//   - cpu_paddle_det.cpp (CPU detector — same role)
//   - onnx_to_trt.cpp (TRT engine builder — sizes the optimization profile
//     MAX, and is included in the engine cache key)
// All three MUST agree or the engine and runtime silently disagree.
[[nodiscard]] inline int effective_det_max_side(const DetResizeParams& p = kDetResizeDefault) {
  int v = p.max_side_limit;
  if (const char* env = std::getenv("DET_MAX_SIDE"); env && *env) v = std::atoi(env);
  v = std::clamp(v, kDetMaxSideMin, kDetMaxSideMax);
  return ((v + 31) / 32) * 32;  // /32-ceil stays within [kDetMaxSideMin, Max]
}

} // namespace turbo_ocr::detection
