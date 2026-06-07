#include "turbo_ocr/router/cua_router.h"

#include <algorithm>
#include <cmath>

namespace turbo_ocr::router {

namespace {

// Center-in-rect test, identical to the centroid match used by
// `assign_layout_ids` so router-side mapping and serializer-side mapping
// stay consistent.
[[nodiscard]] inline bool aabb_contains_centroid(
    const std::array<int, 4> &r, const turbo_ocr::Box &b) noexcept {
  float cx = 0.0f, cy = 0.0f;
  for (int k = 0; k < 4; ++k) {
    cx += static_cast<float>(b[k][0]);
    cy += static_cast<float>(b[k][1]);
  }
  cx *= 0.25f;
  cy *= 0.25f;
  return cx >= static_cast<float>(r[0]) && cx <= static_cast<float>(r[2]) &&
         cy >= static_cast<float>(r[1]) && cy <= static_cast<float>(r[3]);
}

} // namespace

float region_aspect_ratio_hint(
    const std::array<int, 4> &layout_aabb,
    const std::vector<turbo_ocr::Box> &det_boxes) noexcept {
  float sum = 0.0f;
  int n = 0;
  for (const auto &b : det_boxes) {
    if (!aabb_contains_centroid(layout_aabb, b)) continue;
    auto ab = turbo_ocr::aabb(b);
    int w = ab[2] - ab[0];
    int h = ab[3] - ab[1];
    if (h <= 0 || w <= 0) continue;
    sum += static_cast<float>(w) / static_cast<float>(h);
    ++n;
  }
  return n == 0 ? 0.0f : sum / static_cast<float>(n);
}

float region_symbol_density_hint(
    const std::array<int, 4> &layout_aabb,
    int contained_det_count) noexcept {
  // Cheap proxy: small layout cells with several det boxes are
  // formula-shaped — sub/superscripts, fraction bars, integral
  // symbols all surface as separate det boxes within a compact
  // region. log1p smooths over the wide area range without needing
  // a per-page normalization pass.
  const float w = static_cast<float>(std::max(0, layout_aabb[2] - layout_aabb[0]));
  const float h = static_cast<float>(std::max(0, layout_aabb[3] - layout_aabb[1]));
  const float area = std::max(1.0f, w * h);
  const float density = static_cast<float>(contained_det_count) /
                        std::log1p(area);
  return std::clamp(density, 0.0f, 1.0f);
}

} // namespace turbo_ocr::router
