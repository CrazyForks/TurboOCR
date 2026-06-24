#pragma once

#include <algorithm>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

#include "turbo_ocr/layout/layout_types.h"

namespace turbo_ocr::layout {

enum class MergeMode { kUnion, kLarge, kSmall };

// How nested layout boxes are reconciled. Default "large" keeps outer regions
// and drops boxes nested inside them; on forms (every field is a box inside an
// outer frame) that collapses the page to a few containers, so those callers
// set "small" (keep innermost) or "union" (keep all).
inline MergeMode layout_merge_mode() {
  static const MergeMode mode = [] {
    const char *v = std::getenv("LAYOUT_MERGE_MODE");
    const std::string s = v ? v : "large";
    if (s == "union") return MergeMode::kUnion;
    if (s == "small") return MergeMode::kSmall;
    return MergeMode::kLarge;
  }();
  return mode;
}

// display_formula / inline_formula are kept even when nested in a text or table
// region, so standalone math is never swallowed by its surrounding block.
constexpr bool is_formula_class(int cls) {
  return cls == 5 /*display_formula*/ || cls == 15 /*inline_formula*/;
}

// Pin the magic class indices used here so a future relabel of kLayoutLabels
// can't silently break the formula guard. (kImageClassId, the nestable-child
// set, and the rest of the pins live in layout_types.h.)
static_assert(kLayoutLabels[5] == "display_formula" &&
                  kLayoutLabels[15] == "inline_formula",
              "layout class indices drifted — update layout_postfilter.h");

// OPT-IN: when LAYOUT_KEEP_NESTED_CHILDREN is set, the nested-box
// reconciliation also protects the model's legitimate child classes
// (figure_title, footnote, formula_number, paragraph_title — see
// is_nestable_class) from being dropped inside a parent region, the same way
// formulas are always protected. Default (unset) leaves every merge mode
// byte-identical to before: only formulas survive nesting.
inline bool layout_keep_nested_children() {
  static const bool on = [] {
    const char *v = std::getenv("LAYOUT_KEEP_NESTED_CHILDREN");
    return v && v[0] && v[0] != '0';
  }();
  return on;
}

// Shared post-decode cleanup for the GPU and CPU layout paths: NMS, oversized
// "image" drop, then nested-box reconciliation per LAYOUT_MERGE_MODE.
inline std::vector<LayoutBox>
postfilter_layout_boxes(std::vector<LayoutBox> out, int orig_h, int orig_w) {
  auto box_coords = [](const LayoutBox &lb) {
    return std::tuple{lb.box[0][0], lb.box[0][1], lb.box[2][0], lb.box[2][1]};
  };

  // 1. NMS: same-class IoU >= 0.6 or cross-class IoU >= 0.98 suppresses the
  //    lower-scoring box.
  std::sort(out.begin(), out.end(),
            [](const LayoutBox &a, const LayoutBox &b) {
              return a.score > b.score;
            });
  auto compute_iou = [&](const LayoutBox &a, const LayoutBox &b) -> float {
    auto [ax0, ay0, ax1, ay1] = box_coords(a);
    auto [bx0, by0, bx1, by1] = box_coords(b);
    int ix0 = std::max(ax0, bx0), iy0 = std::max(ay0, by0);
    int ix1 = std::min(ax1, bx1), iy1 = std::min(ay1, by1);
    float inter = std::max(0, ix1 - ix0 + 1) * std::max(0, iy1 - iy0 + 1);
    float area_a = static_cast<float>(ax1 - ax0 + 1) * (ay1 - ay0 + 1);
    float area_b = static_cast<float>(bx1 - bx0 + 1) * (by1 - by0 + 1);
    float union_area = area_a + area_b - inter;
    return union_area > 0 ? inter / union_area : 0.0f;
  };
  constexpr float kIoUSame = 0.6f;
  constexpr float kIoUDiff = 0.98f;
  std::vector<LayoutBox> nms_out;
  nms_out.reserve(out.size());
  std::vector<bool> suppressed(out.size(), false);
  for (size_t i = 0; i < out.size(); ++i) {
    if (suppressed[i]) continue;
    nms_out.push_back(out[i]);
    for (size_t j = i + 1; j < out.size(); ++j) {
      if (suppressed[j]) continue;
      float thresh = (out[i].class_id == out[j].class_id) ? kIoUSame : kIoUDiff;
      if (compute_iou(out[i], out[j]) >= thresh) suppressed[j] = true;
    }
  }

  // 2. Drop "image" detections covering >82% (portrait) / >93% (landscape) of
  //    the page — a full-page "image" box is a detector artefact, not content.
  const float img_area = static_cast<float>(orig_w) * orig_h;
  const float area_thresh = (orig_h > orig_w) ? 0.82f : 0.93f;
  constexpr int kImageClassId = 14; // "image" in kLayoutLabels
  std::erase_if(nms_out, [&](const LayoutBox &lb) {
    if (lb.class_id != kImageClassId) return false;
    float box_area = static_cast<float>(lb.box[2][0] - lb.box[0][0]) *
                     (lb.box[2][1] - lb.box[0][1]);
    return box_area > area_thresh * img_area;
  });

  // 3. Reconcile nested boxes per LAYOUT_MERGE_MODE. union keeps everything.
  //    large drops boxes nested in another (keep containers). small drops the
  //    pure containers (keep innermost). A is "inside" B when >=90% of A's area
  //    overlaps B; formula boxes are never counted as inside a non-formula box.
  const MergeMode mode = layout_merge_mode();
  if (mode != MergeMode::kUnion) {
    auto inside = [&](const LayoutBox &a, const LayoutBox &b) -> bool {
      auto [ax0, ay0, ax1, ay1] = box_coords(a);
      auto [bx0, by0, bx1, by1] = box_coords(b);
      int ix0 = std::max(ax0, bx0), iy0 = std::max(ay0, by0);
      int ix1 = std::min(ax1, bx1), iy1 = std::min(ay1, by1);
      float inter = std::max(0, ix1 - ix0) * std::max(0, iy1 - iy0);
      float area_a = static_cast<float>(ax1 - ax0) * (ay1 - ay0);
      return area_a > 0 && (inter / area_a) >= 0.9f;
    };
    const size_t n = nms_out.size();
    std::vector<bool> contains_other(n, false), inside_other(n, false);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        if (is_formula_class(nms_out[i].class_id) &&
            !is_formula_class(nms_out[j].class_id))
          continue;
        if (inside(nms_out[i], nms_out[j])) {
          inside_other[i] = true;
          contains_other[j] = true;
        }
      }
    }
    std::vector<LayoutBox> merged;
    merged.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const bool keep = (mode == MergeMode::kLarge)
                            ? !inside_other[i]
                            : (!contains_other[i] || inside_other[i]);
      if (keep) merged.push_back(std::move(nms_out[i]));
    }
    nms_out = std::move(merged);
  }

  return nms_out;
}

} // namespace turbo_ocr::layout
