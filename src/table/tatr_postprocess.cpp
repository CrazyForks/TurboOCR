#include "turbo_ocr/table/tatr_postprocess.h"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <utility>
#include <vector>

namespace turbo_ocr::table {

namespace {

float iou_aabb(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  const float ix0 = std::max(a[0], b[0]);
  const float iy0 = std::max(a[1], b[1]);
  const float ix1 = std::min(a[2], b[2]);
  const float iy1 = std::min(a[3], b[3]);
  const float iw = std::max(0.0f, ix1 - ix0);
  const float ih = std::max(0.0f, iy1 - iy0);
  const float inter = iw * ih;
  const float aw = std::max(0.0f, a[2] - a[0]);
  const float ah = std::max(0.0f, a[3] - a[1]);
  const float bw = std::max(0.0f, b[2] - b[0]);
  const float bh = std::max(0.0f, b[3] - b[1]);
  const float uni = aw * ah + bw * bh - inter;
  if (uni <= 0.0f) return 0.0f;
  return inter / uni;
}

// Per-class NMS over a flat box vector. Different IoU thresholds per class
// (rows/cols can overlap heavily by design; cell-class boxes should stay
// distinct).
std::vector<TatrBox> nms_per_class(std::vector<TatrBox> boxes) {
  std::ranges::sort(boxes, [](const TatrBox& a, const TatrBox& b) {
    return a.score > b.score;
  });
  std::vector<TatrBox> kept;
  std::vector<char> suppressed(boxes.size(), 0);
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    if (suppressed[i]) continue;
    kept.push_back(boxes[i]);
    const float iou_thresh = (boxes[i].class_id == 5)
                                 ? kTatrCellNmsIoU
                                 : kTatrRowColNmsIoU;
    for (std::size_t j = i + 1; j < boxes.size(); ++j) {
      if (suppressed[j]) continue;
      if (boxes[i].class_id != boxes[j].class_id) continue;
      if (iou_aabb(boxes[i].bbox, boxes[j].bbox) > iou_thresh)
        suppressed[j] = 1;
    }
  }
  return kept;
}

void cap_top_k(std::vector<NemotronBox>& v, int cap) {
  if (static_cast<int>(v.size()) <= cap) return;
  std::ranges::partial_sort(
      v, v.begin() + cap,
      [](const NemotronBox& a, const NemotronBox& b) {
        return a.score > b.score;
      });
  v.resize(cap);
}

// Stable, numerically-safe softmax over a small (<= kTatrNumClasses) row.
std::array<float, kTatrNumClasses> softmax_row(const float* logits) {
  std::array<float, kTatrNumClasses> probs{};
  float m = logits[0];
  for (int k = 1; k < kTatrNumClasses; ++k) m = std::max(m, logits[k]);
  float sum = 0.0f;
  for (int k = 0; k < kTatrNumClasses; ++k) {
    probs[k] = std::exp(logits[k] - m);
    sum += probs[k];
  }
  const float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
  for (int k = 0; k < kTatrNumClasses; ++k) probs[k] *= inv;
  return probs;
}

} // namespace

NemotronDecode decode_tatr(const float* logits,
                           const float* pred_boxes,
                           std::size_t num_queries,
                           int letterbox_size,
                           int new_w,
                           int new_h,
                           int ori_w,
                           int ori_h) {
  NemotronDecode out;
  if (!logits || !pred_boxes || num_queries == 0 ||
      letterbox_size <= 0 || new_w <= 0 || new_h <= 0 ||
      ori_w <= 0 || ori_h <= 0) {
    return out;
  }

  const float canvas = static_cast<float>(letterbox_size);
  const float ratio_w = static_cast<float>(ori_w) / static_cast<float>(new_w);
  const float ratio_h = static_cast<float>(ori_h) / static_cast<float>(new_h);
  const float fw = static_cast<float>(ori_w);
  const float fh = static_cast<float>(ori_h);

  std::vector<TatrBox> raw;
  raw.reserve(num_queries);

  for (std::size_t q = 0; q < num_queries; ++q) {
    const auto probs = softmax_row(logits + q * kTatrNumClasses);
    int best = 0;
    for (int k = 1; k < kTatrNumClasses; ++k) {
      if (probs[k] > probs[best]) best = k;
    }
    if (best == kTatrBgClass) continue;
    const float score = probs[best];

    float floor;
    if (best == 1 || best == 2)       floor = kTatrRowColThresh;
    else if (best == 5)                floor = kTatrCellThresh;
    else if (best == 3 || best == 4)   floor = kTatrHeaderThresh;
    else                                floor = 1.1f;  // skip class 0 (whole-table)
    if (score < floor) continue;

    const float* b = pred_boxes + q * 4;
    const float cx = b[0] * canvas;
    const float cy = b[1] * canvas;
    const float bw = b[2] * canvas;
    const float bh = b[3] * canvas;
    float x1 = (cx - 0.5f * bw) * ratio_w;
    float y1 = (cy - 0.5f * bh) * ratio_h;
    float x2 = (cx + 0.5f * bw) * ratio_w;
    float y2 = (cy + 0.5f * bh) * ratio_h;
    x1 = std::clamp(x1, 0.0f, fw);
    y1 = std::clamp(y1, 0.0f, fh);
    x2 = std::clamp(x2, 0.0f, fw);
    y2 = std::clamp(y2, 0.0f, fh);
    if (x2 <= x1 || y2 <= y1) continue;
    raw.push_back({{x1, y1, x2, y2}, score, best});
  }

  auto kept = nms_per_class(std::move(raw));

  // Translate TATR class ids to NemotronDecode buckets so the assembler and
  // HTML emitter can be reused as-is. NemotronBox carries class_id == 1
  // (cell), 2 (row), 3 (column), 4 (header) — same downstream semantics.
  for (const auto& tb : kept) {
    NemotronBox nb;
    nb.bbox = tb.bbox;
    nb.score = tb.score;
    switch (tb.class_id) {
      case 1: nb.class_id = 3; out.columns.push_back(nb); break;
      case 2: nb.class_id = 2; out.rows.push_back(nb);    break;
      case 5: nb.class_id = 1; out.cells.push_back(nb);   break;
      case 3:
      case 4: nb.class_id = 4; out.headers.push_back(nb); break;
      default: break;
    }
  }

  cap_top_k(out.rows,    kTatrMaxRows);
  cap_top_k(out.columns, kTatrMaxCols);

  std::ranges::sort(out.rows, [](const NemotronBox& a, const NemotronBox& b) {
    return 0.5f * (a.bbox[1] + a.bbox[3]) < 0.5f * (b.bbox[1] + b.bbox[3]);
  });
  std::ranges::sort(out.columns, [](const NemotronBox& a, const NemotronBox& b) {
    return 0.5f * (a.bbox[0] + a.bbox[2]) < 0.5f * (b.bbox[0] + b.bbox[2]);
  });
  return out;
}

} // namespace turbo_ocr::table
