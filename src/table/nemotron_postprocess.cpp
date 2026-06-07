#include "turbo_ocr/table/nemotron_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <utility>

namespace turbo_ocr::table {

namespace {

inline float sigmoid(float x) noexcept {
  return 1.0f / (1.0f + std::exp(-x));
}

float iou_aabb(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  float ix0 = std::max(a[0], b[0]);
  float iy0 = std::max(a[1], b[1]);
  float ix1 = std::min(a[2], b[2]);
  float iy1 = std::min(a[3], b[3]);
  float iw = std::max(0.0f, ix1 - ix0);
  float ih = std::max(0.0f, iy1 - iy0);
  float inter = iw * ih;
  float aw = std::max(0.0f, a[2] - a[0]);
  float ah = std::max(0.0f, a[3] - a[1]);
  float bw = std::max(0.0f, b[2] - b[0]);
  float bh = std::max(0.0f, b[3] - b[1]);
  float uni = aw * ah + bw * bh - inter;
  if (uni <= 0.0f) return 0.0f;
  return inter / uni;
}

// Per-class NMS with class-specific IoU thresholds. `row` and `column`
// detections (classes 2, 3) overlap heavily by construction — same row
// fires from dozens of anchors — so they need a much higher merge
// threshold than `cell` (class 1) which we want to keep distinct.
std::vector<NemotronBox> nms_per_class(std::vector<NemotronBox> boxes) {
  std::ranges::sort(boxes, [](const NemotronBox& a, const NemotronBox& b) {
    return a.score > b.score;
  });
  std::vector<NemotronBox> kept;
  std::vector<char> suppressed(boxes.size(), 0);
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    if (suppressed[i]) continue;
    kept.push_back(boxes[i]);
    // Headers (4) merge under the same band-level threshold as row/col;
    // cells (1) keep the tighter cell threshold so adjacent cells aren't
    // collapsed.
    const float iou_thresh = (boxes[i].class_id == 1)
                                ? kNemotronCellNmsIoU
                                : kNemotronRowColNmsIoU;
    for (std::size_t j = i + 1; j < boxes.size(); ++j) {
      if (suppressed[j]) continue;
      if (boxes[i].class_id != boxes[j].class_id) continue;
      if (iou_aabb(boxes[i].bbox, boxes[j].bbox) > iou_thresh)
        suppressed[j] = 1;
    }
  }
  return kept;
}

// Keep at most `cap` highest-score boxes. Assumes input sorted by score
// desc isn't required — we sort here.
void cap_top_k(std::vector<NemotronBox>& v, int cap) {
  if (static_cast<int>(v.size()) <= cap) return;
  std::ranges::partial_sort(
      v, v.begin() + cap,
      [](const NemotronBox& a, const NemotronBox& b) {
        return a.score > b.score;
      });
  v.resize(cap);
}

// Map a Cell-class bbox to (col_start, colspan): inclusive overlap test
// against the sorted column bands. Coverage ≥ 50% of the column width
// counts as occupied.
std::pair<int, int> colspan_for_cell(const NemotronBox& cell,
                                      const std::vector<NemotronBox>& cols) {
  int first = -1;
  int last = -1;
  for (std::size_t i = 0; i < cols.size(); ++i) {
    float cw = cols[i].bbox[2] - cols[i].bbox[0];
    if (cw <= 0.0f) continue;
    float ix0 = std::max(cell.bbox[0], cols[i].bbox[0]);
    float ix1 = std::min(cell.bbox[2], cols[i].bbox[2]);
    float overlap = std::max(0.0f, ix1 - ix0);
    if (overlap / cw >= 0.5f) {
      if (first < 0) first = static_cast<int>(i);
      last = static_cast<int>(i);
    }
  }
  if (first < 0) return {-1, 1};
  return {first, last - first + 1};
}

std::pair<int, int> rowspan_for_cell(const NemotronBox& cell,
                                      const std::vector<NemotronBox>& rows) {
  int first = -1;
  int last = -1;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    float rh = rows[i].bbox[3] - rows[i].bbox[1];
    if (rh <= 0.0f) continue;
    float iy0 = std::max(cell.bbox[1], rows[i].bbox[1]);
    float iy1 = std::min(cell.bbox[3], rows[i].bbox[3]);
    float overlap = std::max(0.0f, iy1 - iy0);
    if (overlap / rh >= 0.5f) {
      if (first < 0) first = static_cast<int>(i);
      last = static_cast<int>(i);
    }
  }
  if (first < 0) return {-1, 1};
  return {first, last - first + 1};
}

void append_escaped(std::string& out, std::string_view s) {
  for (char c : s) {
    switch (c) {
      case '<': out += "&lt;";  break;
      case '>': out += "&gt;";  break;
      case '&': out += "&amp;"; break;
      default:  out += c;
    }
  }
}

// Same as `append_escaped` but preserves literal `<br/>` tokens emitted by
// the cell-text builder for multi-line cells. Everything around the `<br/>`
// is still HTML-escaped.
void append_escaped_keep_br(std::string& out, std::string_view s) {
  constexpr std::string_view kBr = "<br/>";
  std::size_t i = 0;
  while (i < s.size()) {
    const std::size_t hit = s.find(kBr, i);
    if (hit == std::string_view::npos) {
      append_escaped(out, s.substr(i));
      return;
    }
    append_escaped(out, s.substr(i, hit - i));
    out += kBr;
    i = hit + kBr.size();
  }
}

} // namespace

NemotronDecode decode_nemotron(const float* preds,
                                std::size_t num_queries,
                                int ori_w,
                                int ori_h) {
  NemotronDecode out;
  if (!preds || num_queries == 0 || ori_w <= 0 || ori_h <= 0) return out;

  const float ratio = std::min(
      static_cast<float>(kNemotronCanvas) / static_cast<float>(ori_h),
      static_cast<float>(kNemotronCanvas) / static_cast<float>(ori_w));
  if (!(ratio > 0.0f)) return out;

  std::vector<NemotronBox> raw;
  raw.reserve(256);
  const float fw = static_cast<float>(ori_w);
  const float fh = static_cast<float>(ori_h);

  for (std::size_t q = 0; q < num_queries; ++q) {
    const float* row = preds + q * kNemotronStride;
    float obj = sigmoid(row[4]);
    int best_cls = 0;
    float best_score = -1.0f;
    for (int c = 0; c < 5; ++c) {
      float p = sigmoid(row[5 + c]);
      float s = obj * p;
      if (s > best_score) { best_score = s; best_cls = c; }
    }
    // Drop border (0) — never used downstream. Keep header (4) so the
    // assembler can mark <th>/<thead> rows.
    if (best_cls == 0) continue;
    // Per-class score floor — row/col duplicate at low conf and inflate the
    // grid count if we don't gate them harder than cells. Headers are
    // rare per table; we use a slightly lower floor than row/col so we
    // still catch the single header row when the model is unsure.
    float floor = kNemotronConfThresh;
    if (best_cls == 2 || best_cls == 3) floor = kNemotronRowColScoreThresh;
    else if (best_cls == 4)             floor = kNemotronHeaderScoreThresh;
    if (best_score < floor) continue;

    float cx = row[0], cy = row[1], w = row[2], h = row[3];
    float x1 = (cx - 0.5f * w) / ratio;
    float y1 = (cy - 0.5f * h) / ratio;
    float x2 = (cx + 0.5f * w) / ratio;
    float y2 = (cy + 0.5f * h) / ratio;
    x1 = std::clamp(x1, 0.0f, fw);
    y1 = std::clamp(y1, 0.0f, fh);
    x2 = std::clamp(x2, 0.0f, fw);
    y2 = std::clamp(y2, 0.0f, fh);
    if (x2 <= x1 || y2 <= y1) continue;
    raw.push_back({{x1, y1, x2, y2}, best_score, best_cls});
  }

  auto kept = nms_per_class(std::move(raw));
  for (auto& b : kept) {
    if (b.class_id == 1) out.cells.push_back(b);
    else if (b.class_id == 2) out.rows.push_back(b);
    else if (b.class_id == 3) out.columns.push_back(b);
    else if (b.class_id == 4) out.headers.push_back(b);
  }
  // Hard cap on row/col counts — a sane table has <80 rows and <50 cols.
  // Anything past this is residual duplicate detections that NMS failed
  // to merge (e.g. row stripes split into two boxes by a misaligned anchor).
  cap_top_k(out.rows,    kNemotronMaxRows);
  cap_top_k(out.columns, kNemotronMaxCols);
  std::ranges::sort(out.rows, [](const NemotronBox& a, const NemotronBox& b) {
    return 0.5f * (a.bbox[1] + a.bbox[3]) < 0.5f * (b.bbox[1] + b.bbox[3]);
  });
  std::ranges::sort(out.columns, [](const NemotronBox& a, const NemotronBox& b) {
    return 0.5f * (a.bbox[0] + a.bbox[2]) < 0.5f * (b.bbox[0] + b.bbox[2]);
  });
  return out;
}

NemotronAssembly assemble_nemotron(const NemotronDecode& d) {
  NemotronAssembly out;
  out.num_rows = static_cast<int>(d.rows.size());
  out.num_cols = static_cast<int>(d.columns.size());
  if (out.num_rows == 0 || out.num_cols == 0) return out;

  // Track which (row, col) grid slots are already covered so spanning cells
  // emit once and following rows know to skip them. Layout follows HTML
  // semantics: a cell at (r,c) with rowspan=R, colspan=C blocks
  // (r..r+R-1, c..c+C-1).
  std::vector<char> taken(static_cast<std::size_t>(out.num_rows) * out.num_cols, 0);
  auto idx = [&](int r, int c) {
    return static_cast<std::size_t>(r) * out.num_cols + c;
  };

  // Emit detected cell boxes first; they carry colspan/rowspan info.
  // Sort by (row_start, col_start) so emission order matches reading order.
  struct PendingCell {
    int rs, re, cs, ce;
    std::array<float, 4> bbox;
  };
  std::vector<PendingCell> pending;
  pending.reserve(d.cells.size());
  // Cap spurious mega-spans. A real merged cell covers a few rows/cols; when
  // a Cell-class detection ends up overlapping > half the grid it's almost
  // always a bad detection that NMS leaked through (the model sometimes
  // emits a "whole-table" box at low confidence). Clamping to 1×1 in those
  // cases prevents the assembler from emitting `rowspan="6"`/`colspan="5"`
  // attributes that wreck the predicted structure. We compare against half
  // the detected grid dimension so we never clamp on small tables where a
  // 2-row merged header (e.g. 2 of 3 rows) is legitimate.
  const int span_row_cap = std::max(1, out.num_rows / 2);
  const int span_col_cap = std::max(1, out.num_cols / 2);
  for (const auto& cell : d.cells) {
    auto [cs, csp] = colspan_for_cell(cell, d.columns);
    auto [rs, rsp] = rowspan_for_cell(cell, d.rows);
    if (cs < 0 || rs < 0) continue;
    if (rsp > span_row_cap) rsp = 1;
    if (csp > span_col_cap) csp = 1;
    pending.push_back({rs, rs + rsp - 1, cs, cs + csp - 1, cell.bbox});
  }
  std::ranges::sort(pending, [](const PendingCell& a, const PendingCell& b) {
    if (a.rs != b.rs) return a.rs < b.rs;
    return a.cs < b.cs;
  });

  for (const auto& pc : pending) {
    if (taken[idx(pc.rs, pc.cs)]) continue;
    NemotronCell cc;
    cc.row_idx = pc.rs;
    cc.col_idx = pc.cs;
    cc.colspan = pc.ce - pc.cs + 1;
    cc.rowspan = pc.re - pc.rs + 1;
    cc.bbox = pc.bbox;
    out.cells.push_back(cc);
    for (int r = pc.rs; r <= pc.re; ++r)
      for (int c = pc.cs; c <= pc.ce; ++c)
        taken[idx(r, c)] = 1;
  }

  // Fill in any uncovered grid slots with row×column intersection bboxes.
  // Keeps reading order intact and gives cell_matcher something to OCR.
  for (int r = 0; r < out.num_rows; ++r) {
    for (int c = 0; c < out.num_cols; ++c) {
      if (taken[idx(r, c)]) continue;
      NemotronCell cc;
      cc.row_idx = r;
      cc.col_idx = c;
      cc.colspan = 1;
      cc.rowspan = 1;
      cc.bbox = {d.columns[c].bbox[0], d.rows[r].bbox[1],
                 d.columns[c].bbox[2], d.rows[r].bbox[3]};
      out.cells.push_back(cc);
      taken[idx(r, c)] = 1;
    }
  }

  std::ranges::sort(out.cells, [](const NemotronCell& a, const NemotronCell& b) {
    if (a.row_idx != b.row_idx) return a.row_idx < b.row_idx;
    return a.col_idx < b.col_idx;
  });

  // Tag row indices whose y-center sits inside any detected header bbox.
  // The model emits one or more header boxes spanning the topmost N rows;
  // we treat a row as "header" when its band center is contained.
  if (!d.headers.empty()) {
    for (int r = 0; r < out.num_rows; ++r) {
      float yc = 0.5f * (d.rows[r].bbox[1] + d.rows[r].bbox[3]);
      for (const auto& h : d.headers) {
        if (yc >= h.bbox[1] && yc <= h.bbox[3]) {
          out.header_rows.push_back(r);
          break;
        }
      }
    }
  }
  return out;
}

std::string nemotron_to_html(const NemotronAssembly& assembly,
                              const std::vector<std::string>& cell_texts) {
  std::string out = "<html><body><table>";
  if (assembly.cells.empty()) {
    out += "</table></body></html>";
    return out;
  }

  // O(1) lookup for header membership. Sets are small (≤2 rows typical).
  auto is_header_row = [&](int r) {
    for (int h : assembly.header_rows) if (h == r) return true;
    return false;
  };
  const bool any_header = !assembly.header_rows.empty();

  // When no header detected, fall back to the legacy bare-<tr>/<td>
  // emission so we don't regress tables whose GT has no <thead>/<tbody>.
  // When at least one header row exists, emit <thead> + <th> for those
  // rows and <tbody> + <td> for the rest, matching the OmniDocBench GT
  // structure on tables that carry a column-header band.
  int  cur_row  = -1;
  bool in_thead = false;
  bool in_tbody = false;
  for (std::size_t i = 0; i < assembly.cells.size(); ++i) {
    const auto& cc = assembly.cells[i];
    const bool head = any_header && is_header_row(cc.row_idx);
    if (cc.row_idx != cur_row) {
      if (cur_row >= 0) out += "</tr>";
      if (any_header) {
        if (head && !in_thead) {
          if (in_tbody) { out += "</tbody>"; in_tbody = false; }
          out += "<thead>"; in_thead = true;
        } else if (!head && !in_tbody) {
          if (in_thead) { out += "</thead>"; in_thead = false; }
          out += "<tbody>"; in_tbody = true;
        }
      }
      out += "<tr>";
      cur_row = cc.row_idx;
    }
    const char* tag = head ? "th" : "td";
    out += '<';
    out += tag;
    if (cc.colspan > 1) {
      out += " colspan=\"";
      out += std::to_string(cc.colspan);
      out += '"';
    }
    if (cc.rowspan > 1) {
      out += " rowspan=\"";
      out += std::to_string(cc.rowspan);
      out += '"';
    }
    out += '>';
    if (i < cell_texts.size()) append_escaped_keep_br(out, cell_texts[i]);
    out += "</";
    out += tag;
    out += '>';
  }
  if (cur_row >= 0) out += "</tr>";
  if (in_thead) out += "</thead>";
  if (in_tbody) out += "</tbody>";
  out += "</table></body></html>";
  return out;
}

} // namespace turbo_ocr::table
