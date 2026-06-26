#include "ocr_pipeline_detail.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <utility>

#include "turbo_ocr/formula/formula_recognizer.h"
#include "turbo_ocr/router/router_types.h"
#include "turbo_ocr/table/table_recognizer.h"

namespace turbo_ocr::pipeline::detail {

Box adjust_table_region(const Box &in,
                        const std::vector<OCRResultItem> &results) {
  static const float kCropMargin = [] {
    const char *v = std::getenv("TABLE_CROP_MARGIN");
    return v && v[0] ? std::strtof(v, nullptr) : 0.03f;
  }();
  static const int kDetUnion = [] {
    const char *v = std::getenv("TABLE_CROP_MODE");
    return (v && std::string(v) == "detunion") ? 1 : 0;
  }();
  Box region = in;
  if (kDetUnion) {
    int lx1 = INT_MAX, ly1 = INT_MAX, lx2 = INT_MIN, ly2 = INT_MIN;
    for (const auto &p : region.pts) {
      lx1 = std::min(lx1, p[0]); ly1 = std::min(ly1, p[1]);
      lx2 = std::max(lx2, p[0]); ly2 = std::max(ly2, p[1]);
    }
    int ux1 = INT_MAX, uy1 = INT_MAX, ux2 = INT_MIN, uy2 = INT_MIN;
    bool any = false;
    for (const auto &r : results) {
      int cx = 0, cy = 0;
      for (const auto &p : r.box.pts) { cx += p[0]; cy += p[1]; }
      cx /= 4; cy /= 4;
      if (cx >= lx1 && cx <= lx2 && cy >= ly1 && cy <= ly2) {
        for (const auto &p : r.box.pts) {
          ux1 = std::min(ux1, p[0]); uy1 = std::min(uy1, p[1]);
          ux2 = std::max(ux2, p[0]); uy2 = std::max(uy2, p[1]);
        }
        any = true;
      }
    }
    if (any) {
      int nx1 = std::max(lx1, ux1), ny1 = std::max(ly1, uy1);
      int nx2 = std::min(lx2, ux2), ny2 = std::min(ly2, uy2);
      if (nx2 > nx1 && ny2 > ny1)
        region.pts = {{{nx1, ny1}, {nx2, ny1}, {nx2, ny2}, {nx1, ny2}}};
    }
  }
  if (kCropMargin > 0.0f) {
    int ax1 = INT_MAX, ay1 = INT_MAX, ax2 = INT_MIN, ay2 = INT_MIN;
    for (const auto &p : region.pts) {
      ax1 = std::min(ax1, p[0]); ay1 = std::min(ay1, p[1]);
      ax2 = std::max(ax2, p[0]); ay2 = std::max(ay2, p[1]);
    }
    const int mw = static_cast<int>((ax2 - ax1) * kCropMargin);
    const int mh = static_cast<int>((ay2 - ay1) * kCropMargin);
    ax1 -= mw; ay1 -= mh; ax2 += mw; ay2 += mh;  // backend clamps to image
    region.pts = {{{ax1, ay1}, {ax2, ay1}, {ax2, ay2}, {ax1, ay2}}};
  }
  return region;
}

} // namespace turbo_ocr::pipeline::detail

namespace turbo_ocr::pipeline {

// Await + parse + assemble deferred external (VLM) crops, OFF the GPU worker.
// See pipeline_result.h. No-op when out.pending is empty (sync path).
void finalize_deferred(OcrPipelineResult &out) {
  auto &pe = out.pending;
  // Backstop the future joins: the crop pool resolves every promise (success or
  // its own per-crop timeout), so this only fires if a pool worker itself died/
  // wedged. The promise-backed futures are safe to abandon (no blocking dtor),
  // so a timed-out crop is left empty -> tagged degraded rather than hanging the
  // GPU worker. Configurable via FINALIZE_DEFERRED_TIMEOUT_MS (default 120000).
  static const long kFinalizeTimeoutMs = [] {
    const char *e = std::getenv("FINALIZE_DEFERRED_TIMEOUT_MS");
    long v = e ? std::atol(e) : 120000;
    if (v <= 0) v = 120000;
    // The backstop must never outlive the per-request deadline: a wedged remote
    // crop would otherwise keep the GPU worker waiting long past the point the
    // client already got its 504. Cap at REQUEST_TIMEOUT_MS (same env the server
    // reads; default 60s, 0 == unbounded so the cap is skipped).
    const char *rt = std::getenv("REQUEST_TIMEOUT_MS");
    const long req = rt ? std::atol(rt) : 60000;
    if (req > 0 && v > req) v = req;
    return v;
  }();
  // The deferred (async) path is what the DEFAULT vlm backend uses. Mirror the
  // synchronous path's no-silent-failure contract: every region here was a
  // table/formula crop dispatched to the backend, so an empty parse result —
  // whether from an exhausted transport failure (the pool resolves the future
  // with "" on a 4xx/exhausted-retry) or a broken-promise throw — means the
  // stage produced nothing for a region it was asked to handle. Count those and
  // surface a degradation flag so a remote timeout can never look byte-identical
  // to a page that genuinely had no table/formula.
  if (pe.formula_parse && !pe.formula.empty()) {
    out.formulas.reserve(out.formulas.size() + pe.formula.size());
    std::size_t degraded = 0;
    for (auto &pc : pe.formula) {
      std::string latex;
      try {
        if (pc.fut.wait_for(std::chrono::milliseconds(kFinalizeTimeoutMs)) ==
            std::future_status::ready) {
          std::string raw = pc.fut.get();
          latex = pe.formula_parse(raw);
        } else {
          std::cerr << "[finalize_deferred] formula future timed out after "
                    << kFinalizeTimeoutMs << " ms (crop-pool worker stalled?); "
                       "tagging degraded\n";
        }
      } catch (const std::exception &e) {
        std::cerr << "[finalize_deferred] formula future threw: " << e.what() << '\n';
      }
      if (latex.empty()) ++degraded;
      router::FormulaResult fr;
      fr.layout_id = pc.layout_id;
      fr.latex     = std::move(latex);
      fr.score     = pc.score;
      fr.box       = pc.box;
      out.formulas.push_back(std::move(fr));
    }
    if (degraded > 0) {
      out.formula_degraded = true;
      out.formula_warning =
          "formula stage degraded: " + std::to_string(degraded) + " of " +
          std::to_string(pe.formula.size()) +
          " region(s) produced no LaTeX (async backend transport failure or "
          "empty response, not empty input)";
    }
  }
  if (pe.table_parse && !pe.table.empty()) {
    out.tables.reserve(out.tables.size() + pe.table.size());
    std::size_t degraded = 0;
    for (auto &pc : pe.table) {
      std::string html;
      try {
        if (pc.fut.wait_for(std::chrono::milliseconds(kFinalizeTimeoutMs)) ==
            std::future_status::ready) {
          std::string raw = pc.fut.get();
          html = pe.table_parse(raw);
        } else {
          std::cerr << "[finalize_deferred] table future timed out after "
                    << kFinalizeTimeoutMs << " ms (crop-pool worker stalled?); "
                       "tagging degraded\n";
        }
      } catch (const std::exception &e) {
        std::cerr << "[finalize_deferred] table future threw: " << e.what() << '\n';
      }
      if (html.empty()) ++degraded;
      router::TableResult tr;
      tr.layout_id = pc.layout_id;
      tr.html      = std::move(html);
      tr.score     = pc.score;
      tr.box       = pc.box;
      out.tables.push_back(std::move(tr));
    }
    if (degraded > 0) {
      out.table_degraded = true;
      out.table_warning =
          "table stage degraded: " + std::to_string(degraded) + " of " +
          std::to_string(pe.table.size()) +
          " region(s) produced no HTML (async backend transport failure or "
          "empty response, not empty input)";
    }
  }
  pe = PendingExternal{};
}

} // namespace turbo_ocr::pipeline
