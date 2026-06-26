#include "turbo_ocr/table/slanext_table_recognizer.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "turbo_ocr/common/cuda_check.h"       // abort_on_sticky_cuda_fault
#include "turbo_ocr/engine/onnx_to_trt.h"
#include "turbo_ocr/recognition/paddle_rec.h"  // per-cell crop OCR fill
#include "turbo_ocr/table/cell_matcher.h"      // OcrLine, match_cells_to_ocr
#include "turbo_ocr/table/html_reconstruct.h"  // reconstruct_html

namespace turbo_ocr::table {

namespace {
std::string env_or(const char *k, const std::string &dflt) {
  const char *v = std::getenv(k);
  return (v && v[0]) ? std::string(v) : dflt;
}
// Derive the decoder.bin / dict path that sits next to the encoder ONNX,
// mirroring the prior maybe_load_router_models default derivation.
std::string default_decoder_bin(const std::string &enc) {
  const std::string suf = "_encoder.onnx";
  std::string d = enc;
  auto p = d.rfind(suf);
  if (p != std::string::npos) d.replace(p, suf.size(), "_decoder.bin");
  return d;
}
std::string default_dict(const std::string &enc) {
  auto slash = enc.find_last_of('/');
  std::string dir = (slash == std::string::npos) ? std::string()
                                                  : enc.substr(0, slash + 1);
  return dir + "SLANeXt_dict_infer.txt";
}
std::unique_ptr<SlanextEncSplit> build_enc(const std::string &enc_onnx,
                                           const std::string &dec,
                                           const std::string &dict) {
  const std::string trt = engine::ensure_trt_engine(enc_onnx, "slanext_encoder");
  if (trt.empty()) {
    std::cerr << "[slanext] encoder engine build failed: " << enc_onnx << '\n';
    return nullptr;
  }
  auto s = std::make_unique<SlanextEncSplit>();
  if (!s->load_model(trt, dec, dict)) {
    std::cerr << "[slanext] load_model failed: " << enc_onnx << '\n';
    return nullptr;
  }
  return s;
}
} // namespace

bool SlanextTableRecognizer::load() {
  const std::string enc = env_or("TABLE_SLANEXT_ENCODER_ONNX", "");
  if (enc.empty()) return false;
  const std::string dec = env_or("TABLE_SLANEXT_DECODER_BIN", default_decoder_bin(enc));
  const std::string dict = env_or("TABLE_SLANEXT_DICT", default_dict(enc));
  wired_ = build_enc(enc, dec, dict);
  if (!wired_) return false;

  // Optional wireless encoder + table_cls router (wired/wireless routing).
  const std::string wenc = env_or("TABLE_SLANEXT_WIRELESS_ENCODER_ONNX", "");
  if (!wenc.empty()) {
    const std::string wdec = env_or("TABLE_SLANEXT_WIRELESS_DECODER_BIN", default_decoder_bin(wenc));
    const std::string wdict = env_or("TABLE_SLANEXT_DICT", default_dict(wenc));
    wireless_ = build_enc(wenc, wdec, wdict);  // soft: routing falls back to wired
    if (wireless_) {
      const std::string cls_onnx = env_or("TABLE_CLS_ONNX", "models/table/table_cls.onnx");
      if (std::filesystem::exists(cls_onnx)) {
        const std::string ctrt = engine::ensure_trt_engine(cls_onnx, "table_cls");
        if (!ctrt.empty()) {
          auto c = std::make_unique<TableCls>();
          if (c->load_model(ctrt)) cls_ = std::move(c);
        }
      }
      std::cout << "[Pipeline] Table backend=slanext (wired+wireless"
                << (cls_ ? "+cls router" : ", no cls -> wired only") << ")\n";
    }
  } else {
    std::cout << "[Pipeline] Table backend=slanext (wired, TRT FP16 encoder + host decode)\n";
  }
  return true;
}

std::vector<router::TableResult>
SlanextTableRecognizer::run(const GpuImage &page, const std::vector<Box> &regions,
                            const std::vector<OCRResultItem> &page_ocr,
                            cudaStream_t stream) {
  std::vector<router::TableResult> out;
  out.reserve(regions.size());

  std::vector<TableVariant> tvars;
  if (cls_ && wireless_ && !regions.empty()) {
    try {
      tvars = cls_->classify_batch(page, regions, stream);
    } catch (const std::exception &e) {
      // The wired/wireless classifier is only a router; a classifier hiccup
      // must not discard every table. Sticky CUDA faults still exit the process
      // (poisoned context); a recoverable failure falls back to the common case
      // (all regions wired) — an empty tvars makes the loop below default to
      // wired_ for every region.
      turbo_ocr::abort_on_sticky_cuda_fault("table_cls classify_batch");
      cudaGetLastError();  // clear the recoverable error before the run loop
      std::cerr << "[slanext] table_cls classify_batch FAILED (" << e.what()
                << ") — defaulting all regions to wired\n";
      tvars.clear();
    }
  }

  for (std::size_t ti = 0; ti < regions.size(); ++ti) {
   try {
    const Box &region = regions[ti];
    int rx = INT_MAX, ry = INT_MAX, rax2 = INT_MIN, ray2 = INT_MIN;
    for (const auto &p : region.pts) {
      rx = std::min(rx, p[0]); ry = std::min(ry, p[1]);
      rax2 = std::max(rax2, p[0]); ray2 = std::max(ray2, p[1]);
    }
    rx = std::max(rx, 0); ry = std::max(ry, 0);  // match infer()'s clamp origin

    SlanextEncSplit *enc = wired_.get();
    if (ti < tvars.size() && tvars[ti] == TableVariant::Wireless && wireless_)
      enc = wireless_.get();
    const auto sr = enc->infer(page, region, stream);

    // Cells from the page text-OCR: geometry-match each structure-order cell
    // quad to OCR lines inside the region, then reconstruct_html substitutes.
    std::vector<OcrLine> region_ocr;
    std::vector<std::string> region_texts;
    region_ocr.reserve(page_ocr.size());
    region_texts.reserve(page_ocr.size());
    for (const auto &r : page_ocr) {
      int bx1 = INT_MAX, by1 = INT_MAX, bx2 = INT_MIN, by2 = INT_MIN, cx = 0, cy = 0;
      for (const auto &p : r.box.pts) {
        cx += p[0]; cy += p[1];
        bx1 = std::min(bx1, p[0]); by1 = std::min(by1, p[1]);
        bx2 = std::max(bx2, p[0]); by2 = std::max(by2, p[1]);
      }
      cx /= 4; cy /= 4;
      if (cx < rx || cx > rax2 || cy < ry || cy > ray2) continue;
      region_ocr.push_back(OcrLine{{static_cast<float>(bx1), static_cast<float>(by1),
                                    static_cast<float>(bx2), static_cast<float>(by2)},
                                   r.text});
      region_texts.push_back(r.text);
    }
    // SLANeXt per-token cell quads are region-local -> shift to page coords.
    std::vector<std::array<int, 8>> quads;
    quads.reserve(sr.cells.size());
    for (const auto &c : sr.cells) {
      std::array<int, 8> q = c.bbox;
      for (int k = 0; k < 4; ++k) { q[2 * k] += rx; q[2 * k + 1] += ry; }
      quads.push_back(q);
    }
    auto matched = match_cells_to_ocr(quads, region_ocr);

    // Per-cell crop OCR: SLANeXt grid cells with no page-OCR line get recognized
    // directly from their quad crop. Recovers dense-grid cells the page text
    // detector under-segmented (the dominant local-table content loss). One
    // batched rec call per table over all empty cells; page-coordinate quads.
    if (cell_rec_ && !matched.empty()) {
      std::vector<Box> empty_boxes;
      std::vector<std::size_t> empty_ci;
      for (std::size_t ci = 0; ci < matched.size() && ci < quads.size(); ++ci) {
        if (!matched[ci].ocr_indices.empty()) continue;
        const auto &q = quads[ci];
        const int w = std::abs(q[2] - q[0]);
        const int h = std::abs(q[5] - q[1]);
        if (w < 4 || h < 4) continue;  // skip degenerate / spacer cells
        Box b;
        b.pts = {{{q[0], q[1]}, {q[2], q[3]}, {q[4], q[5]}, {q[6], q[7]}}};
        empty_boxes.push_back(b);
        empty_ci.push_back(ci);
      }
      if (!empty_boxes.empty()) {
        auto rec = cell_rec_->run(page, empty_boxes, stream);
        for (std::size_t k = 0; k < rec.size() && k < empty_ci.size(); ++k) {
          const std::string &txt = rec[k].first;
          if (txt.empty() || rec[k].second < 0.5f) continue;  // drop empty / low-conf noise
          region_texts.push_back(txt);
          matched[empty_ci[k]].ocr_indices.push_back(region_texts.size() - 1);
        }
      }
    }

    router::TableResult tr;
    tr.layout_id = -1;  // stamped by caller (input order)
    tr.html      = reconstruct_html(sr.structure, matched, region_texts);
    tr.score     = sr.structure_score;
    tr.box       = region;
    out.push_back(std::move(tr));
   } catch (const std::exception &e) {
    // Per-region degrade: a CUDA/inference fault on ONE table region must not
    // abort the whole page (every other table lost) — mirror the graceful
    // "table region DROPPED" path inside infer(). Sticky faults still exit the
    // process; a recoverable error degrades just this region (empty html ->
    // counted degraded upstream) and we continue. One TableResult per region is
    // pushed unconditionally so the caller's layout_id stamping stays aligned.
    turbo_ocr::abort_on_sticky_cuda_fault("slanext_table_recognizer region");
    cudaGetLastError();  // clear the recoverable error before the next region
    std::cerr << "[slanext] table region " << ti << " FAILED (" << e.what()
              << ") — region DROPPED, continuing\n";
    router::TableResult tr;
    tr.layout_id = -1;
    tr.box       = regions[ti];  // html left empty -> degraded accounting
    out.push_back(std::move(tr));
   }
  }
  return out;
}

} // namespace turbo_ocr::table
