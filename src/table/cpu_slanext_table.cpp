#include "turbo_ocr/table/cpu_slanext_table.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <onnxruntime_cxx_api.h>

#include "turbo_ocr/recognition/cpu_paddle_rec.h"
#include "turbo_ocr/table/cell_matcher.h"
#include "turbo_ocr/table/html_reconstruct.h"

namespace turbo_ocr::table {

namespace {

constexpr int DST = SlanextDecoderWeights::kInputSize;  // 488
constexpr int kTenc = SlanextDecoderWeights::kTenc;     // 256
constexpr int kCtx = SlanextDecoderWeights::kCtx;       // 96

// CPU port of slanext_pre_rgb_kernel (table_kernels.cu): ResizeByLong-488
// (preserve AR, bilinear) + per-channel ImageNet normalize + bottom-right pad
// with 0 in normalized space, written R,G,B plane order. Source crop is BGR8.
// Bit-matched to the GPU kernel: same (dx+0.5)*scale-0.5 sampling, same
// sub-rect clamp, same means/stds. Produces [3,488,488] CHW float.
void slanext_preprocess(const cv::Mat &crop, std::vector<float> &dst_chw) {
  const int rect_w = crop.cols, rect_h = crop.rows;
  dst_chw.assign(static_cast<std::size_t>(3) * DST * DST, 0.0f);
  if (rect_w <= 0 || rect_h <= 0) return;

  const float long_side = static_cast<float>(std::max(rect_w, rect_h));
  const float s = static_cast<float>(DST) / long_side;
  int new_w = static_cast<int>(std::lround(rect_w * s));
  int new_h = static_cast<int>(std::lround(rect_h * s));
  new_w = std::clamp(new_w, 1, DST);
  new_h = std::clamp(new_h, 1, DST);
  const float scale = 1.0f / s;

  // BGR source channel means/inv-stds (kernel passes them so that, after the
  // R,G,B store, Red gets 0.485/0.229, Green 0.456/0.224, Blue 0.406/0.225).
  const float mean_b = 0.406f, mean_g = 0.456f, mean_r = 0.485f;
  const float istd_b = 1.0f / 0.225f, istd_g = 1.0f / 0.224f, istd_r = 1.0f / 0.229f;
  const float inv255 = 1.0f / 255.0f;
  const int plane = DST * DST;
  const auto *src = crop.ptr<uint8_t>();
  const int step = static_cast<int>(crop.step);

  auto px = [&](int x, int y, int c) -> float {
    return static_cast<float>(src[static_cast<std::size_t>(y) * step + x * 3 + c]);
  };

  for (int dy = 0; dy < new_h; ++dy) {
    const float sy_rel = (dy + 0.5f) * scale - 0.5f;
    int y0 = static_cast<int>(std::floor(sy_rel));
    const float fy = sy_rel - y0;
    int y1 = y0 + 1;
    y0 = std::clamp(y0, 0, rect_h - 1);
    y1 = std::clamp(y1, 0, rect_h - 1);
    for (int dx = 0; dx < new_w; ++dx) {
      const float sx_rel = (dx + 0.5f) * scale - 0.5f;
      int x0 = static_cast<int>(std::floor(sx_rel));
      const float fx = sx_rel - x0;
      int x1 = x0 + 1;
      x0 = std::clamp(x0, 0, rect_w - 1);
      x1 = std::clamp(x1, 0, rect_w - 1);
      const float w00 = (1.0f - fx) * (1.0f - fy), w10 = fx * (1.0f - fy);
      const float w01 = (1.0f - fx) * fy, w11 = fx * fy;
      const int idx = dy * DST + dx;
      // channel 0=Blue, 1=Green, 2=Red in source
      const float b = w00 * px(x0, y0, 0) + w10 * px(x1, y0, 0) +
                      w01 * px(x0, y1, 0) + w11 * px(x1, y1, 0);
      const float g = w00 * px(x0, y0, 1) + w10 * px(x1, y0, 1) +
                      w01 * px(x0, y1, 1) + w11 * px(x1, y1, 1);
      const float r = w00 * px(x0, y0, 2) + w10 * px(x1, y0, 2) +
                      w01 * px(x0, y1, 2) + w11 * px(x1, y1, 2);
      dst_chw[0 * plane + idx] = (r * inv255 - mean_r) * istd_r;  // R
      dst_chw[1 * plane + idx] = (g * inv255 - mean_g) * istd_g;  // G
      dst_chw[2 * plane + idx] = (b * inv255 - mean_b) * istd_b;  // B
    }
  }
}

std::string env_or(const char *k, const std::string &dflt) {
  const char *v = std::getenv(k);
  return (v && v[0]) ? std::string(v) : dflt;
}

std::string default_decoder_bin(const std::string &enc) {
  const std::string suf = "_encoder.onnx";
  std::string d = enc;
  auto p = d.rfind(suf);
  if (p != std::string::npos) d.replace(p, suf.size(), "_decoder.bin");
  return d;
}

std::string default_dict(const std::string &enc) {
  auto slash = enc.find_last_of('/');
  std::string dir = (slash == std::string::npos) ? std::string() : enc.substr(0, slash + 1);
  return dir + "SLANeXt_dict_infer.txt";
}

}  // namespace

struct CpuSlanextEncoder::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "slanext_cpu"};
  std::unique_ptr<Ort::Session> sess;
  Ort::MemoryInfo mem{nullptr};
  std::string in_name, out_name;
};

CpuSlanextEncoder::~CpuSlanextEncoder() = default;

bool CpuSlanextEncoder::load(const std::string &encoder_onnx,
                             const std::string &decoder_bin,
                             const std::string &dict_path) {
  p_ = std::make_unique<Impl>();
  try {
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    p_->sess = std::make_unique<Ort::Session>(p_->env, encoder_onnx.c_str(), opts);
    p_->mem = Ort::MemoryInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions alloc;
    p_->in_name = p_->sess->GetInputNameAllocated(0, alloc).get();
    p_->out_name = p_->sess->GetOutputNameAllocated(0, alloc).get();
  } catch (const std::exception &e) {
    std::cerr << "[slanext-cpu] encoder load failed (" << encoder_onnx << "): " << e.what() << '\n';
    return false;
  }
  if (!weights_.load(decoder_bin)) return false;
  try {
    dict_ = dict_path.empty() ? CharDict::from_text(DEFAULT_DICT_TEXT)
                              : CharDict::from_path(dict_path);
  } catch (const std::exception &e) {
    // Match the GPU path: a missing/unreadable CONFIGURED dict falls back to the
    // bundled DEFAULT_DICT_TEXT (loudly) instead of hard-failing table recog.
    std::cerr << "[slanext-cpu] WARNING: dict load failed (" << e.what()
              << ") — falling back to bundled default dict\n";
    try {
      dict_ = CharDict::from_text(DEFAULT_DICT_TEXT);
    } catch (const std::exception &e2) {
      std::cerr << "[slanext-cpu] bundled default dict load failed: " << e2.what() << '\n';
      return false;
    }
  }
  return true;
}

StructureResult CpuSlanextEncoder::infer(const cv::Mat &page, const Box &region) {
  StructureResult empty{};
  if (!p_ || !p_->sess || page.empty()) return empty;

  auto bb = aabb(region);
  const int rx = std::clamp(bb[0], 0, page.cols - 1);
  const int ry = std::clamp(bb[1], 0, page.rows - 1);
  const int rw = std::clamp(bb[2] - rx, 1, page.cols - rx);
  const int rh = std::clamp(bb[3] - ry, 1, page.rows - ry);
  if (rw <= 0 || rh <= 0) return empty;

  cv::Mat crop = page(cv::Rect(rx, ry, rw, rh)).clone();  // contiguous BGR8

  std::vector<float> in_chw;
  slanext_preprocess(crop, in_chw);

  std::vector<float> feat(static_cast<std::size_t>(kTenc) * kCtx);
  try {
    const std::array<int64_t, 4> in_shape{1, 3, DST, DST};
    Ort::Value xv = Ort::Value::CreateTensor<float>(
        p_->mem, in_chw.data(), in_chw.size(), in_shape.data(), in_shape.size());
    const std::array<int64_t, 3> out_shape{1, kTenc, kCtx};
    Ort::Value ov = Ort::Value::CreateTensor<float>(
        p_->mem, feat.data(), feat.size(), out_shape.data(), out_shape.size());
    const char *ins[] = {p_->in_name.c_str()};
    const char *outs[] = {p_->out_name.c_str()};
    p_->sess->Run(Ort::RunOptions{nullptr}, ins, &xv, 1, outs, &ov, 1);
  } catch (const std::exception &e) {
    std::cerr << "[slanext-cpu] encoder run failed for region [" << rx << ',' << ry
              << ',' << rw << 'x' << rh << "] — DROPPED: " << e.what() << '\n';
    return empty;
  }

  return slanext_host_decode(feat.data(), weights_, dict_, rw, rh);
}

bool CpuSlanextTableRecognizer::load() {
  const std::string enc = env_or("TABLE_SLANEXT_ENCODER_ONNX", "");
  if (enc.empty()) return false;
  if (!std::filesystem::exists(enc)) {
    std::cerr << "[slanext-cpu] encoder ONNX missing: " << enc << '\n';
    return false;
  }
  const std::string dec = env_or("TABLE_SLANEXT_DECODER_BIN", default_decoder_bin(enc));
  const std::string dict = env_or("TABLE_SLANEXT_DICT", default_dict(enc));
  wired_ = std::make_unique<CpuSlanextEncoder>();
  if (!wired_->load(enc, dec, dict)) {
    wired_.reset();
    return false;
  }

  const std::string wenc = env_or("TABLE_SLANEXT_WIRELESS_ENCODER_ONNX", "");
  if (!wenc.empty()) {
    // A CONFIGURED wireless encoder that won't load is a soft downgrade to
    // wired-only — but never SILENTLY. Warn loudly so an operator who set the env
    // knows the wireless path is inactive (no-silent-failure).
    if (!std::filesystem::exists(wenc)) {
      std::cerr << "[slanext-cpu] WARNING: configured wireless encoder ONNX missing ("
                << wenc << ") — downgrading to wired-only\n";
    } else {
      const std::string wdec = env_or("TABLE_SLANEXT_WIRELESS_DECODER_BIN", default_decoder_bin(wenc));
      const std::string wdict = env_or("TABLE_SLANEXT_DICT", default_dict(wenc));
      auto w = std::make_unique<CpuSlanextEncoder>();
      if (w->load(wenc, wdec, wdict)) wireless_ = std::move(w);
      else
        std::cerr << "[slanext-cpu] WARNING: configured wireless encoder failed to load ("
                  << wenc << ") — downgrading to wired-only\n";
    }
  }
  std::cout << "[Pipeline] Table backend=slanext-cpu (ORT-CPU encoder + host decode"
            << (wireless_ ? ", wired+wireless" : ", wired") << ")\n";
  return true;
}

std::vector<router::TableResult>
CpuSlanextTableRecognizer::run(const cv::Mat &page, const std::vector<Box> &regions,
                               const std::vector<OCRResultItem> &page_ocr) {
  std::vector<router::TableResult> out;
  out.reserve(regions.size());

  for (std::size_t ti = 0; ti < regions.size(); ++ti) {
   try {
    const Box &region = regions[ti];
    int rx = INT_MAX, ry = INT_MAX, rax2 = INT_MIN, ray2 = INT_MIN;
    for (const auto &p : region.pts) {
      rx = std::min(rx, p[0]); ry = std::min(ry, p[1]);
      rax2 = std::max(rax2, p[0]); ray2 = std::max(ray2, p[1]);
    }
    rx = std::max(rx, 0); ry = std::max(ry, 0);  // match infer()'s clamp origin

    const auto sr = wired_->infer(page, region);

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

    std::vector<std::array<int, 8>> quads;
    quads.reserve(sr.cells.size());
    for (const auto &c : sr.cells) {
      std::array<int, 8> q = c.bbox;
      for (int k = 0; k < 4; ++k) { q[2 * k] += rx; q[2 * k + 1] += ry; }
      quads.push_back(q);
    }
    auto matched = match_cells_to_ocr(quads, region_ocr);

    if (cell_rec_ && !matched.empty()) {
      std::vector<Box> empty_boxes;
      std::vector<std::size_t> empty_ci;
      for (std::size_t ci = 0; ci < matched.size() && ci < quads.size(); ++ci) {
        if (!matched[ci].ocr_indices.empty()) continue;
        const auto &q = quads[ci];
        const int w = std::abs(q[2] - q[0]);
        const int h = std::abs(q[5] - q[1]);
        if (w < 4 || h < 4) continue;
        Box b;
        b.pts = {{{q[0], q[1]}, {q[2], q[3]}, {q[4], q[5]}, {q[6], q[7]}}};
        empty_boxes.push_back(b);
        empty_ci.push_back(ci);
      }
      if (!empty_boxes.empty()) {
        auto rec = cell_rec_->run(page, empty_boxes);
        for (std::size_t k = 0; k < rec.size() && k < empty_ci.size(); ++k) {
          const std::string &txt = rec[k].first;
          if (txt.empty() || rec[k].second < 0.5f) continue;
          region_texts.push_back(txt);
          matched[empty_ci[k]].ocr_indices.push_back(region_texts.size() - 1);
        }
      }
    }

    router::TableResult tr;
    tr.layout_id = -1;
    tr.html = reconstruct_html(sr.structure, matched, region_texts);
    tr.score = sr.structure_score;
    tr.box = region;
    out.push_back(std::move(tr));
   } catch (const std::exception &e) {
    // Per-region degrade, mirroring the GPU SlanextTableRecognizer: an ORT /
    // decode fault on ONE table region must not abort the whole page (every
    // other table lost) — also mirrors infer()'s graceful "region DROPPED"
    // path. One TableResult per region is pushed unconditionally so the
    // caller's layout_id stamping stays aligned; the empty html is counted as
    // degraded upstream.
    std::cerr << "[slanext-cpu] table region " << ti << " FAILED (" << e.what()
              << ") — region DROPPED, continuing\n";
    router::TableResult tr;
    tr.layout_id = -1;
    tr.box = regions[ti];  // html left empty -> degraded accounting
    out.push_back(std::move(tr));
   }
  }
  return out;
}

}  // namespace turbo_ocr::table
