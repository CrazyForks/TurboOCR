#include "turbo_ocr/formula/auto_cjk_formula.h"

#include <cstdint>
#include <filesystem>
#include <iostream>

#include "turbo_ocr/formula/ppformulanet_ort.h"

namespace turbo_ocr::formula {

// Is codepoint `cp` a CJK ideograph? CJK Unified (incl. Ext A), Compatibility
// Ideographs, and Ext B+ (SIP).
inline bool is_cjk_cp(uint32_t cp) noexcept {
  return (cp >= 0x3400 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
         (cp >= 0x20000 && cp <= 0x2FFFF);
}

CjkStat cjk_stats(std::string_view s) noexcept {
  CjkStat st;
  const auto *p = reinterpret_cast<const unsigned char *>(s.data());
  const auto *end = p + s.size();
  while (p < end) {
    uint32_t cp;
    int len;
    const unsigned char c = *p;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
    else { ++p; continue; }  // stray continuation / invalid lead
    if (p + len > end) break;
    bool ok = true;
    for (int i = 1; i < len; ++i) {
      if ((p[i] & 0xC0) != 0x80) { ok = false; break; }
      cp = (cp << 6) | (p[i] & 0x3F);
    }
    if (!ok) { ++p; continue; }
    p += len;
    ++st.total;
    if (is_cjk_cp(cp)) ++st.cjk;
  }
  return st;
}

bool text_has_cjk(std::string_view s) noexcept { return cjk_stats(s).cjk > 0; }

AutoCjkFormula::AutoCjkFormula()
    : fast_(std::make_unique<PPFormulaNetOrt>("ppformulanet_s")),
      cjk_(std::make_unique<PPFormulaNetOrt>("ppformulanet_plus_m")) {}

AutoCjkFormula::~AutoCjkFormula() noexcept = default;

bool AutoCjkFormula::load_model_dir(const std::string &model_dir) {
  namespace fs = std::filesystem;
  cjk_model_dir_ =
      (fs::path(model_dir).parent_path() / "ppformulanet_plus_m").string();
  if (!fast_->load_model_dir(model_dir)) {
    std::cerr << "[auto_cjk] FATAL: -S backend failed to load from " << model_dir
              << '\n';
    return false;
  }
  if (!cjk_->load_model_dir(cjk_model_dir_)) {
    std::cerr << "[auto_cjk] FATAL: plus-M backend failed to load from "
              << cjk_model_dir_
              << " (FORMULA_BACKEND=auto needs both models; use "
                 "FORMULA_BACKEND=ppformulanet_s for -S only)\n";
    return false;
  }
  std::cout << "[auto_cjk] loaded -S (" << model_dir << ") + plus-M ("
            << cjk_model_dir_ << "); per-crop CJK escalation active\n";
  return true;
}

bool AutoCjkFormula::load_tokenizer(const std::string &path) {
  namespace fs = std::filesystem;
  const std::string cjk_tok =
      (fs::path(cjk_model_dir_) / "tokenizer.json").string();
  return fast_->load_tokenizer(path) && cjk_->load_tokenizer(cjk_tok);
}

std::vector<FormulaEngineResult>
AutoCjkFormula::run(const GpuImage &page, const std::vector<Box> &boxes,
                    cudaStream_t stream) {
  // Fast pass: -S over every crop (EN/Latin — the common case).
  std::vector<FormulaEngineResult> res = fast_->run(page, boxes, stream);

  // Escalate a crop to plus-M when EITHER the page has CJK text (plus-M is the
  // stronger model — wins on hard math on CJK pages too, not just Chinese
  // glyphs) OR -S's own output shows CJK it mangled. On a pure-EN page neither
  // fires, so EN crops keep -S speed and accuracy (no regression).
  std::vector<Box> cjk_boxes;
  std::vector<std::size_t> cjk_idx;
  for (std::size_t i = 0; i < res.size() && i < boxes.size(); ++i) {
    if (res[i].ok && (page_has_cjk_ || text_has_cjk(res[i].latex))) {
      cjk_boxes.push_back(boxes[i]);
      cjk_idx.push_back(i);
    }
  }
  if (!cjk_boxes.empty()) {
    std::vector<FormulaEngineResult> cjk_res =
        cjk_->run(page, cjk_boxes, stream);
    for (std::size_t j = 0; j < cjk_res.size() && j < cjk_idx.size(); ++j)
      res[cjk_idx[j]] = std::move(cjk_res[j]);
  }
  return res;
}

bool AutoCjkFormula::is_ready() const noexcept {
  return fast_ && cjk_ && fast_->is_ready() && cjk_->is_ready();
}

} // namespace turbo_ocr::formula
