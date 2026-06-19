#pragma once

#include <array>
#include <string_view>

#include "turbo_ocr/detection/det_config.h"

namespace turbo_ocr::server {

using turbo_ocr::detection::DbParams;
using turbo_ocr::detection::DetResizeParams;

// Origin of a registry model. v6 entries are the script-agnostic PP-OCRv6
// tiers (Latin + Chinese + Japanese); V5Lang entries are the retained legacy
// recognizers for scripts v6 does not cover.
enum class ModelFamily { V6, V5Lang };

// Per-model detection inference config: the official PaddleOCR values for this
// model's detector (resize policy + DB post-processing). Env vars still override
// each field at read time (read_det_resize/read_db_params).
struct DetInferConfig {
  DetResizeParams resize;
  DbParams db;
};

// Official PaddleOCR det config for the PP-OCRv6 tiers whose box_thresh is 0.45
// (medium, small) — and reused by the V5Lang rows, which serve through the
// shared default v6 detector (models/det.onnx). resize policy is the official
// PaddleOCR "min"/64 (native resolution); max_side_limit 1280 is the pooled-
// server practical cap (official 4000 OOMs the pre-allocated pool — see
// det_config.h kDetResizeDefault). DB is thresh 0.2 / box 0.45 / unclip 1.4.
// Env-overridable per field at read time.
inline constexpr DetInferConfig kV6DetConfig{
    DetResizeParams{"min", 64, 1280}, DbParams{0.2f, 0.45f, 1.4f}};

// Tiny tier differs only in box_thresh (0.40, per its inference.yml).
inline constexpr DetInferConfig kV6DetConfigTiny{
    DetResizeParams{"min", 64, 1280}, DbParams{0.2f, 0.40f, 1.4f}};

// One selectable OCR model. Adding a model is a single row.
//
// `det_path` empty => the default detector (`models/det.onnx`, v6 medium-tier).
// Each v6 tier ships its own detector (det.onnx / det_small.onnx /
// det_tiny.onnx); the V5Lang rec recognizers carry no detector of their own and
// fall back to the default, reusing the v6 det config.
struct ModelEntry {
  std::string_view name;
  std::string_view rec_path;
  std::string_view dict_path;
  std::string_view det_path;  // empty => kDefaultDet
  ModelFamily family;
  DetInferConfig det_cfg;
};

inline constexpr std::string_view kDefaultDet = "models/det.onnx";
inline constexpr std::string_view kDefaultModel = "tiny";

// The model registry. v6 tiers share the medium/small dictionary except tiny,
// which ships its own (6,904 chars vs 18,708). Legacy scripts keep their v5
// recognizer + dict under models/rec/<name>/ and the shared v6 detector + its
// det config. Each model carries its official det inference values.
inline constexpr std::array<ModelEntry, 8> kModelCatalog{{
    {"medium", "models/rec.onnx",       "models/keys.txt",      "models/det.onnx",       ModelFamily::V6,     kV6DetConfig},
    {"small",  "models/rec_small.onnx", "models/keys.txt",      "models/det_small.onnx", ModelFamily::V6,     kV6DetConfig},
    {"tiny",   "models/rec_tiny.onnx",  "models/keys_tiny.txt", "models/det_tiny.onnx",  ModelFamily::V6,     kV6DetConfigTiny},
    {"arabic", "models/rec/arabic/rec.onnx", "models/rec/arabic/dict.txt", "", ModelFamily::V5Lang, kV6DetConfig},
    {"eslav",  "models/rec/eslav/rec.onnx",  "models/rec/eslav/dict.txt",  "", ModelFamily::V5Lang, kV6DetConfig},
    {"korean", "models/rec/korean/rec.onnx", "models/rec/korean/dict.txt", "", ModelFamily::V5Lang, kV6DetConfig},
    {"thai",   "models/rec/thai/rec.onnx",   "models/rec/thai/dict.txt",   "", ModelFamily::V5Lang, kV6DetConfig},
    {"greek",  "models/rec/greek/rec.onnx",  "models/rec/greek/dict.txt",  "", ModelFamily::V5Lang, kV6DetConfig},
}};

// Look up a model by name. Returns nullptr if no entry matches.
[[nodiscard]] inline constexpr const ModelEntry *find_model(
    std::string_view name) noexcept {
  for (const auto &e : kModelCatalog)
    if (e.name == name) return &e;
  return nullptr;
}

// The default model must exist in the catalog: resolve_model and
// lang_alias_entry dereference find_model(kDefaultModel) unconditionally.
static_assert(find_model(kDefaultModel) != nullptr,
              "kDefaultModel is not present in kModelCatalog");

// Resolve an entry's detector: its own det_path, or the default detector when
// empty. Always a valid non-empty path.
[[nodiscard]] inline constexpr std::string_view model_det_path(
    const ModelEntry &e) noexcept {
  return e.det_path.empty() ? kDefaultDet : e.det_path;
}

} // namespace turbo_ocr::server
