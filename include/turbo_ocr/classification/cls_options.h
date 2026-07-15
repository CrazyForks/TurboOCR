#pragma once

// Shared textline-orientation classifier options (GPU + CPU pipelines).

#include <cctype>
#include <cstdlib>
#include <string>
#include <string_view>

namespace turbo_ocr::classification {

// True for every value env_bool_strict accepts as true (1/true/yes/on,
// case-insensitive). Must stay in sync with env_utils.h: the boot validator
// and this runtime reader MUST agree, or a validated-true value would
// silently run with the feature off.
[[nodiscard]] inline bool truthy_env_value(const char *v) {
  if (!v || !*v) return false;
  std::string s(v);
  for (auto &ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

// CLS_ALL_BOXES=1 — run the 0/180 orientation classifier on every crop
// instead of only vertical-looking ones (h >= 1.5*w). Off by default: the
// vertical-only gate exists because upright documents gain nothing from
// classifying horizontal lines, but it also means an upside-down horizontal
// line is never checked — scans with mixed per-line orientations need this.
// The value is validated strictly at boot (server_config); this helper only
// re-reads the already-validated env.
[[nodiscard]] inline bool cls_all_boxes_enabled() {
  static const bool e = truthy_env_value(std::getenv("CLS_ALL_BOXES"));
  return e;
}

// CLS_ONNX (GPU) / CLS_MODEL (CPU) accept a filesystem path or one of these
// shorthand names for the shipped textline-orientation variants. Returns the
// resolved path (shorthand -> bundled file), or `value` unchanged when it is
// not a known shorthand.
[[nodiscard]] inline std::string resolve_cls_shorthand(std::string_view value) {
  if (value == "x0_25") return "models/cls.onnx";
  if (value == "x1_0")  return "models/cls_x1_0.onnx";
  return std::string(value);
}

} // namespace turbo_ocr::classification
