#pragma once

// VLM-based low-confidence text line re-recognition.
// Compiled only when BUILD_VLM_FORMULA=ON (needs libcurl).
// Entry point: vlm_rerec_text_crops().

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/common/box.h"

namespace turbo_ocr::formula {

// Configuration for the VLM text re-recognizer. Populated once from env vars
// and reused per call. Thread-safe after construction (read-only fields).
struct VlmTextRerecConfig {
  std::string base_url;     // VLM endpoint base, e.g. "http://127.0.0.1:8003"
  std::string model;        // model id on the server
  std::string prompt;       // prompt sent with each crop, e.g. "OCR:"
  int         timeout_s  = 10;
  int         max_tokens = 64;
  float       threshold  = 0.0f; // lines below this confidence are re-sent

  // Read config from env vars (LOW_CONF_VLM_TEXT_THRESHOLD, VLM_TEXT_PROMPT,
  // VLLM_BASE_URL, VLLM_MODEL). Returns a config with threshold=0 when the
  // feature is disabled. Logs warnings to stderr on parse errors.
  static VlmTextRerecConfig from_env();
  bool enabled() const noexcept { return threshold > 0.0f && !base_url.empty(); }
};

// Re-recognize a batch of text crop images via VLM.
// `crops` is a list of already-extracted BGR cv::Mat crops.
// Returns one string per crop: non-empty = use this instead of the CNN result,
// empty = keep the original CNN result.
// Parallel HTTP: one thread per crop (matches VLMFormula pattern).
// Network / parse failures produce empty strings (safe no-op for the caller).
std::vector<std::string>
vlm_rerec_text_crops(const std::vector<cv::Mat>& crops,
                     const VlmTextRerecConfig&    cfg);

} // namespace turbo_ocr::formula
