#include "turbo_ocr/formula/vlm_text_rerec.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "simdutf.h"

namespace turbo_ocr::formula {

namespace {

const char *_env(const char *k, const char *dflt) {
  const char *v = std::getenv(k);
  return (v && v[0]) ? v : dflt;
}

float _env_float(const char *k, float dflt) {
  const char *v = std::getenv(k);
  if (!v || !v[0]) return dflt;
  try { return std::stof(v); } catch (...) { return dflt; }
}

int _env_int(const char *k, int dflt) {
  const char *v = std::getenv(k);
  if (!v || !v[0]) return dflt;
  try { return std::stoi(v); } catch (...) { return dflt; }
}

std::string _to_base64(const std::vector<uint8_t> &bin) {
  size_t n = simdutf::base64_length_from_binary(bin.size());
  std::string out(n, '\0');
  simdutf::binary_to_base64(reinterpret_cast<const char *>(bin.data()),
                             bin.size(), out.data());
  return out;
}

size_t _curl_write(char *ptr, size_t sz, size_t nmemb, void *ud) {
  auto *s = static_cast<std::string *>(ud);
  s->append(ptr, sz * nmemb);
  return sz * nmemb;
}

// Returns (ok, response_body). Reentrant — creates fresh CURL handle each call.
std::pair<bool, std::string>
http_post(const std::string &url, const std::string &body, int timeout_s) {
  CURL *c = curl_easy_init();
  if (!c) return {false, {}};
  std::string resp;
  struct curl_slist *hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
  hdrs = curl_slist_append(hdrs, "Accept: application/json");
  curl_easy_setopt(c, CURLOPT_URL,           url.c_str());
  curl_easy_setopt(c, CURLOPT_POST,          1L);
  curl_easy_setopt(c, CURLOPT_POSTFIELDS,    body.c_str());
  curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE, (long)body.size());
  curl_easy_setopt(c, CURLOPT_HTTPHEADER,    hdrs);
  curl_easy_setopt(c, CURLOPT_TIMEOUT,       (long)timeout_s);
  curl_easy_setopt(c, CURLOPT_NOSIGNAL,      1L);
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, _curl_write);
  curl_easy_setopt(c, CURLOPT_WRITEDATA,     &resp);
  curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
  CURLcode rc = curl_easy_perform(c);
  long status = 0;
  bool ok = false;
  if (rc == CURLE_OK) {
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &status);
    ok = (status >= 200 && status < 300);
  } else {
    std::cerr << "[vlm_text_rerec] curl error: " << curl_easy_strerror(rc) << '\n';
  }
  curl_slist_free_all(hdrs);
  curl_easy_cleanup(c);
  return {ok, std::move(resp)};
}

// One-shot VLM request for a single PNG crop.
// Returns the raw stripped assistant message, or empty on failure.
std::string single_vlm_ocr(const std::vector<uint8_t> &png,
                            const VlmTextRerecConfig    &cfg) {
  std::string b64 = _to_base64(png);
  nlohmann::json body = {
    {"model",       cfg.model},
    {"max_tokens",  cfg.max_tokens},
    {"temperature", 0.0},
    {"messages", nlohmann::json::array({
        {{"role", "user"},
         {"content", nlohmann::json::array({
             {{"type", "image_url"},
              {"image_url", {{"url", "data:image/png;base64," + b64}}}},
             {{"type", "text"}, {"text", cfg.prompt}},
         })}},
    })},
  };
  auto [ok, resp] = http_post(cfg.base_url + "/v1/chat/completions",
                              body.dump(), cfg.timeout_s);
  if (!ok) return {};
  try {
    auto j = nlohmann::json::parse(resp);
    std::string msg = j.at("choices").at(0).at("message").at("content")
                        .get<std::string>();
    // Strip surrounding whitespace.
    while (!msg.empty() && (msg.front() == ' ' || msg.front() == '\n' ||
                            msg.front() == '\r'))
      msg.erase(msg.begin());
    while (!msg.empty() && (msg.back() == ' ' || msg.back() == '\n' ||
                            msg.back() == '\r'))
      msg.pop_back();
    return msg;
  } catch (const std::exception &e) {
    std::cerr << "[vlm_text_rerec] parse error: " << e.what() << '\n';
    return {};
  }
}

} // namespace

VlmTextRerecConfig VlmTextRerecConfig::from_env() {
  VlmTextRerecConfig cfg;
  cfg.threshold = _env_float("LOW_CONF_VLM_TEXT_THRESHOLD", 0.0f);
  if (cfg.threshold <= 0.0f) return cfg; // feature disabled

  cfg.base_url   = _env("VLLM_BASE_URL", "http://localhost:8000");
  while (!cfg.base_url.empty() && cfg.base_url.back() == '/')
    cfg.base_url.pop_back();
  cfg.model      = _env("VLLM_MODEL", "openbmb/MiniCPM-V-4.6");
  cfg.prompt     = _env("VLM_TEXT_PROMPT", "OCR:");
  cfg.timeout_s  = std::max(3, _env_int("VLM_TEXT_TIMEOUT_S", 10));
  cfg.max_tokens = std::max(8, _env_int("VLM_TEXT_MAX_TOKENS", 64));

  std::cout << "[vlm_text_rerec] enabled: threshold=" << cfg.threshold
            << " url=" << cfg.base_url
            << " model=" << cfg.model
            << " prompt=\"" << cfg.prompt << "\"\n";
  return cfg;
}

std::vector<std::string>
vlm_rerec_text_crops(const std::vector<cv::Mat> &crops,
                     const VlmTextRerecConfig    &cfg) {
  std::vector<std::string> out(crops.size());
  if (crops.empty() || !cfg.enabled()) return out;

  // PNG-encode all crops up front (faster than inside threads).
  std::vector<std::vector<uint8_t>> pngs(crops.size());
  for (size_t i = 0; i < crops.size(); ++i) {
    if (crops[i].empty()) continue;
    std::vector<int> params{cv::IMWRITE_PNG_COMPRESSION, 1};
    cv::imencode(".png", crops[i], pngs[i], params);
  }

  // Parallel HTTP: one thread per crop. vLLM continuous batching merges them.
  std::vector<std::thread> threads;
  threads.reserve(crops.size());
  for (size_t i = 0; i < crops.size(); ++i) {
    threads.emplace_back([&, i] {
      if (pngs[i].empty()) return;
      out[i] = single_vlm_ocr(pngs[i], cfg);
    });
  }
  for (auto &t : threads) t.join();
  return out;
}

} // namespace turbo_ocr::formula
