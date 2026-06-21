#include "turbo_ocr/formula/vlm_formula.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include "simdutf.h"
#include "turbo_ocr/common/cuda_check.h"

namespace turbo_ocr::formula {

namespace {

const char *get_env(const char *k, const char *dflt) {
  const char *v = std::getenv(k);
  if (v == nullptr || v[0] == '\0') return dflt;
  return v;
}

int get_env_int(const char *k, int dflt) {
  const char *v = std::getenv(k);
  if (v == nullptr || v[0] == '\0') return dflt;
  try { return std::stoi(v); } catch (...) { return dflt; }
}

// ---- profiling helpers ------------------------------------------------
using Clock    = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
inline int64_t us_since(TimePoint t0) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             Clock::now() - t0).count();
}

static const bool kProfEnabled = [] {
  const char *v = std::getenv("VLM_PROFILE");
  return v && v[0] && v[0] != '0';
}();

static const std::string kProfPath = [] {
  const char *v = std::getenv("VLM_PROFILE_PATH");
  return std::string(v && v[0] ? v : "/tmp/vlm_profile.jsonl");
}();

static std::mutex g_prof_mu;

void prof_append(const nlohmann::json &rec) {
  std::lock_guard<std::mutex> lk(g_prof_mu);
  std::ofstream f(kProfPath, std::ios::app);
  f << rec.dump() << '\n';
}
// -----------------------------------------------------------------------

std::string to_base64(const std::vector<uint8_t> &bin) {
  size_t out_len = simdutf::base64_length_from_binary(bin.size());
  std::string out(out_len, '\0');
  simdutf::binary_to_base64(reinterpret_cast<const char *>(bin.data()),
                             bin.size(), out.data());
  return out;
}

// Initialize libcurl exactly once. Reuse easy handles per-call.
struct CurlGlobalInit {
  CurlGlobalInit() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CurlGlobalInit() { curl_global_cleanup(); }
};
CurlGlobalInit g_curl_init;

size_t curl_write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
  auto *buf = static_cast<std::string *>(userdata);
  buf->append(ptr, size * nmemb);
  return size * nmemb;
}

// Run an HTTP POST with JSON body. Returns (ok, http_status, body).
struct HttpResp {
  bool        ok      = false;
  long        status  = 0;
  std::string body;
};

HttpResp http_post_json(const std::string &url, const std::string &json_body,
                        int timeout_s) {
  HttpResp r;
  CURL *curl = curl_easy_init();
  if (curl == nullptr) return r;
  struct curl_slist *hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
  hdrs = curl_slist_append(hdrs, "Accept: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)json_body.size());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, (long)timeout_s);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &r.body);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  CURLcode rc = curl_easy_perform(curl);
  if (rc == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &r.status);
    r.ok = (r.status >= 200 && r.status < 300);
  } else {
    std::cerr << "[VLMFormula] curl error: " << curl_easy_strerror(rc) << '\n';
  }
  curl_slist_free_all(hdrs);
  curl_easy_cleanup(curl);
  return r;
}

HttpResp http_get(const std::string &url, int timeout_s) {
  HttpResp r;
  CURL *curl = curl_easy_init();
  if (curl == nullptr) return r;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, (long)timeout_s);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &r.body);
  CURLcode rc = curl_easy_perform(curl);
  if (rc == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &r.status);
    r.ok = (r.status >= 200 && r.status < 300);
  }
  curl_easy_cleanup(curl);
  return r;
}

// Pull LaTeX out of the assistant message.
std::string extract_latex(const std::string &msg) {
  static const std::regex re_fence(R"(```(?:latex|tex|math)?\s*\n?([\s\S]*?)```)",
                                    std::regex::ECMAScript);
  std::smatch m;
  if (std::regex_search(msg, m, re_fence) && m.size() >= 2) {
    std::string s = m[1].str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
      s.pop_back();
    return s;
  }
  static const std::regex re_disp(R"(\$\$([\s\S]*?)\$\$)");
  if (std::regex_search(msg, m, re_disp) && m.size() >= 2) return m[1].str();
  static const std::regex re_brk(R"(\\\[([\s\S]*?)\\\])");
  if (std::regex_search(msg, m, re_brk) && m.size() >= 2) return m[1].str();
  static const std::regex re_inline(R"(\$([^\$\n]+)\$)");
  if (std::regex_search(msg, m, re_inline) && m.size() >= 2) return m[1].str();
  std::string s = msg;
  while (!s.empty() && (s.front() == ' ' || s.front() == '\n' || s.front() == '\r'))
    s.erase(s.begin());
  while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r'))
    s.pop_back();
  for (auto pre : {"LaTeX:", "Latex:", "latex:", "Answer:", "answer:"}) {
    if (s.rfind(pre, 0) == 0) { s.erase(0, std::strlen(pre)); break; }
  }
  while (!s.empty() && (s.front() == ' ' || s.front() == '\n')) s.erase(s.begin());
  return s;
}

// PNG-encode a BGR crop. Returns empty vector on failure.
std::vector<uint8_t> encode_png_bgr(const uint8_t *data, int w, int h, int stride) {
  cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t *>(data), (size_t)stride);
  std::vector<uint8_t> out;
  std::vector<int> params{cv::IMWRITE_PNG_COMPRESSION, 1};
  if (!cv::imencode(".png", src, out, params)) return {};
  return out;
}

nlohmann::json make_image_block(const std::string &b64_png) {
  return {
      {"type", "image_url"},
      {"image_url", {{"url", std::string("data:image/png;base64,") + b64_png}}},
  };
}

} // namespace

VLMFormula::VLMFormula() = default;
VLMFormula::~VLMFormula() noexcept = default;

bool VLMFormula::load_model_dir(const std::string &/*model_dir*/) {
  base_url_   = get_env("VLLM_BASE_URL", "http://localhost:8000");
  while (!base_url_.empty() && base_url_.back() == '/') base_url_.pop_back();
  model_      = get_env("VLLM_MODEL", "openbmb/MiniCPM-V-4.6");
  prompt_     = get_env("VLLM_FORMULA_PROMPT",
                        "Extract the LaTeX of this formula image. Output ONLY "
                        "the LaTeX inside ``` fences, no commentary.");
  batch_      = std::max(1, get_env_int("VLLM_FORMULA_BATCH", 8));
  timeout_s_  = std::max(1, get_env_int("VLLM_FORMULA_TIMEOUT_S", 30));
  max_tokens_ = std::max(16, get_env_int("VLLM_FORMULA_MAX_TOKENS", 512));

  HttpResp r = http_get(base_url_ + "/v1/models", 5);
  if (!r.ok) {
    std::cerr << "[VLMFormula] /v1/models unreachable at " << base_url_
              << " (status=" << r.status << ") — formulas disabled\n";
    return false;
  }
  try {
    auto j = nlohmann::json::parse(r.body);
    if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
      std::string first = j["data"][0].value("id", "");
      if (!first.empty()) {
        if (!std::getenv("VLLM_MODEL") || std::getenv("VLLM_MODEL")[0] == '\0') {
          model_ = first;
        }
        std::cout << "[VLMFormula] endpoint=" << base_url_
                  << " server_model=" << first
                  << " using_model=" << model_ << '\n';
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[VLMFormula] /v1/models parse warning: " << e.what() << '\n';
  }

  ready_ = true;
  std::cout << "[VLMFormula] ready: " << base_url_ << " model=" << model_
            << " batch=" << batch_ << " timeout=" << timeout_s_ << "s"
            << " max_tokens=" << max_tokens_ << '\n';
  if (kProfEnabled) {
    std::cout << "[VLMFormula] VLM_PROFILE=1 — writing to " << kProfPath << '\n';
  }
  return true;
}

bool VLMFormula::load_tokenizer(const std::string &/*path*/) { return true; }

bool VLMFormula::single_request(const std::vector<uint8_t> &crop_png,
                                 std::string &out_latex) {
  std::string b64 = to_base64(crop_png);
  nlohmann::json body = {
      {"model", model_},
      {"max_tokens", max_tokens_},
      {"temperature", 0.0},
      {"messages", nlohmann::json::array({
          {{"role", "user"},
           {"content", nlohmann::json::array({
               make_image_block(b64),
               {{"type", "text"}, {"text", prompt_}},
           })}},
      })},
  };
  HttpResp r = http_post_json(base_url_ + "/v1/chat/completions",
                              body.dump(), timeout_s_);
  if (!r.ok) {
    std::cerr << "[VLMFormula] chat status=" << r.status
              << " body=" << r.body.substr(0, std::min<size_t>(r.body.size(), 200))
              << '\n';
    return false;
  }
  try {
    auto j = nlohmann::json::parse(r.body);
    std::string msg = j.at("choices").at(0).at("message").at("content").get<std::string>();
    out_latex = extract_latex(msg);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[VLMFormula] response parse failed: " << e.what()
              << " body=" << r.body.substr(0, std::min<size_t>(r.body.size(), 200))
              << '\n';
    return false;
  }
}

bool VLMFormula::batched_request(
    const std::vector<std::vector<uint8_t>> &crops_png,
    std::vector<std::string> &out_latex) {
  out_latex.clear();
  if (crops_png.empty()) return true;

  nlohmann::json content = nlohmann::json::array();
  for (size_t i = 0; i < crops_png.size(); ++i) {
    content.push_back(make_image_block(to_base64(crops_png[i])));
  }
  std::string multi_prompt =
      "You will see " + std::to_string(crops_png.size()) +
      " formula images in order. Extract the LaTeX of each. Output ONE LaTeX "
      "expression per image, each on its own line wrapped in ``` fences, in "
      "the same order as the images. No commentary, no numbering.";
  content.push_back({{"type", "text"}, {"text", multi_prompt}});

  nlohmann::json body = {
      {"model", model_},
      {"max_tokens", max_tokens_ * (int)crops_png.size()},
      {"temperature", 0.0},
      {"messages", nlohmann::json::array({
          {{"role", "user"}, {"content", content}},
      })},
  };
  HttpResp r = http_post_json(base_url_ + "/v1/chat/completions",
                              body.dump(), timeout_s_ * std::max(1, (int)crops_png.size() / 4));
  if (!r.ok) return false;
  try {
    auto j = nlohmann::json::parse(r.body);
    std::string msg = j.at("choices").at(0).at("message").at("content").get<std::string>();
    static const std::regex re_fence(R"(```(?:latex|tex|math)?\s*\n?([\s\S]*?)```)");
    auto begin = std::sregex_iterator(msg.begin(), msg.end(), re_fence);
    auto end   = std::sregex_iterator();
    std::vector<std::string> hits;
    for (auto it = begin; it != end; ++it) {
      std::string s = (*it)[1].str();
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
      hits.push_back(std::move(s));
    }
    if (hits.size() == crops_png.size()) {
      out_latex = std::move(hits);
      return true;
    }
    std::cerr << "[VLMFormula] batched response had " << hits.size()
              << " fences but expected " << crops_png.size()
              << " — falling back to per-crop\n";
    return false;
  } catch (const std::exception &e) {
    std::cerr << "[VLMFormula] batched parse failed: " << e.what() << '\n';
    return false;
  }
}

std::vector<FormulaEngineResult>
VLMFormula::run(const GpuImage &page, const std::vector<Box> &boxes,
                cudaStream_t stream) {
  std::vector<FormulaEngineResult> out;
  if (boxes.empty()) return out;
  if (page.empty()) {
    std::cerr << "[VLMFormula] empty page\n";
    return out;
  }
  if (!ready_) {
    std::cerr << "[VLMFormula] not ready\n";
    return out;
  }

  // ---- profiling state for this run() call ----------------------------
  const auto t_run_start = Clock::now();
  int64_t t_mutex_wait_us   = 0;
  int64_t t_d2h_us          = 0;
  int64_t t_png_total_us    = 0;
  int64_t t_b64_total_us    = 0;
  int64_t t_json_build_us   = 0;
  int64_t t_http_wall_us    = 0;
  int64_t t_response_parse_us = 0;
  size_t  bytes_sent_total  = 0;
  size_t  bytes_recv_total  = 0;
  std::vector<int64_t> t_png_per_crop_us;
  std::vector<int64_t> t_http_per_crop_us;
  std::vector<int64_t> t_parse_per_crop_us;
  // -- end profiling state ----------------------------------------------

  const auto t_mu_try = Clock::now();
  std::lock_guard<std::mutex> lk(call_mu_);
  if (kProfEnabled) t_mutex_wait_us = us_since(t_mu_try);

  // D2H
  const auto t_d2h_start = Clock::now();
  const size_t need = static_cast<size_t>(page.rows) * page.step;
  if (host_page_.size() < need) host_page_.resize(need);
  if (cudaSuccess !=
      cudaMemcpyAsync(host_page_.data(), page.data, need,
                      cudaMemcpyDeviceToHost, stream)) {
    std::cerr << "[VLMFormula] page D2H failed\n";
    return out;
  }
  cudaStreamSynchronize(stream);
  if (kProfEnabled) t_d2h_us = us_since(t_d2h_start);

  // PNG-encode every crop up front.
  const auto t_png_start = Clock::now();
  std::vector<std::vector<uint8_t>> crops_png;
  crops_png.reserve(boxes.size());
  if (kProfEnabled) t_png_per_crop_us.reserve(boxes.size());
  for (const auto &b : boxes) {
    auto r = aabb(b);
    int x0 = std::clamp(r[0], 0, std::max(0, page.cols - 1));
    int y0 = std::clamp(r[1], 0, std::max(0, page.rows - 1));
    int x1 = std::clamp(r[2], x0, page.cols);
    int y1 = std::clamp(r[3], y0, page.rows);
    int w = std::max(1, x1 - x0);
    int h = std::max(1, y1 - y0);
    const uint8_t *src = host_page_.data() + (size_t)y0 * page.step + (size_t)x0 * 3;
    const auto t_one_png = Clock::now();
    auto png = encode_png_bgr(src, w, h, (int)page.step);
    if (kProfEnabled) t_png_per_crop_us.push_back(us_since(t_one_png));
    crops_png.push_back(std::move(png));
  }
  if (kProfEnabled) t_png_total_us = us_since(t_png_start);

  out.resize(boxes.size());

  // Parallel HTTP per batch chunk.
  struct Job {
    size_t off;
    std::vector<std::vector<uint8_t>> chunk;
    std::vector<std::string> results;
    // profiling
    int64_t t_b64_us   = 0;
    int64_t t_json_us  = 0;
    int64_t t_http_us  = 0;
    int64_t t_parse_us = 0;
    size_t  sent_bytes = 0;
    size_t  recv_bytes = 0;
    std::vector<int64_t> http_per_crop;
    std::vector<int64_t> parse_per_crop;
  };
  std::vector<Job> jobs;
  for (size_t off = 0; off < crops_png.size(); off += (size_t)batch_) {
    size_t end = std::min(crops_png.size(), off + (size_t)batch_);
    jobs.push_back({off,
                    {crops_png.begin() + off, crops_png.begin() + end},
                    {}});
  }

  std::vector<std::thread> workers;
  workers.reserve(jobs.size());
  for (auto &job : jobs) {
    workers.emplace_back([this, &job] {
      // --- batched_request with per-job timing ---
      // We instrument the per-crop single_request path (the common fallback).
      // For the batched path we measure the whole round-trip.
      const bool do_prof = kProfEnabled;

      // b64 + json build are done inside single_request / batched_request;
      // we separate them here for the single_request fallback path.
      auto do_single = [&](size_t i, std::string &result) {
        // b64
        const auto t_b64 = Clock::now();
        std::string b64 = to_base64(job.chunk[i]);
        const int64_t dt_b64 = do_prof ? us_since(t_b64) : 0;

        // json build
        const auto t_json = Clock::now();
        nlohmann::json body = {
            {"model", model_},
            {"max_tokens", max_tokens_},
            {"temperature", 0.0},
            {"messages", nlohmann::json::array({
                {{"role", "user"},
                 {"content", nlohmann::json::array({
                     make_image_block(b64),
                     {{"type", "text"}, {"text", prompt_}},
                 })}},
            })},
        };
        std::string json_str = body.dump();
        const int64_t dt_json = do_prof ? us_since(t_json) : 0;

        // http
        const auto t_http = Clock::now();
        HttpResp r = http_post_json(base_url_ + "/v1/chat/completions",
                                    json_str, timeout_s_);
        const int64_t dt_http = do_prof ? us_since(t_http) : 0;

        // parse
        const auto t_parse = Clock::now();
        if (r.ok) {
          try {
            auto j = nlohmann::json::parse(r.body);
            std::string msg = j.at("choices").at(0).at("message").at("content").get<std::string>();
            result = extract_latex(msg);
          } catch (...) { result = ""; }
        }
        const int64_t dt_parse = do_prof ? us_since(t_parse) : 0;

        if (do_prof) {
          job.t_b64_us   += dt_b64;
          job.t_json_us  += dt_json;
          job.t_http_us  += dt_http;
          job.t_parse_us += dt_parse;
          job.sent_bytes += json_str.size();
          job.recv_bytes += r.body.size();
          job.http_per_crop.push_back(dt_http);
          job.parse_per_crop.push_back(dt_parse);
        }
      };

      // Try batched first. If it fails (or batch=1), fall to per-crop.
      if (job.chunk.size() == 1) {
        job.results.resize(1);
        do_single(0, job.results[0]);
      } else {
        // Attempt batched_request — on failure fall back
        bool batched_ok = false;
        if (!kProfEnabled) {
          batched_ok = batched_request(job.chunk, job.results);
        } else {
          // time the batched attempt as a whole
          const auto t_b = Clock::now();
          batched_ok = batched_request(job.chunk, job.results);
          if (batched_ok) {
            job.t_http_us = us_since(t_b);
            // approximate: no per-crop breakdown for batched path
          }
        }
        if (!batched_ok) {
          job.results.assign(job.chunk.size(), std::string{});
          // Zero profiling accumulators so per-crop adds are clean
          if (kProfEnabled) {
            job.t_b64_us = job.t_json_us = job.t_http_us = job.t_parse_us = 0;
            job.sent_bytes = job.recv_bytes = 0;
            job.http_per_crop.clear();
            job.parse_per_crop.clear();
          }
          std::vector<std::thread> inner;
          inner.reserve(job.chunk.size());
          for (size_t i = 0; i < job.chunk.size(); ++i) {
            inner.emplace_back([&, i] {
              do_single(i, job.results[i]);
            });
          }
          for (auto &t : inner) t.join();
        }
      }
    });
  }
  for (auto &t : workers) t.join();

  // Collect results + profiling accumulators
  for (auto &job : jobs) {
    for (size_t i = 0; i < job.chunk.size(); ++i) {
      FormulaEngineResult r;
      r.latex = std::move(job.results[i]);
      r.token_count = r.latex.size();
      r.hit_eos = !r.latex.empty();
      out[job.off + i] = std::move(r);
    }
    if (kProfEnabled) {
      t_b64_total_us    += job.t_b64_us;
      t_json_build_us   += job.t_json_us;
      t_http_wall_us    += job.t_http_us;
      t_response_parse_us += job.t_parse_us;
      bytes_sent_total  += job.sent_bytes;
      bytes_recv_total  += job.recv_bytes;
      for (auto v : job.http_per_crop)   t_http_per_crop_us.push_back(v);
      for (auto v : job.parse_per_crop)  t_parse_per_crop_us.push_back(v);
    }
  }

  if (kProfEnabled) {
    const int64_t t_total_us = us_since(t_run_start);
    nlohmann::json rec = {
        {"backend",            "formula"},
        {"n_crops",            (int)boxes.size()},
        {"t_mutex_wait_us",    t_mutex_wait_us},
        {"t_d2h_us",           t_d2h_us},
        {"t_png_total_us",     t_png_total_us},
        {"t_png_per_crop_us",  t_png_per_crop_us},
        {"t_b64_total_us",     t_b64_total_us},
        {"t_json_build_us",    t_json_build_us},
        {"t_http_wall_us",     t_http_wall_us},
        {"t_http_per_crop_us", t_http_per_crop_us},
        {"t_response_parse_us",t_response_parse_us},
        {"t_parse_per_crop_us",t_parse_per_crop_us},
        {"t_total_us",         t_total_us},
        {"bytes_sent_total",   (int64_t)bytes_sent_total},
        {"bytes_recv_total",   (int64_t)bytes_recv_total},
    };
    prof_append(rec);
    std::cerr << "[VLMFormula:PROF] n=" << boxes.size()
              << " total=" << t_total_us << "us"
              << " mutex=" << t_mutex_wait_us << "us"
              << " d2h=" << t_d2h_us << "us"
              << " png=" << t_png_total_us << "us"
              << " b64=" << t_b64_total_us << "us"
              << " json=" << t_json_build_us << "us"
              << " http=" << t_http_wall_us << "us"
              << " parse=" << t_response_parse_us << "us"
              << '\n';
  }

  return out;
}

} // namespace turbo_ocr::formula
