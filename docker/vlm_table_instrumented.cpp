#include "turbo_ocr/table/vlm_table.h"

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

namespace turbo_ocr::table {

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

using Clock     = std::chrono::steady_clock;
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

std::string to_base64(const std::vector<uint8_t> &bin) {
  size_t out_len = simdutf::base64_length_from_binary(bin.size());
  std::string out(out_len, '\0');
  simdutf::binary_to_base64(reinterpret_cast<const char *>(bin.data()),
                             bin.size(), out.data());
  return out;
}

size_t curl_write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
  auto *buf = static_cast<std::string *>(userdata);
  buf->append(ptr, size * nmemb);
  return size * nmemb;
}

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
    std::cerr << "[VLMTable] curl error: " << curl_easy_strerror(rc) << '\n';
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

std::string html_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&':  out += "&amp;";  break;
      case '<':  out += "&lt;";   break;
      case '>':  out += "&gt;";   break;
      case '"':  out += "&quot;"; break;
      case '\'': out += "&#39;";  break;
      default:   out += c;
    }
  }
  return out;
}

std::string trim(const std::string &s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

} // namespace

// ---------------------------------------------------------------------------
// OTSL to HTML
// ---------------------------------------------------------------------------

namespace {

constexpr std::string_view kFcel = "<fcel>";
constexpr std::string_view kEcel = "<ecel>";
constexpr std::string_view kNl   = "<nl>";
constexpr std::string_view kLcel = "<lcel>";
constexpr std::string_view kUcel = "<ucel>";
constexpr std::string_view kXcel = "<xcel>";

enum class OtslTok { Fcel, Ecel, Nl, Lcel, Ucel, Xcel };

struct OtslElement {
  OtslTok     tok;
  std::string text;
};

std::vector<OtslElement> parse_otsl(const std::string &otsl) {
  std::vector<OtslElement> out;
  static const std::regex re(R"((<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>))");
  auto begin = std::sregex_iterator(otsl.begin(), otsl.end(), re);
  auto end2  = std::sregex_iterator();
  std::vector<std::pair<size_t, std::string>> matches;
  for (auto it = begin; it != end2; ++it) {
    matches.emplace_back(static_cast<size_t>(it->position(0)), it->str(0));
  }
  for (size_t i = 0; i < matches.size(); ++i) {
    const std::string &tag = matches[i].second;
    OtslTok t;
    if (tag == kFcel) t = OtslTok::Fcel;
    else if (tag == kEcel) t = OtslTok::Ecel;
    else if (tag == kNl)   t = OtslTok::Nl;
    else if (tag == kLcel) t = OtslTok::Lcel;
    else if (tag == kUcel) t = OtslTok::Ucel;
    else                    t = OtslTok::Xcel;
    OtslElement e{t, {}};
    if (t == OtslTok::Fcel) {
      size_t text_start = matches[i].first + tag.size();
      size_t text_end   = (i + 1 < matches.size()) ? matches[i + 1].first : otsl.size();
      if (text_end > text_start) {
        e.text = trim(otsl.substr(text_start, text_end - text_start));
      }
    }
    out.push_back(std::move(e));
  }
  return out;
}

struct Row { std::vector<OtslElement> cells; };

std::vector<Row> split_rows(const std::vector<OtslElement> &elems) {
  std::vector<Row> rows;
  Row cur;
  for (const auto &e : elems) {
    if (e.tok == OtslTok::Nl) {
      if (!cur.cells.empty()) rows.push_back(std::move(cur));
      cur = {};
    } else {
      cur.cells.push_back(e);
    }
  }
  if (!cur.cells.empty()) rows.push_back(std::move(cur));
  return rows;
}

void pad_rows(std::vector<Row> &rows) {
  if (rows.empty()) return;
  size_t max_w = 0;
  for (const auto &r : rows) max_w = std::max(max_w, r.cells.size());
  for (auto &r : rows) {
    while (r.cells.size() < max_w) r.cells.push_back(OtslElement{OtslTok::Ecel, ""});
  }
}

bool is_l(OtslTok t) { return t == OtslTok::Lcel || t == OtslTok::Xcel; }
bool is_u(OtslTok t) { return t == OtslTok::Ucel || t == OtslTok::Xcel; }

} // namespace

std::string otsl_to_html(const std::string &otsl_in) {
  std::string otsl = trim(otsl_in);
  if (otsl.empty()) return "";
  if (otsl.find(kNl) == std::string::npos) otsl += std::string(kNl);
  auto elems = parse_otsl(otsl);
  if (elems.empty()) return "";
  auto rows = split_rows(elems);
  if (rows.empty()) return "";
  pad_rows(rows);
  const size_t nrows = rows.size();
  const size_t ncols = rows.front().cells.size();
  if (ncols == 0) return "";
  struct CellInfo {
    bool is_root = false; bool empty = false;
    int row_span = 1;     int col_span = 1;
    std::string text;
  };
  std::vector<std::vector<CellInfo>> grid(nrows, std::vector<CellInfo>(ncols));
  for (size_t r = 0; r < nrows; ++r) {
    for (size_t c = 0; c < ncols; ++c) {
      const auto &e = rows[r].cells[c];
      if (e.tok == OtslTok::Fcel || e.tok == OtslTok::Ecel) {
        CellInfo info;
        info.is_root = true;
        info.empty   = (e.tok == OtslTok::Ecel);
        info.text    = e.text;
        size_t cc = c + 1;
        while (cc < ncols && is_l(rows[r].cells[cc].tok)) { ++info.col_span; ++cc; }
        size_t rr = r + 1;
        while (rr < nrows && is_u(rows[rr].cells[c].tok)) { ++info.row_span; ++rr; }
        grid[r][c] = info;
      }
    }
  }
  std::string out = "<table>";
  for (size_t r = 0; r < nrows; ++r) {
    out += "<tr>";
    for (size_t c = 0; c < ncols; ++c) {
      const auto &g = grid[r][c];
      if (!g.is_root) continue;
      std::string tag = "<td";
      if (g.row_span > 1) tag += " rowspan=\"" + std::to_string(g.row_span) + "\"";
      if (g.col_span > 1) tag += " colspan=\"" + std::to_string(g.col_span) + "\"";
      out += tag + ">" + html_escape(g.text) + "</td>";
    }
    out += "</tr>";
  }
  out += "</table>";
  return out;
}

// ---------------------------------------------------------------------------
// VLMTable
// ---------------------------------------------------------------------------

VLMTable::VLMTable() = default;
VLMTable::~VLMTable() noexcept = default;

bool VLMTable::init() {
  const char *url_env = std::getenv("VLLM_TABLE_BASE_URL");
  base_url_ = (url_env && url_env[0])
                  ? std::string(url_env)
                  : std::string(get_env("VLLM_BASE_URL", "http://localhost:8003"));
  while (!base_url_.empty() && base_url_.back() == '/') base_url_.pop_back();
  const char *model_env = std::getenv("VLLM_TABLE_MODEL");
  model_ = (model_env && model_env[0])
               ? std::string(model_env)
               : std::string(get_env("VLLM_MODEL", "PaddleOCR-VL-1.5-0.9B"));
  prompt_     = get_env("VLLM_TABLE_PROMPT", "Table Recognition:");
  batch_      = std::max(1, get_env_int("VLLM_TABLE_BATCH", 8));
  timeout_s_  = std::max(1, get_env_int("VLLM_TABLE_TIMEOUT_S", 60));
  max_tokens_ = std::max(64, get_env_int("VLLM_TABLE_MAX_TOKENS", 4096));
  HttpResp r = http_get(base_url_ + "/v1/models", 5);
  if (!r.ok) {
    std::cerr << "[VLMTable] /v1/models unreachable at " << base_url_
              << " (status=" << r.status << ") — tables disabled\n";
    return false;
  }
  try {
    auto j = nlohmann::json::parse(r.body);
    if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
      bool model_present = false;
      for (const auto &m : j["data"]) {
        if (m.value("id", "") == model_) { model_present = true; break; }
      }
      if (!model_present) {
        std::string first = j["data"][0].value("id", "");
        if (!first.empty()) {
          std::cout << "[VLMTable] requested model='" << model_
                    << "' not found on server; using '" << first << "'\n";
          model_ = first;
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[VLMTable] /v1/models parse warning: " << e.what() << '\n';
  }
  ready_ = true;
  std::cout << "[VLMTable] ready: " << base_url_ << " model=" << model_
            << " batch=" << batch_ << " timeout=" << timeout_s_ << "s"
            << " max_tokens=" << max_tokens_ << '\n';
  if (kProfEnabled) {
    std::cout << "[VLMTable] VLM_PROFILE=1 — writing to " << kProfPath << '\n';
  }
  return true;
}

bool VLMTable::single_request(const std::vector<uint8_t> &crop_png,
                               std::string &out_otsl) {
  std::string b64 = to_base64(crop_png);
  nlohmann::json body = {
      {"model", model_},
      {"max_tokens", max_tokens_},
      {"temperature", 0.0},
      {"top_p", 1.0},
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
    std::cerr << "[VLMTable] chat status=" << r.status
              << " body=" << r.body.substr(0, std::min<size_t>(r.body.size(), 200))
              << '\n';
    return false;
  }
  try {
    auto j = nlohmann::json::parse(r.body);
    out_otsl = j.at("choices").at(0).at("message").at("content").get<std::string>();
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[VLMTable] response parse failed: " << e.what() << '\n';
    return false;
  }
}

std::vector<std::string>
VLMTable::run(const GpuImage &page, const std::vector<Box> &regions,
              cudaStream_t stream) {
  std::vector<std::string> out;
  if (regions.empty()) return out;
  if (page.empty()) return out;
  if (!ready_) return out;

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

  const auto t_mu_try = Clock::now();
  std::lock_guard<std::mutex> lk(call_mu_);
  if (kProfEnabled) t_mutex_wait_us = us_since(t_mu_try);

  const auto t_d2h_start = Clock::now();
  const size_t need = static_cast<size_t>(page.rows) * page.step;
  if (host_page_.size() < need) host_page_.resize(need);
  if (cudaSuccess != cudaMemcpyAsync(host_page_.data(), page.data, need,
                                      cudaMemcpyDeviceToHost, stream)) {
    std::cerr << "[VLMTable] page D2H failed\n";
    return out;
  }
  cudaStreamSynchronize(stream);
  if (kProfEnabled) t_d2h_us = us_since(t_d2h_start);

  const auto t_png_start = Clock::now();
  std::vector<std::vector<uint8_t>> crops_png;
  crops_png.reserve(regions.size());
  if (kProfEnabled) t_png_per_crop_us.reserve(regions.size());
  for (const auto &b : regions) {
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

  out.resize(regions.size());
  const int max_inflight = std::min<int>(batch_, static_cast<int>(crops_png.size()));

  struct CropProf {
    int64_t b64_us = 0, json_us = 0, http_us = 0, parse_us = 0;
    size_t sent = 0, recv = 0;
  };
  std::vector<CropProf> crop_prof(crops_png.size());

  for (size_t off = 0; off < crops_png.size(); off += (size_t)max_inflight) {
    size_t end = std::min(crops_png.size(), off + (size_t)max_inflight);
    std::vector<std::thread> workers;
    workers.reserve(end - off);
    for (size_t i = off; i < end; ++i) {
      workers.emplace_back([this, i, &crops_png, &out, &crop_prof] {
        const bool do_prof = kProfEnabled;

        const auto t_b64 = Clock::now();
        std::string b64 = to_base64(crops_png[i]);
        const int64_t dt_b64 = do_prof ? us_since(t_b64) : 0;

        const auto t_json = Clock::now();
        nlohmann::json body = {
            {"model", model_},
            {"max_tokens", max_tokens_},
            {"temperature", 0.0},
            {"top_p", 1.0},
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

        const auto t_http = Clock::now();
        HttpResp r = http_post_json(base_url_ + "/v1/chat/completions",
                                    json_str, timeout_s_);
        const int64_t dt_http = do_prof ? us_since(t_http) : 0;

        const auto t_parse = Clock::now();
        std::string otsl;
        if (r.ok) {
          try {
            auto j = nlohmann::json::parse(r.body);
            otsl = j.at("choices").at(0).at("message").at("content").get<std::string>();
          } catch (...) {}
        }
        const int64_t dt_parse = do_prof ? us_since(t_parse) : 0;

        if (!otsl.empty()) {
          if (otsl.size() > 6 && otsl[0] == '<' && otsl[1] == 't' && otsl[2] == 'a' &&
              otsl[3] == 'b' && otsl[4] == 'l' && otsl[5] == 'e') {
            out[i] = otsl;
          } else {
            out[i] = otsl_to_html(otsl);
          }
        }

        if (do_prof) {
          crop_prof[i] = {dt_b64, dt_json, dt_http, dt_parse,
                          json_str.size(), r.body.size()};
        }
      });
    }
    for (auto &t : workers) t.join();
  }

  if (kProfEnabled) {
    for (const auto &cp : crop_prof) {
      t_b64_total_us      += cp.b64_us;
      t_json_build_us     += cp.json_us;
      t_http_wall_us      += cp.http_us;
      t_response_parse_us += cp.parse_us;
      bytes_sent_total    += cp.sent;
      bytes_recv_total    += cp.recv;
      t_http_per_crop_us.push_back(cp.http_us);
      t_parse_per_crop_us.push_back(cp.parse_us);
    }
    const int64_t t_total_us = us_since(t_run_start);
    nlohmann::json rec = {
        {"backend",             "table"},
        {"n_crops",             (int)regions.size()},
        {"t_mutex_wait_us",     t_mutex_wait_us},
        {"t_d2h_us",            t_d2h_us},
        {"t_png_total_us",      t_png_total_us},
        {"t_png_per_crop_us",   t_png_per_crop_us},
        {"t_b64_total_us",      t_b64_total_us},
        {"t_json_build_us",     t_json_build_us},
        {"t_http_wall_us",      t_http_wall_us},
        {"t_http_per_crop_us",  t_http_per_crop_us},
        {"t_response_parse_us", t_response_parse_us},
        {"t_parse_per_crop_us", t_parse_per_crop_us},
        {"t_total_us",          t_total_us},
        {"bytes_sent_total",    (int64_t)bytes_sent_total},
        {"bytes_recv_total",    (int64_t)bytes_recv_total},
    };
    prof_append(rec);
    std::cerr << "[VLMTable:PROF] n=" << regions.size()
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

} // namespace turbo_ocr::table
