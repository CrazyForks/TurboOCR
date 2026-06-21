#include "turbo_ocr/routes/pdf_routes.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <format>
#include <future>

#include "turbo_ocr/common/logger.h"

#ifndef USE_CPU_ONLY
#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#endif
#include <mutex>

#include <opencv2/core.hpp>

#include <drogon/HttpAppFramework.h>
#include <drogon/utils/Utilities.h>
#include <json/json.h>

#include "turbo_ocr/classification/doc_orientation_common.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/layout/reading_order.h"
#include "turbo_ocr/pdf/page_image_encoder.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "simdutf.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/server_types.h"

using turbo_ocr::OCRResultItem;
using turbo_ocr::results_to_json;

namespace turbo_ocr::routes {

namespace {

// Render-DPI request bounds, shared by the CPU and GPU /ocr/pdf routes (L5):
// previously the literal 50/600 pair appeared at every dpi range check.
constexpr int kMinPdfDpi = 50;
constexpr int kMaxPdfDpi = 600;
// Default render DPI for the CPU route when ?dpi= is absent. The GPU route
// takes its default from ServerConfig (default_dpi); this is the CPU-only path.
constexpr int kCpuDefaultDpi = 100;

// L3 strict-query-params (opt-in). Default OFF preserves the historical
// lenient behavior (unknown params silently ignored). When
// TURBO_OCR_STRICT_QUERY_PARAMS=1, an unrecognized query parameter is a 400
// INVALID_PARAMETER instead. Read once; cached for the process lifetime.
[[nodiscard]] bool strict_query_params_enabled() noexcept {
  static const bool v = server::env_enabled("TURBO_OCR_STRICT_QUERY_PARAMS");
  return v;
}

// When strict mode is on, reject the request if any query parameter is not in
// `allowed`. Returns true (and invokes `callback`) on rejection. No-op (returns
// false) when strict mode is off, so default behavior is byte-identical.
// NOTE: Drogon's getParameters() merges x-www-form-urlencoded body fields with
// the query string. The PDF route accepts multipart (file fields, which do NOT
// land in getParameters()), JSON, or raw bodies — none populate plain form
// params — so the map here is the query string only. The caller still passes
// the multipart-only check downstream; we only gate query keys.
[[nodiscard]] bool reject_unknown_query_params(
    const drogon::HttpRequestPtr &req,
    std::initializer_list<std::string_view> allowed,
    std::function<void(const drogon::HttpResponsePtr &)> &callback) {
  if (!strict_query_params_enabled()) return false;
  for (const auto &kv : req->getParameters()) {
    const std::string &key = kv.first;
    bool known = false;
    for (auto a : allowed)
      if (a == key) { known = true; break; }
    if (!known) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
          std::format("Unknown query parameter '{}' "
                      "(TURBO_OCR_STRICT_QUERY_PARAMS=1)", key)));
      return true;
    }
  }
  return false;
}

// Parse a query-param int safely (std::atoi is UB on overflow). Returns the
// `fallback` for empty/non-numeric/out-of-int-range input, so the caller's
// own range check then rejects it deterministically instead of acting on a
// wrapped/garbage value.
[[nodiscard]] int query_int(const std::string &s, int fallback) {
  if (s.empty()) return fallback;
  errno = 0;
  char *end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || *end != '\0' || errno == ERANGE ||
      v < INT_MIN || v > INT_MAX)
    return fallback;
  return static_cast<int>(v);
}

// ---------------------------------------------------------------------------
// PDF text-layer helpers — shared by CPU and GPU /ocr/pdf routes.
//
// `geometric` mode reads the PDF's native text layer (extracted by pdfium)
// directly — no rendering, no OCR. `auto` falls back to OCR only when the
// text layer is unusable (image-only PDF, garbage encoding, rotated page).
// `text_layer_quality_for` decides whether the layer is trustworthy:
//   absent   — no chars, no lines, or fewer than 10 chars (looks scanned)
//   rejected — non-zero rotation, too many U+FFFD replacement chars,
//              too many non-printable characters
//   trusted  — usable text we can return without OCRing the page
// `fill_from_text_layer_pt` copies that text + bounding boxes into the
// page result at point coordinates (DPI=72); the caller scales to pixel
// coordinates only when the resolved mode is geometric (so layout output,
// which is in pixel space, doesn't get rescaled twice).
struct PdfPageResultBase {
  std::vector<OCRResultItem> results;
  std::vector<layout::LayoutBox> layout;
  std::vector<int> reading_order;
  int width = 0, height = 0, effective_dpi = 0;
  pdf::PdfMode resolved_mode = pdf::PdfMode::Ocr;
  std::string_view text_layer_quality = "absent";
  // Detected page rotation (clockwise, 0/90/180/270) when autorotate=1; the
  // page image + boxes were de-rotated to upright by this amount.
  int orientation_deg = 0;
  // Optional: encoded page image bytes (set when ?images=inline).
  std::vector<uint8_t> encoded_image;
};

// Image mode for /ocr/pdf?images=... — only "inline" is supported: the page
// image is embedded as base64 in the JSON response. No server-side cache, no
// GET retrieval endpoint.
enum class ImageMode { None, Inline };

ImageMode parse_image_mode(const std::string &s) noexcept {
  if (s == "1" || s == "true" || s == "on" || s == "yes" || s == "inline")
    return ImageMode::Inline;
  return ImageMode::None;
}

// Parse image-capture query params (?images=inline&format=png|jpeg|webp&
// quality=1-100&lossless=0/1&png_compression=0-9&max_side=N). Everything is
// per-request; the only env knob is which JPEG encoder backend runs
// (TURBO_PDF_IMAGE_ENCODER gpu|cpu — same bytes either way). Returns an
// error message on invalid values, empty on success.
std::string parse_image_query_params(const drogon::HttpRequestPtr &req,
                                      ImageMode &image_mode,
                                      pdf::EncodeOptions &encode_opts) {
  image_mode = ImageMode::None;
  encode_opts = {};

  auto images_str = req->getParameter("images");
  if (!images_str.empty())
    image_mode = parse_image_mode(std::string(images_str));

  auto fmt_str = req->getParameter("format");
  if (!fmt_str.empty())
    encode_opts.format = pdf::parse_page_image_format(fmt_str.c_str());

  // lossless: defaults to true (set in EncodeOptions).
  auto lossless_str = req->getParameter("lossless");
  if (!lossless_str.empty()) {
    if (lossless_str == "0" || lossless_str == "false" || lossless_str == "no" || lossless_str == "off")
      encode_opts.lossless = false;
    else if (lossless_str == "1" || lossless_str == "true" || lossless_str == "yes" || lossless_str == "on")
      encode_opts.lossless = true;
    else
      return "lossless must be 0/1/true/false";
  }

  auto png_comp_str = req->getParameter("png_compression");
  if (!png_comp_str.empty()) {
    int c = query_int(std::string(png_comp_str), -1);
    if (c < 0 || c > 9) return "png_compression must be 0-9";
    encode_opts.png_compression = c;
  }

  auto quality_str = req->getParameter("quality");
  if (!quality_str.empty()) {
    int q = query_int(std::string(quality_str), -1);
    if (q < 1 || q > 100)
      return "quality must be 1-100";
    encode_opts.quality = q;
    // An explicit quality means the client wants lossy. Honor it.
    if (lossless_str.empty()) encode_opts.lossless = false;
  }

  auto max_side_str = req->getParameter("max_side");
  if (!max_side_str.empty()) {
    int ms = query_int(std::string(max_side_str), -1);
    if (ms < 0)
      return "max_side must be >= 0";
    encode_opts.max_side = ms;
  }

  return {};
}

template <typename PageResult>
void fill_from_text_layer_pt(PageResult &pg, const pdf::PdfPageText &text) {
  pg.width  = static_cast<int>(std::round(text.page_width_pt));
  pg.height = static_cast<int>(std::round(text.page_height_pt));
  pg.effective_dpi = 72;
  pg.results.reserve(text.lines.size());
  for (const auto &line : text.lines) {
    OCRResultItem item;
    item.source = "pdf";
    item.confidence = 1.0f;
    item.text = line.text;
    int ix0 = static_cast<int>(std::round(line.x0_pt));
    int iy0 = static_cast<int>(std::round(line.y0_pt));
    int ix1 = static_cast<int>(std::round(line.x1_pt));
    int iy1 = static_cast<int>(std::round(line.y1_pt));
    item.box[0] = {ix0, iy0};
    item.box[1] = {ix1, iy0};
    item.box[2] = {ix1, iy1};
    item.box[3] = {ix0, iy1};
    pg.results.push_back(std::move(item));
  }
}

std::string_view text_layer_quality_for(const pdf::PdfPageText &text) {
  if (text.char_count == 0)         return "absent";
  if (text.rotation_deg != 0)       return "rejected";
  if (text.char_count < 10)         return "absent";
  if (text.fffd_count * 20 > text.char_count)     return "rejected";
  if (text.nonprint_count * 10 > text.char_count) return "rejected";
  if (text.lines.empty())           return "absent";
  return "trusted";
}

/// Helper: extract PDF bytes from a Drogon request (raw, base64 JSON, multipart).
/// Returns true on success, fills pdf_ptr/pdf_len and may fill decoded_buf.
/// On failure, calls cb with 400 and returns false.
bool extract_pdf_bytes(const drogon::HttpRequestPtr &req,
                       std::string &decoded_buf,
                       const char *&pdf_ptr, size_t &pdf_len,
                       const std::function<void(const drogon::HttpResponsePtr &)> &cb) {
  auto ct = req->getHeader("Content-Type");
  if (ct.find("multipart/form-data") != std::string::npos) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
      cb(server::error_response(drogon::k400BadRequest, "INVALID_MULTIPART", "Failed to parse multipart body"));
      return false;
    }
    for (auto &file : parser.getFiles()) {
      auto name = file.getItemName();
      if (name == "file" || name == "pdf") {
        decoded_buf.assign(file.fileData(), file.fileLength());
        break;
      }
    }
    if (decoded_buf.empty()) {
      cb(server::error_response(drogon::k400BadRequest, "MISSING_FILE",
          "Multipart request must contain a 'file' or 'pdf' form field"));
      return false;
    }
    pdf_ptr = decoded_buf.data();
    pdf_len = decoded_buf.size();
  } else if (ct.find("application/json") != std::string::npos) {
    auto json = req->getJsonObject();
    if (!json || !json->isMember("pdf")) {
      cb(server::error_response(drogon::k400BadRequest, "MISSING_PDF",
          R"(JSON body must contain {"pdf": "<base64>"})"));
      return false;
    }
    auto b64 = (*json)["pdf"].asString();
    decoded_buf = turbo_ocr::base64_decode(b64);
    if (decoded_buf.empty()) {
      cb(server::error_response(drogon::k400BadRequest, "BASE64_DECODE_FAILED", "Failed to decode base64 PDF"));
      return false;
    }
    pdf_ptr = decoded_buf.data();
    pdf_len = decoded_buf.size();
  } else {
    if (req->body().empty()) {
      cb(server::error_response(drogon::k400BadRequest, "EMPTY_BODY", "Empty body"));
      return false;
    }
    pdf_ptr = req->body().data();
    pdf_len = req->body().size();
  }
  return true;
}

// Inline page-count guard: emits PDF_TOO_LARGE if the document exceeds the
// configured limit (cfg.max_pdf_pages — honors --max-pdf-pages AND
// MAX_PDF_PAGES, matching gRPC). Returns true on guard-trip (caller aborts).
bool reject_if_too_many_pages(const uint8_t *pdf_data, size_t pdf_len_local,
                               int limit, server::DrogonCallback &cb) {
  pdf::PdfDocument check_doc(pdf_data, pdf_len_local);
  if (!check_doc.ok()) return false;
  int np = check_doc.page_count();
  if (np > limit) {
    cb(server::error_response(drogon::k400BadRequest, "PDF_TOO_LARGE",
        std::format("PDF has {} pages, maximum is {} (set MAX_PDF_PAGES to increase)",
                    np, limit)));
    return true;
  }
  return false;
}

// Open the PDF and pre-extract per-page text only when the chosen mode
// actually needs the text layer. mode=ocr skips this. On open failure we
// downgrade to mode=ocr and clear the doc.
void open_pdf_for_text_layer(const uint8_t *pdf_data, size_t pdf_len_local,
                              pdf::PdfMode &mode,
                              std::unique_ptr<pdf::PdfDocument> &pdf_doc,
                              std::vector<pdf::PdfPageText> &page_text_cache) {
  if (mode == pdf::PdfMode::Ocr) return;
  pdf_doc = std::make_unique<pdf::PdfDocument>(pdf_data, pdf_len_local);
  if (!pdf_doc->ok()) {
    TOCR_LOG_WARN("Failed to open PDF for text-layer lookup; falling back to mode=ocr",
                  "route", "/ocr/pdf");
    mode = pdf::PdfMode::Ocr;
    pdf_doc.reset();
    return;
  }
  int np = pdf_doc->page_count();
  page_text_cache.reserve(static_cast<size_t>(std::max(0, np)));
  for (int p = 0; p < np; ++p)
    page_text_cache.push_back(pdf_doc->extract_page(p));
}

// Decide per-page resolved_mode and whether each page needs rendering,
// based on text-layer quality. mode=ocr always renders, so this is only
// called for the non-ocr modes. AutoVerified is GPU-only — on CPU it's
// aliased to Auto before this is called.
template <typename PageResult>
void prepopulate_pages(pdf::PdfMode mode,
                       bool layout_or_want_layout,
                       const std::vector<pdf::PdfPageText> &page_text_cache,
                       std::vector<PageResult> &page_results,
                       std::vector<uint8_t> &need_render,
                       bool *any_need_render,
                       bool want_page_image = false) {
  int np = static_cast<int>(page_text_cache.size());
  page_results.resize(static_cast<size_t>(np));
  need_render.assign(static_cast<size_t>(np), 0);

  for (int p = 0; p < np; ++p) {
    const auto &text = page_text_cache[static_cast<size_t>(p)];
    auto &pg = page_results[static_cast<size_t>(p)];
    pg.text_layer_quality = text_layer_quality_for(text);
    bool has_good_layer = (pg.text_layer_quality == "trusted");

    switch (mode) {
      case pdf::PdfMode::Geometric:
        pg.resolved_mode = pdf::PdfMode::Geometric;
        if (has_good_layer) {
          fill_from_text_layer_pt(pg, text);
        } else {
          pg.width = static_cast<int>(std::round(text.page_width_pt));
          pg.height = static_cast<int>(std::round(text.page_height_pt));
          pg.effective_dpi = 72;
        }
        if (layout_or_want_layout) {
          need_render[static_cast<size_t>(p)] = 1;
          if (any_need_render) *any_need_render = true;
        }
        break;
      case pdf::PdfMode::Auto:
        if (has_good_layer) {
          pg.resolved_mode = pdf::PdfMode::Geometric;
          fill_from_text_layer_pt(pg, text);
          if (layout_or_want_layout) {
            need_render[static_cast<size_t>(p)] = 1;
            if (any_need_render) *any_need_render = true;
          }
        } else {
          pg.resolved_mode = pdf::PdfMode::Ocr;
          need_render[static_cast<size_t>(p)] = 1;
          if (any_need_render) *any_need_render = true;
        }
        break;
      case pdf::PdfMode::AutoVerified:
        pg.resolved_mode = pdf::PdfMode::AutoVerified;
        need_render[static_cast<size_t>(p)] = 1;
        if (any_need_render) *any_need_render = true;
        break;
      default: break;
    }

    // A page image was requested. Text-layer pages (geometric / auto with a
    // trusted layer) skip rasterization for OCR, so force a render here —
    // otherwise the encoder has no pixels and the page comes back with no
    // image. The text still comes from the resolved mode; only the rendered
    // image is added.
    if (want_page_image && !need_render[static_cast<size_t>(p)]) {
      need_render[static_cast<size_t>(p)] = 1;
      if (any_need_render) *any_need_render = true;
    }
  }
}

// Build the final {pages: [...]} JSON. Shared by CPU and GPU paths. The
// per-result + per-page byte estimate keeps dense pages from reallocating
// and tiny pages from over-allocating.
template <typename PageResult>
std::string emit_pdf_response(std::vector<PageResult> &page_results,
                               int request_dpi,
                               bool want_blocks = false,
                               ImageMode image_mode = ImageMode::None,
                               const pdf::EncodeOptions &encode_opts = {},
                               bool want_orientation = false) {
  size_t n_pages = page_results.size();
  size_t total_results = 0;
  size_t total_image_bytes = 0;
  for (size_t i = 0; i < n_pages; ++i) {
    total_results += page_results[i].results.size() + page_results[i].layout.size();
    total_image_bytes += page_results[i].encoded_image.size();
  }
  std::string json_str;
  json_str.reserve(total_results * 256 + n_pages * 256 + 64 +
                   (total_image_bytes * 4) / 3 + n_pages * 48);
  json_str += "{\"pages\":[";
  for (size_t i = 0; i < n_pages; ++i) {
    if (i > 0) json_str += ',';
    auto &pg = page_results[i];
    int page_dpi = pg.effective_dpi > 0 ? pg.effective_dpi : request_dpi;
    json_str += "{\"page\":";
    json_str += std::to_string(i + 1);
    json_str += ",\"page_index\":";
    json_str += std::to_string(i);
    json_str += ",\"dpi\":";
    json_str += std::to_string(page_dpi);
    json_str += ",\"width\":";
    json_str += std::to_string(pg.width);
    json_str += ",\"height\":";
    json_str += std::to_string(pg.height);
    json_str += ',';
    auto page_json = !pg.reading_order.empty()
                         ? emit_results_json(pg.results, pg.layout,
                                              pg.reading_order, want_blocks)
                         : results_to_json(pg.results, pg.layout);
    json_str.append(page_json.data() + 1, page_json.size() - 2);
    json_str += ",\"mode\":\"";
    json_str += pdf::mode_name(pg.resolved_mode);
    json_str += "\",\"text_layer_quality\":\"";
    json_str += pg.text_layer_quality;
    json_str += '"';

    // Detected page rotation (the image + boxes were de-rotated upright by it).
    if (want_orientation) {
      json_str += ",\"orientation_deg\":";
      json_str += std::to_string(pg.orientation_deg);
    }

    // Inline page image: base64 of the encoded bytes (simdutf, SIMD path).
    if (image_mode == ImageMode::Inline && !pg.encoded_image.empty()) {
      const auto &raw = pg.encoded_image;
      size_t b64_len = ((raw.size() + 2) / 3) * 4;
      std::string b64(b64_len, '\0');
      simdutf::binary_to_base64(
          reinterpret_cast<const char *>(raw.data()), raw.size(),
          b64.data());
      json_str += ",\"image_b64\":\"";
      json_str += b64;
      json_str += "\",\"image_content_type\":\"";
      json_str += pdf::page_image_content_type(encode_opts.format);
      json_str += '"';
    }

    json_str += '}';
  }
  json_str += "]}";
  return json_str;
}

#ifndef USE_CPU_ONLY
// C4 NOTE (GPU /ocr/pdf): unlike the single-image GPU routes, this route does
// NOT use the synchronous dispatcher.submit(...).get() pattern that C4 converts
// to submit_for_default. The streamed-render path fans page tasks out into
// std::futures (run_streamed_render_gpu) and joins them with f.get() at the
// end. Those tasks write results into the handler-scoped PdfPageSink and read
// PPMs owned by the handler-scoped StreamHandle, so abandoning a future on
// timeout (the basis of submit_for_default) would tear that shared state down
// while a still-running task references it — a use-after-free, i.e. exactly the
// non-breaking guarantee we must keep. Per-page abandonment-based timeout would
// require restructuring every page task to own all of its state (a larger
// refactor than this change authorizes). The dispatcher-level deadline
// (set_request_timeout_ms, applied in main.cpp) still governs every other GPU
// path; whole-request bounding for PDF stays the WorkPool's responsibility.
// TODO(C4-pdf): make page tasks self-contained, then bound the join per page.

// GPU page-result type carries the same fields as the base. The
// per-page-future render loop resolves each rendered page on the dispatcher
// thread pool and writes back into this shared vector under results_mutex.
struct GpuPdfPageResult : public PdfPageResultBase {};

// Shared state every streamed-render page task writes into. Constructed in
// the route handler, whose scope outlives every page future (the handler
// joins them before returning) — so tasks may hold a reference to it.
// Task lambdas must capture this sink plus their per-task values
// EXPLICITLY: an implicit [&] would bind value parameters of intermediate
// helper frames that die before the tasks run.
struct PdfPageSink {
  std::mutex &results_mutex;
  std::vector<GpuPdfPageResult> &page_results;
  pdf::PdfDocument *pdf_doc;  // null when no text layer was opened
  const std::vector<pdf::PdfPageText> &page_text_cache;
  int dpi;
  // Page-image export (?images=inline): encode each rendered page on the
  // worker right after decode and stash the bytes in the slot.
  ImageMode image_mode = ImageMode::None;
  pdf::EncodeOptions encode_opts{};
  // autorotate=1: detect each page's orientation and de-rotate the decoded
  // image upright BEFORE OCR + encode, so image, boxes and text are all upright.
  bool autorotate = false;
  // Rendered pages whose PPM could not be read back (tmpfs pressure,
  // truncated write). We rendered these ourselves, so a failure here is a
  // server-side error: any non-zero count turns the response into a 500
  // instead of emitting silently blank pages in a 200.
  std::atomic<int> decode_failures{0};
};

// Encode the rendered page when ?images=inline. Runs on the worker thread,
// piggy-backing on decode: JPEG goes through nvJPEG on the GPU by default
// (TURBO_PDF_IMAGE_ENCODER=cpu opts out), PNG/WebP through the CPU encoders.
[[nodiscard]] std::vector<uint8_t>
maybe_encode_page(const PdfPageSink &sink, const cv::Mat &img) {
  if (sink.image_mode != ImageMode::Inline) return {};
  return pdf::encode_page_image(img, sink.encode_opts);
}

// Effective mode of a rendered page. Final by render time: prepopulate_pages
// ran before the render started, and mode==Ocr never resolves per page.
[[nodiscard]] pdf::PdfMode page_mode_of(PdfPageSink &sink, int page_idx) {
  std::lock_guard<std::mutex> lock(sink.results_mutex);
  return page_idx < static_cast<int>(sink.page_results.size())
      ? sink.page_results[page_idx].resolved_mode
      : pdf::PdfMode::Ocr;
}

// Store one OCR'd page: tag sources, apply the auto_verified text-layer
// replacement, write the slot under the sink lock.
void store_ocr_page(PdfPageSink &sink, int page_idx,
                    pipeline::OcrPipelineResult out, int width, int height,
                    std::vector<uint8_t> encoded_image = {},
                    int orientation_deg = 0) {
  for (auto &it : out.results) it.source = "ocr";

  const pdf::PdfMode page_mode = page_mode_of(sink, page_idx);
  if (page_mode == pdf::PdfMode::AutoVerified &&
      page_idx < static_cast<int>(sink.page_text_cache.size()) && sink.pdf_doc)
    pdf::verify_results_with_text_layer(out.results, *sink.pdf_doc,
                                        page_idx, sink.dpi);

  std::lock_guard<std::mutex> lock(sink.results_mutex);
  auto &slot = sink.page_results[page_idx];
  slot.results       = std::move(out.results);
  slot.layout        = std::move(out.layout);
  slot.reading_order = std::move(out.reading_order);
  slot.width         = width;
  slot.height        = height;
  slot.effective_dpi = sink.dpi;
  slot.encoded_image = std::move(encoded_image);
  slot.orientation_deg = orientation_deg;
  if (page_mode == pdf::PdfMode::Ocr)
    slot.resolved_mode = pdf::PdfMode::Ocr;
}

// Geometric page: text already came from the PDF layer in pt-space; store
// the layout detections and rescale the stored text boxes to pixel space
// matching the rendered image. When reading order was requested, compute it
// over the (now pixel-space) text + layout so geometric pages carry the same
// reading_order/blocks keys as OCR pages in the same response.
void store_geometric_page(PdfPageSink &sink, int page_idx,
                          std::vector<layout::LayoutBox> layout,
                          int width, int height, bool want_reading_order,
                          std::vector<uint8_t> encoded_image = {},
                          int orientation_deg = 0) {
  std::lock_guard<std::mutex> lock(sink.results_mutex);
  auto &slot = sink.page_results[page_idx];
  const float pt_to_px = static_cast<float>(sink.dpi) / 72.0f;
  for (auto &item : slot.results)
    for (int k = 0; k < 4; ++k) {
      item.box[k][0] = static_cast<int>(std::round(item.box[k][0] * pt_to_px));
      item.box[k][1] = static_cast<int>(std::round(item.box[k][1] * pt_to_px));
    }
  slot.layout        = std::move(layout);
  slot.width         = width;
  slot.height        = height;
  slot.effective_dpi = sink.dpi;
  slot.encoded_image = std::move(encoded_image);
  slot.orientation_deg = orientation_deg;
  if (want_reading_order && !slot.layout.empty()) {
    turbo_ocr::assign_layout_ids(slot.results, slot.layout);
    slot.reading_order =
        turbo_ocr::layout::assign_reading_order_for_results(slot.results,
                                                            slot.layout);
  }
}

// Single-page task: decode the PPM and run layout-only (Geometric) or the
// full pipeline (everything else). Runs on a dispatcher worker.
void ocr_single_page(pipeline::GpuPipelineEntry &e, PdfPageSink &sink,
                     bool layout_enabled, bool want_reading_order,
                     int page_idx, const std::string &path) {
  cv::Mat img = render::PdfRenderer::decode_ppm(path);
  if (img.empty()) {
    TOCR_LOG_ERROR("Failed to decode PPM for page",
                   "route", "/ocr/pdf", "page", page_idx);
    sink.decode_failures.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  if (page_mode_of(sink, page_idx) == pdf::PdfMode::Geometric) {
    // Geometric pages come from a born-digital text layer (already upright);
    // autorotate does not apply — its text boxes are in PDF pt-space, not the
    // pixel frame, so rotating the image would desync them. Just encode + run
    // layout-only when the client asked for it.
    auto encoded = maybe_encode_page(sink, img);
    auto layout = layout_enabled
        ? e.pipeline->run_layout_only(img, e.stream).layout
        : std::vector<layout::LayoutBox>{};
    store_geometric_page(sink, page_idx, std::move(layout),
                         img.cols, img.rows, want_reading_order,
                         std::move(encoded));
  } else {
    // OCR page: de-rotate upright FIRST (when autorotate=1) so det/rec, the
    // returned boxes, and the encoded image are all in the same upright frame.
    int orient = sink.autorotate
        ? e.pipeline->detect_orientation(img, e.stream) : 0;
    if (orient) classification::rotate_upright(img, orient);
    auto encoded = maybe_encode_page(sink, img);
    auto out = e.pipeline->run_with_layout(img, e.stream, layout_enabled,
                                           want_reading_order);
    store_ocr_page(sink, page_idx, std::move(out), img.cols, img.rows,
                   std::move(encoded), orient);
  }
}

// pdf_only chunk task: decode a chunk of rendered pages and run them
// through the pipeline's batched path — one static-shape det execute +
// cross-image batched rec + one batched layout execute per chunk, which is
// the throughput point of the mode. Runs on a dispatcher worker.
void ocr_page_chunk(pipeline::GpuPipelineEntry &e, PdfPageSink &sink,
                    bool layout_enabled, bool want_reading_order,
                    const std::vector<int> &idxs,
                    const std::vector<std::string> &paths) {
  std::vector<cv::Mat> imgs;
  std::vector<int> live;     // page index per successfully decoded image
  std::vector<int> orients;  // detected rotation per live image (0 unless autorotate)
  imgs.reserve(paths.size());
  live.reserve(paths.size());
  orients.reserve(paths.size());
  for (size_t j = 0; j < paths.size(); ++j) {
    cv::Mat img = render::PdfRenderer::decode_ppm(paths[j]);
    if (img.empty()) {
      TOCR_LOG_ERROR("Failed to decode PPM for page",
                     "route", "/ocr/pdf", "page", idxs[j]);
      sink.decode_failures.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    // De-rotate upright before batching so det/rec + the encoded image are
    // all in the upright frame (per-page detect; batched det follows).
    int orient = sink.autorotate
        ? e.pipeline->detect_orientation(img, e.stream) : 0;
    if (orient) classification::rotate_upright(img, orient);
    imgs.push_back(std::move(img));
    live.push_back(idxs[j]);
    orients.push_back(orient);
  }
  if (imgs.empty()) return;

  auto outs = e.pipeline->run_batch_with_layout(imgs, e.stream,
                                                layout_enabled,
                                                want_reading_order);
  for (size_t j = 0; j < outs.size(); ++j)
    store_ocr_page(sink, live[j], std::move(outs[j]),
                   imgs[j].cols, imgs[j].rows,
                   maybe_encode_page(sink, imgs[j]), orients[j]);
}

// GPU streamed render: per rendered page, either accumulate into a pdf_only
// batch chunk or submit a single-page task; the tasks above do the work.
//
// Returns the StreamHandle by value: the caller MUST keep it alive until
// every future in `page_futures` has completed. The handle owns the
// scratch tmpdir, and ~StreamHandle calls remove_all() on it — destroying
// the handle before the dispatcher workers run their decode_ppm() lambdas
// causes them to open files that no longer exist (the bug fixed here:
// pages would silently come back empty when the GPU pool was busier than
// the renderer, because StreamHandle's lifetime was confined to this
// helper while the futures lived in the caller).
[[nodiscard]] render::PdfRenderer::StreamHandle run_streamed_render_gpu(
    pipeline::PipelineDispatcher &dispatcher,
    render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len_local,
    bool layout_enabled, bool want_reading_order,
    pdf::PdfMode mode,
    PdfPageSink &sink,
    std::vector<uint8_t> &need_render,
    std::vector<std::future<void>> &page_futures,
    std::mutex &futures_mutex,
    int pdf_only_batch,
    std::vector<int> &dropped_pages) {
  // pdf_only batching: OCR pages accumulate here (render callbacks are
  // synchronous inside render_streamed, so no lock needed) and ship to the
  // dispatcher in chunks of pdf_only_batch — matching the static det
  // engine's batch dim. Geometric pages never enter the accumulator.
  std::vector<int> acc_idxs;
  std::vector<std::string> acc_paths;

  auto submit_ocr_chunk = [&](std::vector<int> idxs,
                              std::vector<std::string> paths) {
    std::future<void> fut;
    try {
      // idxs is copied (≤ 8 ints) rather than moved: submit() constructs
      // the task — consuming a move — BEFORE it can throw PoolExhausted,
      // and the catch below still needs the indices.
      fut = dispatcher.submit(
          [&sink, layout_enabled, want_reading_order,
           idxs, paths = std::move(paths)](auto &e) {
            ocr_page_chunk(e, sink, layout_enabled, want_reading_order,
                           idxs, paths);
          });
    } catch (const turbo_ocr::PoolExhaustedError &) {
      // Record the loss instead of emitting silently blank pages — the
      // caller turns any drop into a 503, same contract as every other
      // route's PoolExhaustedError handling.
      TOCR_LOG_WARN("GPU queue full, dropping page chunk", "route", "/ocr/pdf",
                    "first_page", idxs.front(), "pages", idxs.size());
      dropped_pages.insert(dropped_pages.end(), idxs.begin(), idxs.end());
      return;
    }
    std::lock_guard lock(futures_mutex);
    page_futures.push_back(std::move(fut));
  };

  auto handle = pdf_renderer.render_streamed(pdf_data, pdf_len_local, sink.dpi,
      [&](int page_idx, std::string ppm_path) {
        bool is_geometric = false;
        {
          std::lock_guard<std::mutex> rlock(sink.results_mutex);
          auto &page_results = sink.page_results;
          if (page_idx >= static_cast<int>(page_results.size())) {
            page_results.resize(page_idx + 1);
            if (mode != pdf::PdfMode::Ocr &&
                page_idx >= static_cast<int>(need_render.size()))
              need_render.resize(page_idx + 1, 1);
          }
          if (mode != pdf::PdfMode::Ocr &&
              page_idx < static_cast<int>(need_render.size()) &&
              !need_render[page_idx])
            return;
          // Per-page modes are final by render time (prepopulate_pages ran
          // before the render started; mode==Ocr never sets Geometric).
          is_geometric = mode != pdf::PdfMode::Ocr &&
              page_results[page_idx].resolved_mode == pdf::PdfMode::Geometric;
        }

        if (pdf_only_batch > 0 && !is_geometric) {
          acc_idxs.push_back(page_idx);
          acc_paths.push_back(std::move(ppm_path));
          if (static_cast<int>(acc_idxs.size()) >= pdf_only_batch) {
            submit_ocr_chunk(std::move(acc_idxs), std::move(acc_paths));
            acc_idxs.clear();
            acc_paths.clear();
          }
          return;
        }

        std::future<void> fut;
        try {
          fut = dispatcher.submit(
              [&sink, layout_enabled, want_reading_order, page_idx,
               path = std::move(ppm_path)](auto &e) {
                ocr_single_page(e, sink, layout_enabled, want_reading_order,
                                page_idx, path);
              });
        } catch (const turbo_ocr::PoolExhaustedError &) {
          TOCR_LOG_WARN("GPU queue full, dropping page", "route", "/ocr/pdf", "page", page_idx);
          dropped_pages.push_back(page_idx);
          return;
        }
        std::lock_guard lock(futures_mutex);
        page_futures.push_back(std::move(fut));
      });

  // render_streamed is synchronous (all callbacks have fired) — flush the
  // partial last chunk so trailing pages aren't dropped.
  if (!acc_idxs.empty())
    submit_ocr_chunk(std::move(acc_idxs), std::move(acc_paths));

  return handle;
}
#endif // !USE_CPU_ONLY

// CPU streamed render callback. Sequential: decode PPM, run the InferFunc
// inline. mode==Ocr means we never visited prepopulate_pages, so resolved
// mode is pinned here. Geometric pages keep their layer-derived text and
// just rescale point→pixel coords.
// CPU streamed render. Returns num_pages; reports the count of pages that
// failed to read back their rendered PPM via `decode_failures` (a server-side
// fault → the caller surfaces 500, parity with the GPU route). The callback
// body is wrapped in try/catch: it runs INSIDE render_streamed's poll loop on
// the thread that owns the (still-joinable) render thread, so an exception
// escaping here would unwind past that std::thread and std::terminate the
// whole process. Catch everything, tag the page as failed-empty, continue.
int run_streamed_render_cpu(
    const server::InferFunc &infer,
    render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len_local,
    int dpi, bool want_layout, bool want_reading_order, pdf::PdfMode mode,
    std::vector<PdfPageResultBase> &page_results,
    const std::vector<uint8_t> &need_render,
    int &decode_failures,
    ImageMode image_mode = ImageMode::None,
    const pdf::EncodeOptions &encode_opts = {},
    bool autorotate = false,
    const server::OrientFunc &orient_fn = {}) {
  auto stream_handle = pdf_renderer.render_streamed(pdf_data, pdf_len_local, dpi,
      [&](int page_idx, std::string ppm_path) noexcept {
       try {
        if (mode != pdf::PdfMode::Ocr &&
            page_idx < static_cast<int>(need_render.size()) &&
            !need_render[static_cast<size_t>(page_idx)])
          return;

        cv::Mat img = render::PdfRenderer::decode_ppm(ppm_path);
        if (img.empty()) {
          TOCR_LOG_ERROR("Failed to decode PPM for page",
                         "route", "/ocr/pdf", "page", page_idx);
          ++decode_failures;
          return;
        }

        if (page_idx >= static_cast<int>(page_results.size()))
          page_results.resize(page_idx + 1);
        auto &pg = page_results[static_cast<size_t>(page_idx)];

        turbo_ocr::server::InferOptions inf_opts;
        inf_opts.want_layout = want_layout;
        inf_opts.want_reading_order = want_reading_order;
        // mode==Ocr means the per-page resolved_mode wasn't set up
        // front (we skipped the text-layer pre-pass entirely), so
        // pin it to Ocr here. For non-ocr modes pg.resolved_mode is
        // already set per page.
        if (mode == pdf::PdfMode::Ocr)
          pg.resolved_mode = pdf::PdfMode::Ocr;

        // OCR pages: de-rotate upright (autorotate=1) BEFORE encode + infer so
        // image, boxes and text share one frame. Geometric pages are
        // born-digital/upright with pt-space text — skip them.
        if (pg.resolved_mode != pdf::PdfMode::Geometric && autorotate && orient_fn) {
          int orient = orient_fn(img);
          if (orient) classification::rotate_upright(img, orient);
          pg.orientation_deg = orient;
        }

        // Page-image export (?images=inline): encode after any rotation —
        // JPEG via libjpeg-turbo on this build (no GPU), PNG/WebP via OpenCV.
        if (image_mode == ImageMode::Inline)
          pg.encoded_image = pdf::encode_page_image(img, encode_opts);

        if (pg.resolved_mode == pdf::PdfMode::Geometric) {
          // Text already filled from layer in pt-space; only run
          // layout (when requested) and rescale text boxes from
          // points to pixel space matching the rendered image.
          if (want_layout) {
            auto inf = infer(img, inf_opts);
            pg.layout = std::move(inf.layout);
          }
          pg.width = img.cols;
          pg.height = img.rows;
          pg.effective_dpi = dpi;
          const float pt_to_px = static_cast<float>(dpi) / 72.0f;
          for (auto &item : pg.results) {
            for (int k = 0; k < 4; ++k) {
              item.box[k][0] = static_cast<int>(
                  std::round(item.box[k][0] * pt_to_px));
              item.box[k][1] = static_cast<int>(
                  std::round(item.box[k][1] * pt_to_px));
            }
          }
          // Same reading-order parity as OCR pages (see GPU
          // store_geometric_page): geometric pages must carry the
          // reading_order/blocks keys when the client requested them.
          if (want_reading_order && !pg.layout.empty()) {
            turbo_ocr::assign_layout_ids(pg.results, pg.layout);
            pg.reading_order =
                turbo_ocr::layout::assign_reading_order_for_results(
                    pg.results, pg.layout);
          }
        } else {
          // Ocr branch: full pipeline, results from rec.
          auto inf = infer(img, inf_opts);
          pg.results = std::move(inf.results);
          pg.layout = std::move(inf.layout);
          pg.reading_order = std::move(inf.reading_order);
          pg.width = img.cols;
          pg.height = img.rows;
          pg.effective_dpi = dpi;
          for (auto &item : pg.results) item.source = "ocr";
        }
       } catch (const std::exception &e) {
        // Per-page inference failure (ORT/OpenCV throw). Log and leave the
        // page empty — matches the GPU route's per-page f.get() catch.
        // NEVER let this escape render_streamed (joinable render thread).
        TOCR_LOG_ERROR("PDF page inference error", "route", "/ocr/pdf",
                       "page", page_idx, "error", std::string_view(e.what()));
       } catch (...) {
        TOCR_LOG_ERROR("PDF page inference error (unknown)",
                       "route", "/ocr/pdf", "page", page_idx);
       }
      });
  return stream_handle.num_pages;
}

} // namespace

#ifndef USE_CPU_ONLY
void register_pdf_route(server::WorkPool &pool,
                        pipeline::PipelineDispatcher &dispatcher,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available,
                        int pdf_only_batch,
                        int default_dpi,
                        int max_pdf_pages,
                        bool doc_ori_available) {

  drogon::app().registerHandler(
      "/ocr/pdf",
      [&pool, &dispatcher, &pdf_renderer, default_pdf_mode, layout_available,
       pdf_only_batch, default_dpi, max_pdf_pages, doc_ori_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    // Extract PDF bytes (lightweight, on event loop)
    auto pdf_buf = std::make_shared<std::string>();
    const char *pdf_ptr = nullptr;
    size_t pdf_len = 0;

    if (!extract_pdf_bytes(req, *pdf_buf, pdf_ptr, pdf_len, callback))
      return;

    server::InferOptions opts;
    if (auto r = server::parse_query_options(req, layout_available, &opts);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }
    const bool layout_enabled = opts.want_layout;
    const bool want_reading_order = opts.want_reading_order;
    const bool want_blocks = opts.want_blocks;

    if (reject_unknown_query_params(
            req, {"layout", "reading_order", "as_blocks", "dpi", "mode",
                  "images", "format", "lossless", "png_compression", "quality",
                  "max_side", "autorotate"}, callback))
      return;

    auto dpi_str = req->getParameter("dpi");
    // Absent -> default; present-but-garbage/overflow -> -1 -> rejected below
    // (don't silently fall back to default on a bad explicit value).
    int dpi = dpi_str.empty() ? default_dpi : query_int(std::string(dpi_str), -1);
    if (dpi < kMinPdfDpi || dpi > kMaxPdfDpi) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_DPI",
          std::format("DPI must be between {} and {}", kMinPdfDpi, kMaxPdfDpi)));
      return;
    }

    pdf::PdfMode req_mode = default_pdf_mode;
    auto mode_str = req->getParameter("mode");
    if (!mode_str.empty())
      req_mode = pdf::parse_pdf_mode(mode_str.c_str(), default_pdf_mode);

    // Page-image export params (?images=inline&format=...&quality=...)
    ImageMode image_mode;
    pdf::EncodeOptions encode_opts;
    if (auto err = parse_image_query_params(req, image_mode, encode_opts);
        !err.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       "INVALID_PARAMETER", err));
      return;
    }

    // autorotate=1: de-rotate each OCR'd page upright using the doc-orientation
    // model. Rejected when the model isn't loaded (parity with LAYOUT_DISABLED).
    bool autorotate = false;
    if (auto err = server::parse_bool_query(req, "autorotate", &autorotate);
        !err.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       "INVALID_PARAMETER", err));
      return;
    }
    if (autorotate && !doc_ori_available) {
      callback(server::error_response(drogon::k400BadRequest, "AUTOROTATE_DISABLED",
          "autorotate=1 requires the doc-orientation model (models/doc_ori.onnx); "
          "it was not found at startup"));
      return;
    }

    // For raw body case, pdf_ptr points into req->body() — copy into pdf_buf
    if (pdf_buf->empty())
      pdf_buf->assign(pdf_ptr, pdf_len);

    server::submit_work(pool, std::move(callback),
        [pdf_buf, req, &dispatcher, &pdf_renderer,
         layout_enabled, want_reading_order, want_blocks,
         dpi, req_mode, pdf_only_batch, image_mode,
         encode_opts, max_pdf_pages, autorotate](server::DrogonCallback &cb) {
     // Wrap the whole body: post-render work (emit_pdf_response's multi-GB
     // reserve under images=inline, page_results.resize) can throw bad_alloc,
     // which the WorkPool worker would otherwise swallow — leaving the client
     // hung with no response. run_with_error_handling turns it into 500.
     server::run_with_error_handling(cb, "/ocr/pdf", [&] {
      const auto *pdf_data = reinterpret_cast<const uint8_t *>(pdf_buf->data());
      size_t pdf_len_local = pdf_buf->size();

      if (reject_if_too_many_pages(pdf_data, pdf_len_local, max_pdf_pages, cb)) return;

      // Open PDF for text-layer modes
      pdf::PdfMode mode = req_mode;
      std::unique_ptr<pdf::PdfDocument> pdf_doc;
      std::vector<pdf::PdfPageText> page_text_cache;
      open_pdf_for_text_layer(pdf_data, pdf_len_local, mode,
                              pdf_doc, page_text_cache);

      // Shared state — uses the file-scope `fill_from_text_layer_pt` and
      // `text_layer_quality_for` helpers shared with the CPU route.
      std::mutex results_mutex;
      std::vector<GpuPdfPageResult> page_results;

      // Pre-populate pages that don't need rendering
      std::vector<uint8_t> need_render;
      bool any_need_render = (mode == pdf::PdfMode::Ocr);

      if (mode != pdf::PdfMode::Ocr) {
        prepopulate_pages(mode, layout_enabled, page_text_cache,
                          page_results, need_render, &any_need_render,
                          image_mode == ImageMode::Inline);
      }

      // Streamed render + OCR.
      //
      // The StreamHandle owns the scratch tmpdir holding the rendered
      // PPMs. It MUST outlive every future in `page_futures`, because
      // those futures hold ppm_path strings that the dispatcher workers
      // open lazily — destroying the handle while futures are pending
      // unlinks the PPMs out from under them and they decode to empty
      // images. Declared at handler scope and intentionally kept alive
      // through the f.get() loop below.
      // Everything the async page tasks reference is declared BEFORE
      // page_futures so it outlives the futures' blocking destructors on
      // EVERY unwind path (not just the happy join below): `sink` (holds the
      // decode_failures atomic), `dropped_pages`, and `stream_handle` (owns
      // the PPM tmpdir the tasks read). Reverse-destruction then joins the
      // futures before any of these go away.
      PdfPageSink sink{results_mutex, page_results, pdf_doc.get(),
                       page_text_cache, dpi, image_mode, encode_opts,
                       autorotate};
      std::vector<int> dropped_pages;  // pages that couldn't be queued -> 503
      render::PdfRenderer::StreamHandle stream_handle;

      std::mutex futures_mutex;
      std::vector<std::future<void>> page_futures;
      int num_pages = 0;

      if (any_need_render) {
        try {
          stream_handle = run_streamed_render_gpu(dispatcher, pdf_renderer,
                                   pdf_data, pdf_len_local,
                                   layout_enabled, want_reading_order, mode,
                                   sink, need_render,
                                   page_futures, futures_mutex, pdf_only_batch,
                                   dropped_pages);
          num_pages = stream_handle.num_pages;
        } catch (const std::exception &e) {
          for (auto &f : page_futures) { try { f.get(); } catch (...) {} }
          TOCR_LOG_ERROR("PDF render failed", "route", "/ocr/pdf", "error", std::string_view(e.what()));
          cb(server::error_response(drogon::k400BadRequest, "PDF_RENDER_FAILED", "PDF render failed"));
          return;
        }
      } else {
        num_pages = pdf_doc ? pdf_doc->page_count() : 0;
      }

      {
        std::lock_guard<std::mutex> rlock(results_mutex);
        if (static_cast<int>(page_results.size()) < num_pages)
          page_results.resize(num_pages);
      }

      for (auto &f : page_futures) {
        try { f.get(); } catch (const std::exception &e) {
          TOCR_LOG_ERROR("PDF page error", "route", "/ocr/pdf", "error", std::string_view(e.what()));
        }
      }

      if (num_pages == 0) {
        cb(server::error_response(drogon::k400BadRequest, "EMPTY_PDF", "PDF contains no pages"));
        return;
      }

      if (!dropped_pages.empty()) {
        cb(server::error_response(drogon::k503ServiceUnavailable, "SERVER_BUSY",
            std::format("GPU queue full: {} of {} pages could not be processed "
                        "(first dropped page: {}). Retry with backoff.",
                        dropped_pages.size(), num_pages, dropped_pages.front())));
        return;
      }

      if (const int failed = sink.decode_failures.load(std::memory_order_relaxed);
          failed > 0) {
        // We rendered these PPMs ourselves — failing to read them back is a
        // server-side fault (tmpfs pressure, truncated write), not client
        // input. Surfacing 500 beats emitting silently blank pages.
        cb(server::error_response(drogon::k500InternalServerError,
            "PAGE_DECODE_FAILED",
            std::format("{} of {} rendered pages could not be decoded; retry",
                        failed, num_pages)));
        return;
      }

      // num_pages may have grown the vector under page_futures completion;
      // trim to its actual number reported by the renderer.
      std::vector<GpuPdfPageResult> trimmed;
      {
        std::lock_guard<std::mutex> rlock(results_mutex);
        trimmed.reserve(num_pages);
        for (int i = 0; i < num_pages && i < static_cast<int>(page_results.size()); ++i)
          trimmed.push_back(std::move(page_results[i]));
      }
      cb(server::json_response(emit_pdf_response(trimmed, dpi, want_blocks,
                                                  image_mode, encode_opts,
                                                  autorotate)));
     });
    });
  }, {drogon::Post});
}
#endif // !USE_CPU_ONLY

// --- CPU overload: sequential page OCR via InferFunc ---
void register_pdf_route(server::WorkPool &pool,
                        const server::InferFunc &infer,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available,
                        int max_pdf_pages,
                        server::OrientFunc orient_fn) {
  const bool doc_ori_available = static_cast<bool>(orient_fn);

  drogon::app().registerHandler(
      "/ocr/pdf",
      [&pool, &infer, &pdf_renderer, default_pdf_mode, layout_available,
       max_pdf_pages, orient_fn, doc_ori_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    std::string decoded_buf;
    const char *pdf_ptr = nullptr;
    size_t pdf_len = 0;

    if (!extract_pdf_bytes(req, decoded_buf, pdf_ptr, pdf_len, callback))
      return;

    server::InferOptions opts;
    if (auto r = server::parse_query_options(req, layout_available, &opts);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }
    const bool want_layout = opts.want_layout;
    const bool want_reading_order = opts.want_reading_order;
    const bool want_blocks = opts.want_blocks;

    if (reject_unknown_query_params(
            req, {"layout", "reading_order", "as_blocks", "dpi", "mode",
                  "images", "format", "lossless", "png_compression", "quality",
                  "max_side", "autorotate"}, callback))
      return;

    auto dpi_str = req->getParameter("dpi");
    int dpi = dpi_str.empty() ? kCpuDefaultDpi : query_int(std::string(dpi_str), -1);
    if (dpi < kMinPdfDpi || dpi > kMaxPdfDpi) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_DPI",
          std::format("DPI must be between {} and {}", kMinPdfDpi, kMaxPdfDpi)));
      return;
    }

    pdf::PdfMode req_mode = default_pdf_mode;
    auto mode_str = req->getParameter("mode");
    if (!mode_str.empty())
      req_mode = pdf::parse_pdf_mode(mode_str.c_str(), default_pdf_mode);

    // Page-image export params (?images=inline&format=...&quality=...)
    ImageMode image_mode;
    pdf::EncodeOptions encode_opts;
    if (auto err = parse_image_query_params(req, image_mode, encode_opts);
        !err.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       "INVALID_PARAMETER", err));
      return;
    }

    bool autorotate = false;
    if (auto err = server::parse_bool_query(req, "autorotate", &autorotate);
        !err.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       "INVALID_PARAMETER", err));
      return;
    }
    if (autorotate && !doc_ori_available) {
      callback(server::error_response(drogon::k400BadRequest, "AUTOROTATE_DISABLED",
          "autorotate=1 requires the doc-orientation model (models/doc_ori.onnx); "
          "it was not found at startup"));
      return;
    }

    auto pdf_buf = std::make_shared<std::string>(pdf_ptr, pdf_len);

    server::submit_work(pool, std::move(callback),
        [pdf_buf, &infer, &pdf_renderer, want_layout,
         want_reading_order, want_blocks, dpi,
         req_mode, image_mode, encode_opts, max_pdf_pages,
         autorotate, orient_fn](server::DrogonCallback &cb) {
     // See GPU route: wrap the body so a post-render bad_alloc returns 500
     // instead of being swallowed by the WorkPool (client hang).
     server::run_with_error_handling(cb, "/ocr/pdf", [&] {
      const auto *pdf_data = reinterpret_cast<const uint8_t *>(pdf_buf->data());
      size_t pdf_len_local = pdf_buf->size();

      if (reject_if_too_many_pages(pdf_data, pdf_len_local, max_pdf_pages, cb)) return;

      // CPU server runs sequentially; the GPU AutoVerified path
      // cross-checks every OCR detection against the text layer in
      // parallel. Doing the same on CPU would require an extra pdfium
      // text_in_rect call per detection per page, doubling latency on a
      // single-thread pipeline. Honest behavior: alias auto_verified to
      // auto on CPU and emit the actually-resolved per-page mode in the
      // response, so clients who set auto_verified get auto's text-layer
      // fast-path without us claiming verification we didn't perform.
      pdf::PdfMode mode = req_mode;
      if (mode == pdf::PdfMode::AutoVerified) mode = pdf::PdfMode::Auto;

      // Open PDF for text-layer extraction when the resolved mode needs it.
      // For mode=ocr we skip this entirely (matches the legacy CPU path).
      std::unique_ptr<pdf::PdfDocument> pdf_doc;
      std::vector<pdf::PdfPageText> page_text_cache;
      open_pdf_for_text_layer(pdf_data, pdf_len_local, mode,
                              pdf_doc, page_text_cache);

      std::vector<PdfPageResultBase> page_results;
      std::vector<uint8_t> need_render;

      // Decide per-page resolved mode up front. mode=ocr always renders;
      // mode=geometric / mode=auto consult the text layer first and only
      // render when necessary.
      if (mode != pdf::PdfMode::Ocr) {
        prepopulate_pages(mode, want_layout, page_text_cache,
                          page_results, need_render, /*any_need_render=*/nullptr,
                          image_mode == ImageMode::Inline);
      }

      // Render + OCR pass. mode=ocr runs every page; non-ocr modes only
      // render pages that flagged need_render (image-only pages, layout=1
      // requests, auto-fallback OCR pages, auto_verified pages).
      int decode_failures = 0;
      try {
        bool any_need_render = (mode == pdf::PdfMode::Ocr) ||
            std::any_of(need_render.begin(), need_render.end(),
                        [](uint8_t v) { return v != 0; });

        if (any_need_render) {
          int num_pages = run_streamed_render_cpu(infer, pdf_renderer,
              pdf_data, pdf_len_local, dpi, want_layout,
              want_reading_order, mode,
              page_results, need_render, decode_failures,
              image_mode, encode_opts, autorotate, orient_fn);
          if (static_cast<int>(page_results.size()) < num_pages)
            page_results.resize(num_pages);
        }
      } catch (const std::exception &e) {
        cb(server::error_response(drogon::k400BadRequest, "PDF_RENDER_FAILED",
            std::format("PDF render failed: {}", e.what())));
        return;
      }

      if (page_results.empty()) {
        cb(server::error_response(drogon::k400BadRequest, "EMPTY_PDF",
            "PDF contains no pages"));
        return;
      }

      if (decode_failures > 0) {
        // We rendered these PPMs ourselves; a read-back failure is a
        // server-side fault (tmpfs pressure / truncated write). Surface 500
        // instead of emitting silently blank pages (parity with GPU route).
        cb(server::error_response(drogon::k500InternalServerError,
            "PAGE_DECODE_FAILED",
            std::format("{} of {} rendered pages could not be decoded; retry",
                        decode_failures, static_cast<int>(page_results.size()))));
        return;
      }

      cb(server::json_response(emit_pdf_response(page_results, dpi, want_blocks,
                                                  image_mode, encode_opts,
                                                  autorotate)));
     });
    });
  }, {drogon::Post});
}

} // namespace turbo_ocr::routes
