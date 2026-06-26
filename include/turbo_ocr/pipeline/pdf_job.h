#pragma once

// Transport-agnostic PDF orchestrator (H9).
//
// Both the HTTP /ocr/pdf handler (src/routes/pdf_routes.cpp) and the gRPC
// RecognizePDF RPC (include/turbo_ocr/server/grpc_service.h) used to carry a
// near-identical, drifted copy of the per-page PDF pipeline: the text-layer
// pre-pass, the streamed render + OCR fan-out, and the per-page result
// accumulation. This header hosts the single shared implementation; each
// transport calls run_pdf_job() and then serialises the returned per-page
// results into its own envelope (HTTP JSON object vs gRPC proto message).
//
// The orchestrator's PdfJobResult is the contract between the pipeline and the
// transports. It deliberately mirrors the fields the prior code emitted, so
// the serialised output is byte-identical to today's for the same input/mode.
//
// Concurrency (H3): page work is submitted DIRECTLY onto the bounded
// PipelineDispatcher queue (the model the HTTP path already used). Backpressure
// is the dispatcher queue depth (PoolExhaustedError -> the caller maps to 503 /
// RESOURCE_EXHAUSTED); there is no per-page std::async / counting_semaphore.
// The GPU job bounds its page-future join against ONE job-wide deadline
// (request_timeout_ms) so a worker wedged on a single page can't hang the whole
// request. A future that overruns is ABANDONED — its task keeps running — so
// every page task co-owns the PdfPageSink and the render StreamHandle by
// shared_ptr (captured by value): the shared state lives until the last task,
// including any abandoned one, drops it, and no task ever touches a dead object.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/classification/doc_orientation_common.h"
#include "turbo_ocr/common/logger.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/layout/reading_order.h"
#include "turbo_ocr/pdf/page_image_encoder.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/server_types.h"

#ifndef USE_CPU_ONLY
#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#endif

namespace turbo_ocr::pipeline {

// Accepted PDF render DPI range — shared by every transport's dpi validation
// (HTTP pdf_routes + gRPC RecognizePDF) so the bound + message can't drift.
constexpr int kMinPdfDpi = 50;
constexpr int kMaxPdfDpi = 600;

// Image-export mode for ?images=... — only "inline" is supported on HTTP; gRPC
// never requests it. Kept here so the orchestrator carries the parity flag.
enum class PdfImageMode { None, Inline };

// One page's resolved output. Shared by every transport. Field-for-field the
// same data the prior HTTP `PdfPageResultBase` / gRPC `PdfPageResult` carried.
struct PdfPageResult {
  std::vector<OCRResultItem>     results;
  std::vector<layout::LayoutBox> layout;
  std::vector<int>               reading_order;
  int width = 0, height = 0, effective_dpi = 0;
  pdf::PdfMode resolved_mode = pdf::PdfMode::Ocr;
  std::string_view text_layer_quality = "absent";
  // Detected page rotation (clockwise, 0/90/180/270) when autorotate=1.
  int orientation_deg = 0;
  // Encoded page-image bytes (set only when image_mode == Inline).
  std::vector<uint8_t> encoded_image;
};

// Per-request options for run_pdf_job. Defaults preserve today's behaviour.
struct PdfJobOptions {
  int dpi = 100;
  pdf::PdfMode mode = pdf::PdfMode::Ocr;
  bool want_layout = false;
  bool want_reading_order = false;
  bool want_blocks = false;
  bool autorotate = false;
  PdfImageMode image_mode = PdfImageMode::None;
  pdf::EncodeOptions encode_opts{};
  // pdf_only batched-chunk size (GPU only; 0 = per-page path).
  int pdf_only_batch = 0;
  // Per-request deadline (ms; 0 = unbounded). The GPU job bounds its page-future
  // join against this so a worker wedged mid-page can't hang the whole request.
  long request_timeout_ms = 0;
};

// Outcome category for run_pdf_job. The transport maps each to its own error
// surface (HTTP status / gRPC StatusCode) via the shared error_codes table.
enum class PdfJobStatus {
  Ok,
  EmptyPdf,        // no pages -> EMPTY_PDF
  RenderFailed,    // renderer threw -> PDF_RENDER_FAILED
  Dropped,         // GPU queue full mid-stream -> SERVER_BUSY (503 / RES_EXH)
  DecodeFailed,    // rendered PPM unreadable -> PAGE_DECODE_FAILED (500 / INTERNAL)
  PageFailed,      // a page's OCR/inference threw -> PAGE_FAILED (500 / INTERNAL)
  TimedOut,        // job-wide deadline overrun -> INFERENCE_TIMEOUT (504 / DEADLINE_EXCEEDED)
};

struct PdfJobResult {
  PdfJobStatus status = PdfJobStatus::Ok;
  std::vector<PdfPageResult> pages;
  int num_pages = 0;
  int dropped_pages = 0;   // count, for the SERVER_BUSY message
  int first_dropped = -1;  // first dropped page index, for the HTTP message
  int decode_failures = 0;
  int page_failures = 0;   // pages whose inference threw; any > 0 fails the job
};

// ── Text-layer helpers (shared) ──────────────────────────────────────────
//
// `geometric` mode reads the PDF's native text layer directly. `auto` falls
// back to OCR when the layer is unusable. text_layer_quality_for() decides:
//   absent   — no chars / no lines / < 10 chars (looks scanned)
//   rejected — non-zero rotation, too many U+FFFD or non-printable chars
//   trusted  — usable text returned without OCR.
// fill_from_text_layer_pt() copies that text + boxes into the page at PDF-point
// coordinates (DPI=72); the caller rescales to pixels only for geometric pages.

inline void fill_from_text_layer_pt(PdfPageResult &pg,
                                    const pdf::PdfPageText &text) {
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

inline std::string_view text_layer_quality_for(const pdf::PdfPageText &text) {
  if (text.char_count == 0)                         return "absent";
  if (text.rotation_deg != 0)                       return "rejected";
  if (text.char_count < 10)                         return "absent";
  if (text.fffd_count * 20 > text.char_count)       return "rejected";
  if (text.nonprint_count * 10 > text.char_count)   return "rejected";
  if (text.lines.empty())                           return "absent";
  return "trusted";
}

// Open the PDF and pre-extract per-page text only when the mode needs it.
// mode=ocr skips this. On open failure, downgrade to ocr and clear the doc.
inline void open_pdf_for_text_layer(
    const uint8_t *pdf_data, size_t pdf_len, pdf::PdfMode &mode,
    std::unique_ptr<pdf::PdfDocument> &pdf_doc,
    std::vector<pdf::PdfPageText> &page_text_cache) {
  if (mode == pdf::PdfMode::Ocr) return;
  pdf_doc = std::make_unique<pdf::PdfDocument>(pdf_data, pdf_len);
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

// Decide per-page resolved_mode + whether each page needs rendering, from the
// text-layer quality. Only called for non-ocr modes. AutoVerified is GPU-only
// (CPU aliases it to Auto before calling). `want_page_image` forces a render
// for text-layer pages so the encoder has pixels.
inline void prepopulate_pages(pdf::PdfMode mode, bool layout_or_want_layout,
                              const std::vector<pdf::PdfPageText> &page_text_cache,
                              std::vector<PdfPageResult> &page_results,
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

    // A page image was requested but this is a text-layer page that would
    // otherwise skip rasterization — force a render so the encoder has pixels.
    if (want_page_image && !need_render[static_cast<size_t>(p)]) {
      need_render[static_cast<size_t>(p)] = 1;
      if (any_need_render) *any_need_render = true;
    }
  }
}

// ── Shared per-page serializer (H7) ──────────────────────────────────────
//
// Produces the JSON body for one page's results (without any envelope). Both
// transports call this so the per-page result shape cannot drift: HTTP embeds
// it inside its {pages:[...]} object; gRPC stores it in OCRPageResult's
// json_response bytes. Mirrors the prior `emit_results_json` / `results_to_json`
// branch both sites used, so output is byte-identical.
[[nodiscard]] inline std::string
serialize_page_results(PdfPageResult &pg, bool want_blocks) {
  if (!pg.reading_order.empty())
    return emit_results_json(pg.results, pg.layout, pg.reading_order,
                             want_blocks);
  return results_to_json(pg.results, pg.layout);
}

// ── Inference backend abstraction ────────────────────────────────────────
//
// The orchestrator runs the same control flow on both backends. The GPU
// backend submits directly onto the dispatcher (H3); the CPU backend runs the
// synchronous InferFunc inline. Only the leaf "run inference on these pixels"
// step differs, so both share prepopulate/streamed-render orchestration.

namespace detail {

// Shared state every page task writes into. Held by shared_ptr and captured BY
// VALUE into every page task, so a task abandoned on a deadline timeout (its
// future get()-with-timeout abandons it) safely outlives run_pdf_job: the sink
// stays alive until the last task that holds it finishes. It therefore OWNS the
// mutex / results / text cache / document outright (rather than referencing
// run_pdf_job locals that would die at return).
struct PdfPageSink {
  std::mutex results_mutex;
  std::vector<PdfPageResult> page_results;
  std::unique_ptr<pdf::PdfDocument> pdf_doc;  // null when no text layer was opened
  std::vector<pdf::PdfPageText> page_text_cache;
  int dpi = 0;
  PdfImageMode image_mode = PdfImageMode::None;
  pdf::EncodeOptions encode_opts{};
  bool autorotate = false;
  // Rendered pages whose PPM could not be read back (a server-side fault).
  std::atomic<int> decode_failures{0};
  // Pages whose OCR/inference threw — counted by the true page count (the whole
  // chunk on the batched path), so any > 0 fails the job rather than returning a
  // silently-empty page in a 200.
  std::atomic<int> page_failures{0};
};

[[nodiscard]] inline std::vector<uint8_t>
maybe_encode_page(const PdfPageSink &sink, const cv::Mat &img) {
  if (sink.image_mode != PdfImageMode::Inline) return {};
  return pdf::encode_page_image(img, sink.encode_opts);
}

[[nodiscard]] inline pdf::PdfMode page_mode_of(PdfPageSink &sink, int page_idx) {
  std::lock_guard<std::mutex> lock(sink.results_mutex);
  return page_idx < static_cast<int>(sink.page_results.size())
      ? sink.page_results[page_idx].resolved_mode
      : pdf::PdfMode::Ocr;
}

// Store one OCR'd page: tag sources, apply auto_verified text-layer
// replacement, write the slot under the sink lock.
inline void store_ocr_page(PdfPageSink &sink, int page_idx,
                           OcrPipelineResult out, int width, int height,
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

// Geometric page: text came from the PDF layer in pt-space. Store layout, then
// rescale the stored text boxes to pixel space; compute reading order over the
// (now pixel-space) text + layout when requested, for parity with OCR pages.
inline void store_geometric_page(PdfPageSink &sink, int page_idx,
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

#ifndef USE_CPU_ONLY
// Single-page GPU task: decode the PPM and run layout-only (Geometric) or the
// full pipeline (everything else). Runs on a dispatcher worker.
inline void ocr_single_page(GpuPipelineEntry &e, PdfPageSink &sink,
                            bool layout_enabled, bool want_reading_order,
                            int page_idx, const std::string &path) {
  cv::Mat img = render::PdfRenderer::decode_ppm(path);
  if (img.empty()) {
    TOCR_LOG_ERROR("Failed to decode PPM for page",
                   "route", "/ocr/pdf", "page", page_idx);
    sink.decode_failures.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  try {
    if (page_mode_of(sink, page_idx) == pdf::PdfMode::Geometric) {
      // Geometric pages are born-digital/upright with pt-space text boxes;
      // autorotate does not apply (rotating would desync the boxes). Encode +
      // run layout-only when requested.
      auto encoded = maybe_encode_page(sink, img);
      auto layout = layout_enabled
          ? e.pipeline->run_layout_only(img, e.stream).layout
          : std::vector<layout::LayoutBox>{};
      store_geometric_page(sink, page_idx, std::move(layout),
                           img.cols, img.rows, want_reading_order,
                           std::move(encoded));
    } else {
      // OCR page: de-rotate upright FIRST (autorotate=1) so det/rec, the boxes,
      // and the encoded image all share one upright frame.
      int orient = sink.autorotate
          ? e.pipeline->detect_orientation(img, e.stream) : 0;
      if (orient) classification::rotate_upright(img, orient);
      auto encoded = maybe_encode_page(sink, img);
      auto out = e.pipeline->run_with_layout(img, e.stream, layout_enabled,
                                             want_reading_order);
      store_ocr_page(sink, page_idx, std::move(out), img.cols, img.rows,
                     std::move(encoded), orient);
    }
  } catch (const std::exception &ex) {
    // Leave the slot default (empty); the orchestrator turns any page_failures
    // into a job-level PAGE_FAILED rather than a 200 with a blank page.
    sink.page_failures.fetch_add(1, std::memory_order_relaxed);
    TOCR_LOG_ERROR("PDF page inference error", "route", "/ocr/pdf",
                   "page", page_idx, "error", std::string_view(ex.what()));
  }
}

// pdf_only chunk task: decode a chunk of rendered pages and run the batched
// path — one static-shape det execute + batched rec + one batched layout
// execute per chunk. Runs on a dispatcher worker.
inline void ocr_page_chunk(GpuPipelineEntry &e, PdfPageSink &sink,
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
    int orient = sink.autorotate
        ? e.pipeline->detect_orientation(img, e.stream) : 0;
    if (orient) classification::rotate_upright(img, orient);
    imgs.push_back(std::move(img));
    live.push_back(idxs[j]);
    orients.push_back(orient);
  }
  if (imgs.empty()) return;

  try {
    auto outs = e.pipeline->run_batch_with_layout(imgs, e.stream, layout_enabled,
                                                  want_reading_order);
    for (size_t j = 0; j < outs.size(); ++j)
      store_ocr_page(sink, live[j], std::move(outs[j]),
                     imgs[j].cols, imgs[j].rows,
                     maybe_encode_page(sink, imgs[j]), orients[j]);
  } catch (const std::exception &ex) {
    // The whole batch shares one execute — a throw blanks every live page in
    // the chunk, so count them all (not 1) for an accurate PAGE_FAILED.
    sink.page_failures.fetch_add(static_cast<int>(live.size()),
                                 std::memory_order_relaxed);
    TOCR_LOG_ERROR("PDF page chunk inference error", "route", "/ocr/pdf",
                   "first_page", live.front(), "pages", live.size(),
                   "error", std::string_view(ex.what()));
  }
}

// GPU streamed render: per rendered page, accumulate into a pdf_only batch
// chunk or submit a single-page task directly onto the dispatcher (H3 — no
// per-page OS thread). `sink` and `stream_handle` are shared_ptrs captured BY
// VALUE into every task so an abandoned (timed-out) task safely outlives this
// call: the sink and the scratch tmpdir live until the last task drops them.
// Populates *stream_handle with the render result and returns its num_pages.
[[nodiscard]] inline int run_streamed_render_gpu(
    PipelineDispatcher &dispatcher, render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len, bool layout_enabled,
    bool want_reading_order, pdf::PdfMode mode,
    const std::shared_ptr<PdfPageSink> &sink,
    const std::shared_ptr<render::PdfRenderer::StreamHandle> &stream_handle,
    std::vector<uint8_t> &need_render,
    std::vector<std::future<void>> &page_futures, std::mutex &futures_mutex,
    int pdf_only_batch, std::vector<int> &dropped_pages) {
  std::vector<int> acc_idxs;
  std::vector<std::string> acc_paths;

  auto submit_ocr_chunk = [&](std::vector<int> idxs,
                              std::vector<std::string> paths) {
    std::future<void> fut;
    try {
      // idxs is copied (<= pdf_only_batch ints) rather than moved: submit()
      // consumes the move BEFORE it can throw PoolExhausted, and the catch
      // below still needs the indices.
      fut = dispatcher.submit(
          [sink, stream_handle, layout_enabled, want_reading_order,
           idxs, paths = std::move(paths)](auto &e) {
            ocr_page_chunk(e, *sink, layout_enabled, want_reading_order,
                           idxs, paths);
          });
    } catch (const turbo_ocr::PoolExhaustedError &) {
      TOCR_LOG_WARN("GPU queue full, dropping page chunk", "route", "/ocr/pdf",
                    "first_page", idxs.front(), "pages", idxs.size());
      dropped_pages.insert(dropped_pages.end(), idxs.begin(), idxs.end());
      return;
    }
    std::lock_guard lock(futures_mutex);
    page_futures.push_back(std::move(fut));
  };

  auto handle = pdf_renderer.render_streamed(pdf_data, pdf_len, sink->dpi,
      [&](int page_idx, std::string ppm_path) {
        bool is_geometric = false;
        {
          std::lock_guard<std::mutex> rlock(sink->results_mutex);
          auto &page_results = sink->page_results;
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
              [sink, stream_handle, layout_enabled, want_reading_order, page_idx,
               path = std::move(ppm_path)](auto &e) {
                ocr_single_page(e, *sink, layout_enabled, want_reading_order,
                                page_idx, path);
              });
        } catch (const turbo_ocr::PoolExhaustedError &) {
          TOCR_LOG_WARN("GPU queue full, dropping page", "route", "/ocr/pdf",
                        "page", page_idx);
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

  // Hand the tmpdir-owning handle to the shared object the tasks co-own; the
  // tmpdir now lives until the last task (including any abandoned one) drops it.
  *stream_handle = std::move(handle);
  return stream_handle->num_pages;
}
#endif // !USE_CPU_ONLY

// CPU streamed render. Sequential: decode PPM, run the InferFunc inline.
// mode==Ocr means we never visited prepopulate_pages, so resolved mode is
// pinned here. Geometric pages keep their layer-derived text and rescale
// point->pixel coords. Returns num_pages; reports unreadable rendered PPMs via
// `decode_failures` (a server-side fault). The callback body is fully wrapped
// in try/catch: it runs INSIDE render_streamed's poll loop on the thread that
// owns the still-joinable render thread, so an escaping throw would unwind past
// that std::thread and std::terminate the process.
inline int run_streamed_render_cpu(
    const server::InferFunc &infer, render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len, int dpi, bool want_layout,
    bool want_reading_order, pdf::PdfMode mode,
    std::vector<PdfPageResult> &page_results,
    const std::vector<uint8_t> &need_render, int &decode_failures,
    int &page_failures, PdfImageMode image_mode,
    const pdf::EncodeOptions &encode_opts,
    bool autorotate, const server::OrientFunc &orient_fn) {
  auto stream_handle = pdf_renderer.render_streamed(pdf_data, pdf_len, dpi,
      [&](int page_idx, const std::string &ppm_path) noexcept {
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

        server::InferOptions inf_opts;
        inf_opts.want_layout = want_layout;
        inf_opts.want_reading_order = want_reading_order;
        if (mode == pdf::PdfMode::Ocr)
          pg.resolved_mode = pdf::PdfMode::Ocr;

        // OCR pages: de-rotate upright (autorotate=1) BEFORE encode + infer.
        // Geometric pages are born-digital/upright with pt-space text — skip.
        if (pg.resolved_mode != pdf::PdfMode::Geometric && autorotate && orient_fn) {
          int orient = orient_fn(img);
          if (orient) classification::rotate_upright(img, orient);
          pg.orientation_deg = orient;
        }

        if (image_mode == PdfImageMode::Inline)
          pg.encoded_image = pdf::encode_page_image(img, encode_opts);

        if (pg.resolved_mode == pdf::PdfMode::Geometric) {
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
          if (want_reading_order && !pg.layout.empty()) {
            turbo_ocr::assign_layout_ids(pg.results, pg.layout);
            pg.reading_order =
                turbo_ocr::layout::assign_reading_order_for_results(
                    pg.results, pg.layout);
          }
        } else {
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
        ++page_failures;
        TOCR_LOG_ERROR("PDF page inference error", "route", "/ocr/pdf",
                       "page", page_idx, "error", std::string_view(e.what()));
       } catch (...) {
        ++page_failures;
        TOCR_LOG_ERROR("PDF page inference error (unknown)",
                       "route", "/ocr/pdf", "page", page_idx);
       }
      });
  return stream_handle.num_pages;
}

} // namespace detail

// ── Orchestrator entry points ────────────────────────────────────────────

#ifndef USE_CPU_ONLY
// GPU PDF job. Submits page work directly onto the dispatcher (H3). Page tasks
// co-own the sink + StreamHandle by shared_ptr, so a task abandoned when its
// future overruns the job-wide deadline (request_timeout_ms) is memory-safe:
// the shared state outlives this call until that task finishes. Backpressure is
// the dispatcher queue depth: PoolExhaustedError mid-stream sets status=Dropped
// (caller -> SERVER_BUSY).
[[nodiscard]] inline PdfJobResult run_pdf_job(
    PipelineDispatcher &dispatcher, render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len, const PdfJobOptions &opts) {
  PdfJobResult job;

  // The sink OWNS every piece of state a page task touches (results, mutex,
  // text cache, document), heap-allocated and co-owned by the tasks, so an
  // abandoned task never dereferences a dead run_pdf_job local.
  auto sink = std::make_shared<detail::PdfPageSink>();
  sink->dpi = opts.dpi;
  sink->image_mode = opts.image_mode;
  sink->encode_opts = opts.encode_opts;
  sink->autorotate = opts.autorotate;

  pdf::PdfMode mode = opts.mode;
  open_pdf_for_text_layer(pdf_data, pdf_len, mode, sink->pdf_doc,
                          sink->page_text_cache);

  std::vector<uint8_t> need_render;
  bool any_need_render = (mode == pdf::PdfMode::Ocr);

  if (mode != pdf::PdfMode::Ocr)
    prepopulate_pages(mode, opts.want_layout, sink->page_text_cache,
                      sink->page_results, need_render, &any_need_render,
                      opts.image_mode == PdfImageMode::Inline);

  std::vector<int> dropped;
  auto stream_handle = std::make_shared<render::PdfRenderer::StreamHandle>();

  std::mutex futures_mutex;
  std::vector<std::future<void>> page_futures;
  int num_pages = 0;

  if (any_need_render) {
    try {
      num_pages = detail::run_streamed_render_gpu(
          dispatcher, pdf_renderer, pdf_data, pdf_len, opts.want_layout,
          opts.want_reading_order, mode, sink, stream_handle, need_render,
          page_futures, futures_mutex, opts.pdf_only_batch, dropped);
    } catch (const std::exception &e) {
      // No drain needed: every in-flight page task co-owns the sink/handle via
      // shared_ptr, so abandoning the futures here is memory-safe (a still-running
      // task can't dereference a dead local). An unbounded f.get() drain would
      // instead let one wedged worker hang the whole render-failure response.
      TOCR_LOG_ERROR("PDF render failed", "route", "/ocr/pdf",
                     "error", std::string_view(e.what()));
      job.status = PdfJobStatus::RenderFailed;
      return job;
    }
  } else {
    num_pages = sink->pdf_doc ? sink->pdf_doc->page_count() : 0;
  }

  {
    std::lock_guard<std::mutex> rlock(sink->results_mutex);
    if (static_cast<int>(sink->page_results.size()) < num_pages)
      sink->page_results.resize(num_pages);
  }

  // Bounded join against ONE job-wide deadline: a worker wedged on a single page
  // can't hang the whole request. The deadline is SCALED by page count — each
  // page gets request_timeout_ms, the whole job num_pages * that — so a flat 60s
  // budget can't kill a big-but-healthy PDF. On a per-future throw the future is
  // abandoned (the task keeps its shared_ptr copies, so this is safe) and the page
  // is counted as failed; the catch also backstops any UNEXPECTED escape (the
  // worker bodies catch their own) so nothing slips through as a silently-empty
  // page in a 200. A pure deadline overrun (TimeoutError) is NOT an inference page
  // failure: it surfaces as a distinct 504 / DEADLINE_EXCEEDED.
  const long per_page_budget = opts.request_timeout_ms;  // 0 == unbounded
  const long long job_budget =
      per_page_budget > 0
          ? static_cast<long long>(per_page_budget) * std::max(1, num_pages)
          : 0;
  const auto job_deadline = std::chrono::steady_clock::now() +
                            std::chrono::milliseconds(job_budget);
  bool deadline_exceeded = false;
  for (auto &f : page_futures) {
    try {
      if (job_budget > 0) {
        const auto now = std::chrono::steady_clock::now();
        long long remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                job_deadline - now).count();
        if (remaining < 1) remaining = 1;  // already past: near-instant abandon
        get_with_timeout(f, static_cast<long>(remaining));
      } else {
        f.get();
      }
    } catch (const turbo_ocr::TimeoutError &) {
      // Job-wide deadline overrun, not a page-level inference failure. Abandon the
      // remaining futures (their tasks co-own the sink) and report a 504.
      deadline_exceeded = true;
      break;
    } catch (const std::exception &e) {
      sink->page_failures.fetch_add(1, std::memory_order_relaxed);
      TOCR_LOG_ERROR("PDF page error", "route", "/ocr/pdf",
                     "error", std::string_view(e.what()));
      continue;
    }
  }

  job.num_pages = num_pages;
  if (num_pages == 0) { job.status = PdfJobStatus::EmptyPdf; return job; }

  if (deadline_exceeded) { job.status = PdfJobStatus::TimedOut; return job; }

  if (!dropped.empty()) {
    job.status = PdfJobStatus::Dropped;
    job.dropped_pages = static_cast<int>(dropped.size());
    job.first_dropped = dropped.front();
    return job;
  }

  if (const int failed = sink->decode_failures.load(std::memory_order_relaxed);
      failed > 0) {
    job.status = PdfJobStatus::DecodeFailed;
    job.decode_failures = failed;
    return job;
  }

  if (const int failed = sink->page_failures.load(std::memory_order_relaxed);
      failed > 0) {
    job.status = PdfJobStatus::PageFailed;
    job.page_failures = failed;
    return job;
  }

  // num_pages may have grown the vector under page_futures completion; trim to
  // the renderer-reported count.
  {
    std::lock_guard<std::mutex> rlock(sink->results_mutex);
    job.pages.reserve(num_pages);
    for (int i = 0; i < num_pages && i < static_cast<int>(sink->page_results.size()); ++i)
      job.pages.push_back(std::move(sink->page_results[i]));
  }
  return job;
}
#endif // !USE_CPU_ONLY

// CPU PDF job. Sequential page OCR via the synchronous InferFunc. AutoVerified
// is GPU-only, so the caller aliases it to Auto before invoking.
[[nodiscard]] inline PdfJobResult run_pdf_job(
    const server::InferFunc &infer, render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len, const PdfJobOptions &opts,
    const server::OrientFunc &orient_fn) {
  PdfJobResult job;

  pdf::PdfMode mode = opts.mode;
  std::unique_ptr<pdf::PdfDocument> pdf_doc;
  std::vector<pdf::PdfPageText> page_text_cache;
  open_pdf_for_text_layer(pdf_data, pdf_len, mode, pdf_doc, page_text_cache);

  std::vector<PdfPageResult> page_results;
  std::vector<uint8_t> need_render;

  if (mode != pdf::PdfMode::Ocr)
    prepopulate_pages(mode, opts.want_layout, page_text_cache, page_results,
                      need_render, /*any_need_render=*/nullptr,
                      opts.image_mode == PdfImageMode::Inline);

  int decode_failures = 0;
  int page_failures = 0;
  try {
    bool any_need_render = (mode == pdf::PdfMode::Ocr) ||
        std::any_of(need_render.begin(), need_render.end(),
                    [](uint8_t v) { return v != 0; });
    if (any_need_render) {
      int num_pages = detail::run_streamed_render_cpu(
          infer, pdf_renderer, pdf_data, pdf_len, opts.dpi, opts.want_layout,
          opts.want_reading_order, mode, page_results, need_render,
          decode_failures, page_failures, opts.image_mode, opts.encode_opts,
          opts.autorotate, orient_fn);
      if (static_cast<int>(page_results.size()) < num_pages)
        page_results.resize(num_pages);
    }
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR("PDF render failed", "route", "/ocr/pdf",
                   "error", std::string_view(e.what()));
    job.status = PdfJobStatus::RenderFailed;
    return job;
  }

  if (page_results.empty()) { job.status = PdfJobStatus::EmptyPdf; return job; }

  if (decode_failures > 0) {
    job.status = PdfJobStatus::DecodeFailed;
    job.decode_failures = decode_failures;
    job.num_pages = static_cast<int>(page_results.size());
    return job;
  }

  if (page_failures > 0) {
    job.status = PdfJobStatus::PageFailed;
    job.page_failures = page_failures;
    job.num_pages = static_cast<int>(page_results.size());
    return job;
  }

  job.num_pages = static_cast<int>(page_results.size());
  job.pages = std::move(page_results);
  return job;
}

} // namespace turbo_ocr::pipeline
