#include "turbo_ocr/routes/image_routes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/logger.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/server/env_utils.h"

using turbo_ocr::OCRResultItem;
using turbo_ocr::base64_decode;
using turbo_ocr::results_to_json;
using turbo_ocr::emit_results_json;
using turbo_ocr::decode::NvJpegDecoder;

namespace turbo_ocr::routes {

namespace {

// --- /ocr/raw: GPU-direct JPEG decode, Wuffs PNG ---
void register_ocr_raw_route_gpu(server::WorkPool &pool,
                                 pipeline::PipelineDispatcher &dispatcher,
                                 const server::ImageDecoder &decode,
                                 bool nvjpeg_available,
                                 bool layout_available) {
  drogon::app().registerHandler(
      "/ocr/raw",
      [&pool, &dispatcher, &decode, nvjpeg_available, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    if (req->body().empty()) {
      callback(server::error_response(drogon::k400BadRequest, "EMPTY_BODY", "Empty body"));
      return;
    }

    server::InferOptions opts;
    if (auto r = server::parse_query_options(req, layout_available, &opts);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }

    server::submit_work(pool, std::move(callback),
        [req, &dispatcher, &decode, nvjpeg_available, opts](server::DrogonCallback &cb) {
      server::run_with_error_handling(cb, "/ocr/raw", [&] {
        const auto *data = reinterpret_cast<const unsigned char *>(req->body().data());
        size_t len = req->body().size();

        // Configurable cap (MAX_IMAGE_DIM, default 16384, bounds [64, 65535]).
        // Same env var as /ocr/pixels — single knob across all image routes.
        const int kMaxImageDim = decode::max_image_dim();
        auto reject_too_large = [&](int w, int h) {
          cb(server::error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE",
              std::format("Image dimensions {}x{} exceed maximum of {}x{}",
                          w, h, kMaxImageDim, kMaxImageDim)));
        };

        // Pre-decode header sniff (PNG IHDR / JPEG SOFn). Refuses
        // decompression bombs without ever calling the decoder.
        if (auto d = turbo_ocr::decode::peek_image_dimensions(data, len)) {
          if (d->width > kMaxImageDim || d->height > kMaxImageDim) {
            reject_too_large(d->width, d->height);
            return;
          }
        }

        // JPEG with nvJPEG: submit GPU-direct decode + infer as one work item
        if (nvjpeg_available && NvJpegDecoder::is_jpeg(data, len)) {
          auto out = dispatcher.submit([data, len, opts, kMaxImageDim](auto &e) {
            auto &nvjpeg = e.get_nvjpeg();
            auto [w, h] = nvjpeg.get_dimensions(data, len);
            // Decompression-bomb guard for JPEGs the pre-decode header sniff
            // couldn't parse (progressive / unusual SOF markers): nvJPEG's
            // own dims are authoritative — reject before allocating GPU
            // memory for them.
            if (w > kMaxImageDim || h > kMaxImageDim)
              throw turbo_ocr::ImageTooLargeError(std::format(
                  "Image dimensions {}x{} exceed maximum of {}x{}",
                  w, h, kMaxImageDim, kMaxImageDim));
            if (w > 0 && h > 0) {
              auto [d_buf, pitch] = e.pipeline->ensure_gpu_buf(h, w);
              if (nvjpeg.decode_to_gpu(data, len, d_buf, pitch, w, h, e.stream)) {
                turbo_ocr::GpuImage gpu_img{.data = d_buf, .step = pitch, .rows = h, .cols = w};
                try {
                  return e.pipeline->run_with_layout(gpu_img, e.stream,
                                                     opts.want_layout,
                                                     opts.want_reading_order);
                } catch (const std::exception &) {}
              }
            }
            cv::Mat img = nvjpeg.decode(data, len);
            if (img.empty()) {
              if (len <= static_cast<size_t>(INT_MAX))
                img = cv::imdecode(
                    cv::Mat(1, static_cast<int>(len), CV_8UC1,
                            const_cast<unsigned char *>(data)),
                    cv::IMREAD_COLOR);
            }
            if (img.empty())
              throw turbo_ocr::ImageDecodeError("Failed to decode JPEG");
            // CPU-fallback bomb guard: nvjpeg.get_dimensions returns {0,0}
            // for headers it can't parse (so the dims check above was a
            // no-op), and the sniff is 64KB-bounded — re-check the decoded
            // size, same post-decode net as the non-JPEG branch.
            if (img.cols > kMaxImageDim || img.rows > kMaxImageDim)
              throw turbo_ocr::ImageTooLargeError(std::format(
                  "Image dimensions {}x{} exceed maximum of {}x{}",
                  img.cols, img.rows, kMaxImageDim, kMaxImageDim));
            return e.pipeline->run_with_layout(img, e.stream,
                                               opts.want_layout,
                                               opts.want_reading_order);
          }).get();
          cb(server::json_response(emit_results_json(out.results, out.layout, out.reading_order, opts.want_blocks)));
          return;
        }

        // Non-JPEG (PNG, etc.) or nvJPEG not available
        cv::Mat img = decode(data, len);
        if (img.empty()) {
          cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
          return;
        }
        // Post-decode safety net for formats we didn't sniff (BMP/TIFF/WEBP).
        if (img.cols > kMaxImageDim || img.rows > kMaxImageDim) {
          reject_too_large(img.cols, img.rows);
          return;
        }

        auto out = dispatcher.submit([&img, opts](auto &e) {
          return e.pipeline->run_with_layout(img, e.stream,
                                              opts.want_layout,
                                              opts.want_reading_order);
        }).get();
        cb(server::json_response(emit_results_json(out.results, out.layout, out.reading_order, opts.want_blocks)));
      });
    });
  }, {drogon::Post});
}

// --- /ocr/batch helpers ----------------------------------------------------
//
// The batch flow goes: base64 decode → pre-decode dim sniff → image decode
// (nvJPEG batch path for JPEGs when available, fallback per-image decoder
// otherwise) → post-decode dim guard → pipeline submit. Each stage tags
// per-slot errors so the response keeps a 1:1 mapping with the input array.

// One slot of the batch response: the full per-page pipeline result
// (text + layout + tables/formulas + reading order when requested).
struct BatchItem {
  pipeline::OcrPipelineResult out;
};

// Stage 1: base64 decode every input slot. Empty inputs short-circuit to
// "empty" so we don't pay decode cost on slots the caller never filled in.
void batch_decode_base64(const std::vector<std::string> &b64_strings,
                          std::vector<std::string> &raw_bytes,
                          std::vector<std::string> &errors) {
  size_t n = b64_strings.size();
  for (size_t i = 0; i < n; ++i) {
    const auto &b64 = b64_strings[i];
    if (b64.empty()) {
      errors[i] = "empty";
      continue;
    }
    raw_bytes[i] = base64_decode(b64);
    if (raw_bytes[i].empty()) errors[i] = "base64_decode_failed";
  }
}

// Stage 2: header-sniff (PNG/JPEG) every still-valid slot and reject
// oversized inputs before paying decode cost. Same MAX_IMAGE_DIM env as
// /ocr/raw and /ocr/pixels — but errors are per-slot, not whole-request 400s.
void batch_check_dims_pre(const std::vector<std::string> &raw_bytes,
                           int max_image_dim,
                           std::vector<std::string> &errors) {
  size_t n = raw_bytes.size();
  for (size_t i = 0; i < n; ++i) {
    if (!errors[i].empty()) continue;
    const auto &raw = raw_bytes[i];
    if (auto d = turbo_ocr::decode::peek_image_dimensions(
            reinterpret_cast<const unsigned char *>(raw.data()), raw.size())) {
      if (d->width > max_image_dim || d->height > max_image_dim) {
        errors[i] = std::format("dimensions_too_large ({}x{} > {}x{})",
                                 d->width, d->height,
                                 max_image_dim, max_image_dim);
      }
    }
  }
}

// Stage 3: decode pixels. JPEGs go through nvJPEG batch decode (when
// available and we have ≥2 of them); everything else runs through the
// per-image fallback decoder.
void batch_decode_images(const std::vector<std::string> &raw_bytes,
                          bool nvjpeg_available,
                          const server::ImageDecoder &decode,
                          std::vector<cv::Mat> &imgs,
                          std::vector<std::string> &errors) {
  size_t n = raw_bytes.size();
  std::vector<size_t> jpeg_indices;
  std::vector<std::pair<const unsigned char *, size_t>> jpeg_buffers;

  thread_local NvJpegDecoder tl_nvjpeg;
  const int kMaxImageDim = decode::max_image_dim();
  if (nvjpeg_available) {
    for (size_t i = 0; i < n; ++i) {
      if (!errors[i].empty()) continue;
      const auto &raw = raw_bytes[i];
      if (raw.size() >= 2 &&
          static_cast<unsigned char>(raw[0]) == 0xFF &&
          static_cast<unsigned char>(raw[1]) == 0xD8) {
        // Bomb guard: nvJPEG's own header dims are authoritative — reject
        // oversized JPEGs per-slot BEFORE batch_decode allocates a host Mat
        // from them (the pre-decode sniff is only 64KB-bounded).
        const auto *p = reinterpret_cast<const unsigned char *>(raw.data());
        auto [jw, jh] = tl_nvjpeg.get_dimensions(p, raw.size());
        if (jw > kMaxImageDim || jh > kMaxImageDim) {
          errors[i] = "dimensions_too_large";
          continue;
        }
        jpeg_indices.push_back(i);
        jpeg_buffers.emplace_back(p, raw.size());
      }
    }
  }

  if (jpeg_buffers.size() >= 2) {
    auto batch_mats = tl_nvjpeg.batch_decode(jpeg_buffers);
    for (size_t j = 0; j < jpeg_indices.size(); ++j)
      imgs[jpeg_indices[j]] = std::move(batch_mats[j]);
  }

  for (size_t i = 0; i < n; ++i) {
    if (!errors[i].empty()) continue;
    if (!imgs[i].empty()) continue;
    const auto &raw = raw_bytes[i];
    imgs[i] = decode(
        reinterpret_cast<const unsigned char *>(raw.data()), raw.size());
    if (imgs[i].empty()) errors[i] = "decode_failed";
  }
}

// Stage 4: post-decode safety net for formats we didn't header-sniff
// (BMP/TIFF/GIF). Releases the image so it doesn't get fed to the pipeline.
void batch_check_dims_post(std::vector<cv::Mat> &imgs,
                            int max_image_dim,
                            std::vector<std::string> &errors) {
  size_t n = imgs.size();
  for (size_t i = 0; i < n; ++i) {
    if (!errors[i].empty()) continue;
    if (imgs[i].cols > max_image_dim || imgs[i].rows > max_image_dim) {
      errors[i] = std::format("dimensions_too_large ({}x{} > {}x{})",
                               imgs[i].cols, imgs[i].rows,
                               max_image_dim, max_image_dim);
      imgs[i].release();
    }
  }
}

// Stage 5: run the pipeline on every slot that survived decode. Chunks
// into kMaxBatch-sized batches through run_batch_with_layout — both with
// and without layout the batched path applies (batched det/rec; in
// pdf_only mode layout is one batched TRT execute per chunk too), so
// ?layout=1 no longer falls back to serial single-image inference.
void batch_run_pipeline(pipeline::PipelineDispatcher &dispatcher,
                         std::vector<cv::Mat> &valid_imgs,
                         const std::vector<size_t> &valid_indices,
                         bool want_layout,
                         const server::InferOptions &opts,
                         std::vector<BatchItem> &all_items,
                         std::vector<std::string> &errors) {
  if (valid_imgs.empty()) return;

  constexpr int kMaxBatch = 8;
  try {
    dispatcher.submit([&](auto &e) {
      for (size_t offset = 0; offset < valid_imgs.size(); offset += kMaxBatch) {
        size_t end = std::min(offset + kMaxBatch, valid_imgs.size());
        // Per-chunk try wraps chunk construction too: a failure here taints
        // ONLY this chunk's slots, never the completed results of earlier
        // chunks (whose errors[] are empty == success, the same sentinel a
        // blanket outer-catch would wrongly overwrite).
        try {
          std::vector<cv::Mat> chunk(
              std::make_move_iterator(valid_imgs.begin() + offset),
              std::make_move_iterator(valid_imgs.begin() + end));
          try {
            auto chunk_results = e.pipeline->run_batch_with_layout(
                chunk, e.stream, want_layout, opts.want_reading_order);
            for (size_t j = 0; j < chunk_results.size(); ++j)
              all_items[valid_indices[offset + j]].out =
                  std::move(chunk_results[j]);
          } catch (const std::exception &) {
            // One degenerate image (1×1 / corrupt-decoded Mat) poisons the
            // whole batched chunk. Retry its images individually through
            // run_with_layout, whose internal guard degrades exactly the
            // bad slot to an empty result and resets the stream — per-slot
            // isolation, same as the single-image routes.
            for (size_t j = 0; j < chunk.size(); ++j) {
              const size_t idx = valid_indices[offset + j];
              try {
                all_items[idx].out = e.pipeline->run_with_layout(
                    chunk[j], e.stream, want_layout, opts.want_reading_order);
              } catch (const turbo_ocr::PoolExhaustedError &) {
                throw;  // -> 503 at the route, never a per-slot tag
              } catch (const std::exception &ex) {
                // Don't leak raw CUDA_CHECK text (absolute source paths) to
                // clients; log the detail, tag the slot with a stable code.
                TOCR_LOG_ERROR_RL("batch slot inference error",
                                  "slot", idx, "error", std::string_view(ex.what()));
                errors[idx] = "inference_failed";
              }
            }
          }
        } catch (const turbo_ocr::PoolExhaustedError &) {
          throw;  // queue full mid-batch -> 503, don't swallow as per-slot
        } catch (const std::exception &ex) {
          // chunk construction (e.g. bad_alloc) failed: tag only THIS
          // chunk's still-empty slots, leaving completed chunks intact.
          TOCR_LOG_ERROR_RL("batch chunk error", "offset", offset,
                            "error", std::string_view(ex.what()));
          for (size_t j = offset; j < end; ++j)
            if (errors[valid_indices[j]].empty())
              errors[valid_indices[j]] = "inference_failed";
        }
      }
    }).get();
  } catch (const turbo_ocr::PoolExhaustedError &) {
    // GPU queue full at submit — no chunk ran. Propagate so the route's
    // run_with_error_handling turns it into 503 SERVER_BUSY, matching every
    // other inference route instead of a 200 with a fully-errored array.
    throw;
  } catch (const std::exception &ex) {
    // Submission-level failure before any chunk ran — tag every still-empty
    // slot so the caller knows their request didn't silently succeed (stable
    // code, detail to the log, no raw internal text to the client).
    TOCR_LOG_ERROR_RL("batch submission error", "error", std::string_view(ex.what()));
    for (size_t k : valid_indices)
      if (errors[k].empty()) errors[k] = "inference_failed";
  }
}

// Stage 6: serialize {batch_results, errors} JSON. Mirrors the CPU contract
// — null in the errors array for successful slots, an error string otherwise.
std::string batch_emit_json(std::vector<BatchItem> &all_items,
                             const std::vector<std::string> &errors,
                             bool want_layout,
                             bool want_blocks) {
  size_t n = all_items.size();
  std::string json_str;
  json_str.reserve(n * 1024);
  json_str += "{\"batch_results\":[";
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) json_str += ',';
    if (want_layout) {
      // Full per-page emitter: text + layout (+ tables/formulas when the
      // CUA router fired, byte-identical to the legacy shape otherwise).
      json_str += emit_pipeline_result_json(all_items[i].out, want_blocks);
    } else {
      json_str += results_to_json(all_items[i].out.results);
    }
  }
  json_str += "],\"errors\":[";
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) json_str += ',';
    const auto &e = errors[i];
    if (e.empty()) {
      json_str += "null";
    } else {
      json_str += '"';
      for (char c : e) {
        if (c == '"' || c == '\\') json_str += '\\';
        json_str += c;
      }
      json_str += '"';
    }
  }
  json_str += "]}";
  return json_str;
}

void register_ocr_batch_route_gpu(server::WorkPool &pool,
                                   pipeline::PipelineDispatcher &dispatcher,
                                   const server::ImageDecoder &decode,
                                   bool nvjpeg_available,
                                   bool layout_available,
                                   int max_batch_images) {
  drogon::app().registerHandler(
      "/ocr/batch",
      [&pool, &dispatcher, &decode, nvjpeg_available, layout_available,
       max_batch_images](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    server::InferOptions opts;
    if (auto r = server::parse_query_options(req, layout_available, &opts);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }

    auto json = req->getJsonObject();
    if (!json) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Invalid JSON"));
      return;
    }
    if (!json->isMember("images") || !(*json)["images"].isArray()) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Missing images array"));
      return;
    }

    auto &images_json = (*json)["images"];
    size_t n = images_json.size();
    if (n == 0) {
      callback(server::error_response(drogon::k400BadRequest, "EMPTY_BATCH", "Empty images array"));
      return;
    }
    // Cap before the O(n) per-slot allocations + n*1KB JSON reserve below —
    // an unbounded images[] (tens of millions of tiny strings in a <100MB
    // body) is otherwise a memory-amplification OOM lever.
    if (n > static_cast<size_t>(max_batch_images)) {
      callback(server::error_response(drogon::k400BadRequest, "BATCH_TOO_LARGE",
          std::format("images array has {} entries, max is {}", n, max_batch_images)));
      return;
    }

    // asString() throws Json::LogicError on a non-string element ({}/[]);
    // guard the type so a malformed slot becomes an empty string the per-slot
    // decode path tags, instead of an unhandled exception / 500.
    auto b64_strings = std::make_shared<std::vector<std::string>>(n);
    for (size_t i = 0; i < n; ++i) {
      const auto &el = images_json[static_cast<int>(i)];
      if (el.isString())
        (*b64_strings)[i] = el.asString();
    }

    server::submit_work(pool, std::move(callback),
        [b64_strings, n, &dispatcher, &decode, nvjpeg_available, opts](server::DrogonCallback &cb) {
      const bool want_layout = opts.want_layout;
      server::run_with_error_handling(cb, "/ocr/batch", [&] {
        // Per-slot error tags for the response. Empty string == success.
        // Mirrors the CPU /ocr/batch contract: callers see which slots
        // failed (empty / decode_failed / dimensions_too_large) instead
        // of getting a silently-zero result list.
        std::vector<std::string> errors(n);
        std::vector<std::string> raw_bytes(n);

        batch_decode_base64(*b64_strings, raw_bytes, errors);

        const int kMaxImageDim = decode::max_image_dim();
        batch_check_dims_pre(raw_bytes, kMaxImageDim, errors);

        std::vector<cv::Mat> imgs(n);
        batch_decode_images(raw_bytes, nvjpeg_available, decode, imgs, errors);
        batch_check_dims_post(imgs, kMaxImageDim, errors);

        std::vector<cv::Mat> valid_imgs;
        std::vector<size_t> valid_indices;
        valid_imgs.reserve(n);
        valid_indices.reserve(n);
        for (size_t i = 0; i < n; ++i) {
          if (errors[i].empty() && !imgs[i].empty()) {
            valid_imgs.push_back(std::move(imgs[i]));
            valid_indices.push_back(i);
          }
        }

        std::vector<BatchItem> all_items(n);
        batch_run_pipeline(dispatcher, valid_imgs, valid_indices,
                            want_layout, opts, all_items, errors);

        cb(server::json_response(batch_emit_json(all_items, errors, want_layout, opts.want_blocks)));
      });
    });
  }, {drogon::Post});
}

// --- /ocr/pixels: raw BGR pixel data, zero decode overhead ---
void register_ocr_pixels_route_gpu(server::WorkPool &pool,
                                    pipeline::PipelineDispatcher &dispatcher,
                                    bool layout_available) {
  drogon::app().registerHandler(
      "/ocr/pixels",
      [&pool, &dispatcher, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    server::InferOptions opts;
    if (auto r = server::parse_query_options(req, layout_available, &opts);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }

    auto w_str = req->getHeader("X-Width");
    auto h_str = req->getHeader("X-Height");
    auto c_str = req->getHeader("X-Channels");

    if (w_str.empty() || h_str.empty()) {
      callback(server::error_response(drogon::k400BadRequest, "MISSING_HEADER", "Missing X-Width or X-Height headers"));
      return;
    }

    int width, height, channels;
    try {
      width = std::stoi(w_str);
      height = std::stoi(h_str);
      channels = c_str.empty() ? 3 : std::stoi(c_str);
    } catch (const std::exception &) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_HEADER",
          "Invalid X-Width, X-Height, or X-Channels header value"));
      return;
    }

    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_DIMENSIONS", "Invalid dimensions or channels"));
      return;
    }

    // Configurable via MAX_IMAGE_DIM (default 16384). Read once on first
    // request and cached for the process lifetime.
    const int kMaxPixelDim = decode::max_image_dim();
    if (width > kMaxPixelDim || height > kMaxPixelDim) {
      callback(server::error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE",
          std::format("Dimensions {}x{} exceed maximum of {}x{}", width, height, kMaxPixelDim, kMaxPixelDim)));
      return;
    }

    size_t expected = static_cast<size_t>(width) * height * channels;
    if (req->body().size() != expected) {
      callback(server::error_response(drogon::k400BadRequest, "BODY_SIZE_MISMATCH",
          std::format("Body size mismatch: expected {} bytes ({}x{}x{}), got {}",
                      expected, width, height, channels, req->body().size())));
      return;
    }

    server::submit_work(pool, std::move(callback),
        [req, &dispatcher, width, height, channels, opts](server::DrogonCallback &cb) {
      server::run_with_error_handling(cb, "/ocr/pixels", [&] {
        cv::Mat img(height, width, channels == 3 ? CV_8UC3 : CV_8UC1,
                    const_cast<char *>(req->body().data()));
        // The pipeline is BGR-only: a 1-channel Mat reaches the GPU upload
        // with step == cols, trips the degenerate-input guard, and comes
        // back empty. Expand grayscale to BGR up front.
        if (channels == 1)
          cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

        auto out = dispatcher.submit([&img, opts](auto &e) {
          return e.pipeline->run_with_layout(img, e.stream,
                                              opts.want_layout,
                                              opts.want_reading_order);
        }).get();
        cb(server::json_response(emit_results_json(out.results, out.layout, out.reading_order, opts.want_blocks)));
      });
    });
  }, {drogon::Post});
}

} // namespace

void register_image_routes(server::WorkPool &pool,
                           pipeline::PipelineDispatcher &dispatcher,
                           const server::ImageDecoder &decode,
                           bool nvjpeg_available,
                           bool layout_available,
                           int max_batch_images) {
  register_ocr_raw_route_gpu(pool, dispatcher, decode, nvjpeg_available, layout_available);
  register_ocr_batch_route_gpu(pool, dispatcher, decode, nvjpeg_available, layout_available,
                               max_batch_images);
  register_ocr_pixels_route_gpu(pool, dispatcher, layout_available);
}

} // namespace turbo_ocr::routes
