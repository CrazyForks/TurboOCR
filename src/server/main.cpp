#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "turbo_ocr/common/logger.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <drogon/HttpAppFramework.h>

#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/engine/onnx_to_trt.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/grpc_service.h"
#include "turbo_ocr/server/language_paths.h"
#include "turbo_ocr/server/metrics.h"
#include "turbo_ocr/server/server_bootstrap.h"
#include "turbo_ocr/server/server_config.h"
#include "turbo_ocr/server/server_types.h"
#include "turbo_ocr/server/work_pool.h"
#include "turbo_ocr/routes/common_routes.h"
#include "turbo_ocr/routes/image_routes.h"
#include "turbo_ocr/routes/pdf_routes.h"

using turbo_ocr::decode::FastPngDecoder;
using turbo_ocr::decode::NvJpegDecoder;
using turbo_ocr::render::PdfRenderer;

namespace bootstrap = turbo_ocr::server::bootstrap;

int main(int argc, char **argv) {
  const auto cfg = turbo_ocr::server::ServerConfig::load_or_die(argc, argv);
  cfg.log_effective();
  bootstrap::g_shutdown_grace_seconds.store(cfg.shutdown_grace_seconds,
                                            std::memory_order_release);

  // Parallelism lives at the request level (work-pool threads); OpenCV's
  // per-op thread pool on top of that just oversubscribes cores under load.
  cv::setNumThreads(0);

  const auto &rec_paths = cfg.rec_paths;
  if (!cfg.selected_model_name.empty())
    TOCR_LOG_INFO("OCR model selected",
                  "model", std::string_view(cfg.selected_model_name),
                  "det",   std::string_view(cfg.det_onnx),
                  "rec",   std::string_view(rec_paths.rec),
                  "dict",  std::string_view(rec_paths.dict));
  auto rec_dict = rec_paths.dict;

  // Validate model paths up front so a missing models/ tree fails fast with
  // a clear error rather than tripping a confusing CUDA/TRT error deep in
  // pipeline construction. ensure_trt_engine() returns "" on missing ONNX,
  // which the dispatcher only notices much later.
  auto require_model = [](const std::string &path, const char *purpose) {
    bootstrap::require_model(path, purpose, "model", "_ONNX");
  };
  require_model(cfg.det_onnx, "DET");
  require_model(rec_paths.rec, "REC");
  if (!cfg.pdf_only)
    require_model(cfg.cls_onnx, "CLS");

  // Build the PdfRenderer here — AFTER the fail-fast model validation above
  // (so a missing model std::exit(1)s without orphaning forked daemons) but
  // BEFORE any CUDA/TRT call below. The renderer fork()s a pool of fastpdf2png
  // daemons, and fork() in a process that has already initialized a CUDA
  // context is undefined behaviour (CUDA's driver threads, pinned allocations
  // and fds don't survive a fork cleanly). sweep_orphan_engine_temps /
  // ensure_trt_engine / cudaMemGetInfo below all touch the CUDA context, so the
  // daemons must be spawned ahead of them. The renderer depends on neither the
  // dispatcher nor any model path, so hoisting it above them is safe.
  const int pdf_daemons = cfg.pdf_daemons;
  const int pdf_workers = cfg.pdf_workers;
  PdfRenderer pdf_renderer(pdf_daemons, pdf_workers);
  TOCR_LOG_INFO("PDF renderer initialized", "daemons", pdf_daemons, "workers", pdf_workers);

  // Auto-build TRT engines from ONNX (cached by TRT version + model hash)
  // Sweep orphan .trt.tmp.* files left by previous crashed processes; safe
  // because in-progress builds by sibling replicas are protected by the
  // 60-second min-age window inside the sweeper.
  turbo_ocr::engine::sweep_orphan_engine_temps();
  // PDF-only hybrid mode: build DET with a STATIC shape (min==opt==max at
  // {pdf_batch, 3, pdf_page_h, pdf_page_w}) so TRT specializes tactics for
  // the one shape every rendered page has. The "det_pdf_static" type makes
  // onnx_to_trt emit that profile and bake the dims into the cache key.
  // REC stays the DYNAMIC 5-bucket engine: benchmarks showed a single-width
  // static rec squishes wide text lines and drops recall to 70-80% on
  // multi-column/dense pages (vs 98% dynamic) for only ~10-20% speed.
  // CLS is skipped entirely (rendered PDF pages are upright by
  // construction), saving its build + per-request pass.
  const char *det_type = cfg.pdf_only ? "det_pdf_static" : "det";
  auto det_model = turbo_ocr::engine::ensure_trt_engine(cfg.det_onnx, det_type);
  auto rec_model = turbo_ocr::engine::ensure_trt_engine(rec_paths.rec, "rec");
  std::string cls_model;
  if (!cfg.pdf_only) {
    cls_model = turbo_ocr::engine::ensure_trt_engine(cfg.cls_onnx, "cls");
  } else {
    TOCR_LOG_INFO("PDF-only hybrid: static batched det + dynamic 5-bucket rec, cls skipped",
                  "dpi",       cfg.pdf_dpi,
                  "page_h",    cfg.pdf_page_h,
                  "page_w",    cfg.pdf_page_w,
                  "batch",     cfg.pdf_batch);
  }
  if (cfg.disable_angle_cls) {
    cls_model.clear();
    TOCR_LOG_INFO("Angle classification disabled via DISABLE_ANGLE_CLS=1");
  }

  // Optional PP-DocLayoutV3 stage. ON by default — users can disable with
  // DISABLE_LAYOUT=1 to save ~300-500 MB VRAM.
  std::string layout_model;
  if (!cfg.layout_disabled) {
    if (cfg.layout_trt && !cfg.layout_trt->empty()) {
      layout_model = *cfg.layout_trt;
      TOCR_LOG_INFO("Layout detection enabled", "engine", std::string_view(layout_model));
    } else {
      // Layout ONNX is optional — soft-disable with a warning if missing
      // rather than hard-failing, so installs without layout still serve.
      layout_model = turbo_ocr::engine::ensure_trt_engine(cfg.layout_onnx, "layout");
      if (layout_model.empty()) {
        TOCR_LOG_WARN("Layout model (layout.onnx) not found; layout stage will be disabled");
      } else {
        TOCR_LOG_INFO("Layout detection enabled");
      }
    }
  } else {
    TOCR_LOG_INFO("Layout detection disabled (set DISABLE_LAYOUT=0 to enable)");
  }

  // PDF extraction mode default
  const turbo_ocr::pdf::PdfMode default_pdf_mode = cfg.default_pdf_mode;
  if (cfg.default_pdf_mode_was_set) {
    TOCR_LOG_INFO("PDF extraction default mode configured", "mode", turbo_ocr::pdf::mode_name(default_pdf_mode));
  } else {
    TOCR_LOG_INFO("PDF extraction default mode: ocr (override per-request with /ocr/pdf?mode=<geometric|auto|auto_verified>)");
  }
  turbo_ocr::pdf::ensure_pdfium_initialized();

  // Pipeline pool. Throughput is GPU-compute-bound on this stack: a clean
  // pool sweep (RTX 5090, FUNSD) plateaus by ~5 pipelines (5→278, 8→263,
  // 10→258 img/s) because the det/rec kernels already saturate the GPU and
  // extra pipelines only add scheduling/cache pressure. So the ladder caps at
  // 5 deliberately — raising it costs VRAM for negative throughput. Explicit
  // --pool-size overrides.
  int pool_size = 4;
  if (cfg.pipeline_pool_size) {
    pool_size = *cfg.pipeline_pool_size;
  } else {
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
      int vram_gb = static_cast<int>(total_mem >> 30);
      if (vram_gb >= 14) pool_size = 5;
      else if (vram_gb >= 12) pool_size = 3;
      else if (vram_gb >= 8)  pool_size = 2;
      else                     pool_size = 1;

      // Footprint-based safety floor: the tier above keys off TOTAL VRAM, but a
      // card that *reports* 16 GB while another process already holds most of
      // it would OOM during warmup. Estimate each pipeline's resident footprint
      // (engines + activation/workspace buffers, generous to stay conservative)
      // and reduce the tier so it fits in the FREE VRAM measured right now.
      // This only ever LOWERS the count — never raises it above the tier cap —
      // and never below 1, so a healthy 32 GB card with the tiny model still
      // picks 5 (a few hundred MB per pipeline against ~31 GB free). The
      // explicit --pool-size / PIPELINE_POOL_SIZE override above bypasses this
      // entirely and always wins.
      constexpr size_t kPerPipelineFootprintBytes = size_t{2} << 30;  // 2 GiB
      // Leave a 1 GiB headroom so the renderer daemons / CUDA context / OS
      // don't get squeezed to the byte.
      constexpr size_t kVramHeadroomBytes = size_t{1} << 30;
      size_t budget = free_mem > kVramHeadroomBytes ? free_mem - kVramHeadroomBytes : 0;
      int fits = static_cast<int>(budget / kPerPipelineFootprintBytes);
      if (fits < 1) fits = 1;
      if (fits < pool_size) {
        TOCR_LOG_WARN("Reducing pipeline pool to fit available VRAM",
                      "tier_pool_size", pool_size, "footprint_capped", fits,
                      "free_mem_gb", static_cast<int>(free_mem >> 30));
        pool_size = fits;
      }
      TOCR_LOG_INFO("Auto-detected pipeline pool size", "pool_size", pool_size, "vram_gb", vram_gb);
    }
  }

  // Document-orientation engine (optional) — powers /ocr/pdf?autorotate=1.
  // Soft-disable if the model is absent: ensure_trt_engine returns "" and
  // autorotate requests are then rejected with AUTOROTATE_DISABLED.
  std::string doc_ori_model =
      turbo_ocr::engine::ensure_trt_engine(cfg.doc_ori_onnx, "doc_ori");
  if (doc_ori_model.empty())
    TOCR_LOG_WARN("Doc-orientation model (doc_ori.onnx) not found; autorotate disabled");
  else
    TOCR_LOG_INFO("Doc-orientation (autorotate) enabled");

  auto dispatcher = turbo_ocr::pipeline::make_pipeline_dispatcher(
      pool_size, det_model, rec_model, rec_dict, cls_model, layout_model,
      cfg.pdf_only ? cfg.pdf_batch  : 0,
      cfg.pdf_only ? cfg.pdf_page_h : 0,
      cfg.pdf_only ? cfg.pdf_page_w : 0,
      doc_ori_model, cfg.det_cfg);
  // Some pipelines may have failed to init (logged); key downstream sizing
  // off the count actually built.
  pool_size = static_cast<int>(dispatcher->worker_count());

  // Per-request inference deadline (C4). 0 leaves the dispatcher in its
  // legacy unbounded-wait mode; cfg.request_timeout_ms defaults to 30s, which
  // is per-image/per-page so it only trips a genuinely wedged worker, never a
  // normal request. A timed-out future is ABANDONED, so every GPU submit below
  // captures its request inputs BY VALUE (see the infer lambda).
  dispatcher->set_request_timeout_ms(cfg.request_timeout_ms);

  // nvJPEG
  TOCR_LOG_INFO("Initializing nvJPEG decoders");
  thread_local NvJpegDecoder tl_nvjpeg;
  bool nvjpeg_available = tl_nvjpeg.available();
  if (nvjpeg_available)
    TOCR_LOG_INFO("nvJPEG GPU-accelerated JPEG decode enabled");
  else
    TOCR_LOG_WARN("nvJPEG not available, using OpenCV JPEG decode");

  // Image decoder: JPEG via nvJPEG (GPU), PNG via Wuffs (fast path), every
  // other format (WebP, BMP, TIFF, GIF, …) via cv::imdecode. OpenCV's
  // imgcodecs is linked to libwebp/libtiff so cv::imdecode covers the rest.
  turbo_ocr::server::ImageDecoder decode =
      [nvjpeg_available](const unsigned char *data, size_t len) -> cv::Mat {
    auto opencv_decode = [&]() -> cv::Mat {
      if (len > static_cast<size_t>(INT_MAX)) return {};
      return cv::imdecode(
          cv::Mat(1, static_cast<int>(len), CV_8UC1,
                  const_cast<unsigned char *>(data)),
          cv::IMREAD_COLOR);
    };
    if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
      if (nvjpeg_available) {
        cv::Mat img = tl_nvjpeg.decode(data, len);
        if (!img.empty()) return img;
      }
      return opencv_decode();
    }
    if (FastPngDecoder::is_png(data, len))
      return FastPngDecoder::decode(data, len);
    return opencv_decode();
  };

  // Inference function for shared routes (/ocr base64)
  const bool layout_available = !layout_model.empty();
  turbo_ocr::server::InferFunc infer =
      [&dispatcher](const cv::Mat &img,
                    const turbo_ocr::server::InferOptions &opts)
          -> turbo_ocr::server::InferResult {
    // submit_for_default may ABANDON the task on timeout, so the lambda owns
    // its inputs BY VALUE: img is a cheap cv::Mat refcount bump, the flags are
    // bools. The dispatcher itself is long-lived and captured by reference.
    // On deadline this throws turbo_ocr::TimeoutError; the route's
    // run_with_error_handling maps it to HTTP 504 (INFERENCE_TIMEOUT).
    const bool want_layout = opts.want_layout;
    const bool want_reading_order = opts.want_reading_order;
    auto out = dispatcher->submit_for_default(
        [img, want_layout, want_reading_order](auto &e) {
          return e.pipeline->run_with_layout(img, e.stream, want_layout,
                                             want_reading_order);
        });
    return turbo_ocr::server::InferResult{
        .results       = std::move(out.results),
        .layout        = std::move(out.layout),
        .reading_order = std::move(out.reading_order),
    };
  };

  // Work pool for offloading blocking inference from Drogon event loop
  int work_threads = std::max(pool_size * 32, 128);
  if (cfg.http_threads) work_threads = *cfg.http_threads;
  turbo_ocr::server::WorkPool work_pool(work_threads);

  // --- Register all routes ---
  turbo_ocr::server::Metrics::instance().set_pool_size(pool_size);
  turbo_ocr::server::register_observability_middleware();
  turbo_ocr::server::register_metrics_route(
      &work_pool, [d = dispatcher.get()] { return d->queue_depth(); });
  // Single readiness probe shared by HTTP /health/ready and gRPC Health
  // so k8s probes behave identically across protocols.
  //
  // The probe runs a tiny real inference (48x48 dummy) through the
  // pipeline so a corrupt TRT engine, missing layout, or wedged stream
  // surfaces here instead of on the first real client request. To keep
  // probe traffic from eating GPU, we cache the result for 5 seconds —
  // probes within that window get the cached verdict without GPU work.
  struct ProbeState {
    std::mutex mu;
    std::atomic<int64_t> last_check_ms{0};
    std::atomic<bool> ok{false};
  };
  auto probe = std::make_shared<ProbeState>();
  auto readiness = [&dispatcher, probe, layout_available]() -> bool {
    using namespace std::chrono;
    const int64_t now_ms = duration_cast<milliseconds>(
        steady_clock::now().time_since_epoch()).count();
    if (now_ms - probe->last_check_ms.load(std::memory_order_acquire) < 5000)
      return probe->ok.load(std::memory_order_acquire);

    std::lock_guard lock(probe->mu);
    // Recheck under lock — another thread may have refreshed the cache
    // while we waited. Avoids stampedes during probe spikes.
    if (now_ms - probe->last_check_ms.load(std::memory_order_acquire) < 5000)
      return probe->ok.load(std::memory_order_acquire);

    bool ok = false;
    try {
      // submit_for_default honours the request timeout, so a wedged worker
      // flips readiness to not-ready (caught below) instead of blocking this
      // probe thread forever. The dummy Mat is created inside the task, so
      // there is nothing request-scoped to abandon on timeout.
      dispatcher->submit_for_default([layout_available](auto &e) {
        cv::Mat dummy(48, 48, CV_8UC3, cv::Scalar(255, 255, 255));
        (void)e.pipeline->run_with_layout(dummy, e.stream,
                                          /*want_layout=*/layout_available,
                                          /*want_reading_order=*/false);
      });
      ok = true;
    } catch (...) {}
    probe->ok.store(ok, std::memory_order_release);
    probe->last_check_ms.store(now_ms, std::memory_order_release);
    return ok;
  };
  turbo_ocr::routes::register_health_route(readiness, &work_pool);
  turbo_ocr::routes::register_ocr_base64_route(work_pool, infer, decode, layout_available);
  turbo_ocr::routes::register_image_routes(work_pool, *dispatcher, decode, nvjpeg_available, layout_available,
                                           cfg.max_batch_images);
  turbo_ocr::routes::register_pdf_route(work_pool, *dispatcher, pdf_renderer, default_pdf_mode, layout_available,
                                        cfg.pdf_only ? cfg.pdf_batch : 0,
                                        cfg.pdf_only ? cfg.pdf_dpi : 100,
                                        cfg.max_pdf_pages,
                                        /*doc_ori_available=*/!doc_ori_model.empty());

  // GET /capabilities (M6) — advertise this build's honored feature set so a
  // client can discover the known GPU/CPU divergences without trial requests.
  // GPU honors auto_verified as its own path and has no /profile endpoint.
  {
    turbo_ocr::routes::CapabilitiesInfo caps;
    caps.is_gpu              = true;
    caps.layout_available    = layout_available;
    caps.autorotate_available = !doc_ori_model.empty();
    caps.profile_endpoint    = false;
    caps.grpc_response_mode  =
        std::string(turbo_ocr::server::detail::grpc_mode_str(cfg.grpc_response_mode));
    caps.honored_auto_verified = true;
    caps.pdf_default_dpi     = cfg.pdf_only ? cfg.pdf_dpi : 100;
    caps.max_pdf_pages       = cfg.max_pdf_pages;
    caps.max_body_mb         = cfg.max_body_mb;
    caps.max_image_dim       = cfg.max_image_dim;
    caps.max_batch_images    = cfg.max_batch_images;
    turbo_ocr::routes::register_capabilities_route(caps);
  }

  // Seed the readiness cache once now (the dispatcher just warmed up), so the
  // gRPC cached-only probe has a real verdict before the first HTTP probe runs.
  (void)readiness();

  // gRPC Health uses a CACHE-ONLY view of the readiness verdict (H2): it must
  // never run the GPU probe inline, because Health() executes on the gRPC CQ
  // poller thread and a fresh GPU pass there would stall every queued RPC on
  // that poller. It reads the last cached value the HTTP /health/ready probe
  // (above) refreshes; the cache TTL keeps it fresh under any real probe
  // traffic and we seeded it just above. This also keeps gRPC liveness/
  // readiness GPU-free (M2): a busy GPU can't make Health() block, so it can't
  // flap the process out of service.
  auto readiness_cached = [probe]() -> bool {
    return probe->ok.load(std::memory_order_acquire);
  };

  // gRPC
  auto grpc_handle = turbo_ocr::server::start_grpc_server(
      *dispatcher, cfg, &pdf_renderer, layout_available, readiness_cached);
  bootstrap::g_grpc_server_for_drain = grpc_handle.server.get();

  // HTTP (Drogon) — behind nginx (port 8000), direct access on 8080
  const int port = cfg.http_port;
  int io_threads = std::max(pool_size, 4);

  // MAX_BODY_MB caps the largest accepted upload (default 100). Same env
  // var the entrypoint uses to render the nginx body cap, so the two
  // layers always agree. MAX_BODY_MEMORY_MB controls how much of each
  // body Drogon buffers in memory before spilling to a temp file —
  // larger values cut /tmp I/O at the cost of letting RSS grow with
  // concurrent connections. Default 1024 MB (1 GiB) effectively keeps
  // every body in RAM until MAX_BODY_MB is raised above 1 GiB, since
  // the memory cap is clamped to the total cap below. Lower it on
  // memory-constrained hosts (e.g. MAX_BODY_MEMORY_MB=50 caps body
  // buffer RSS at ~50 MB × concurrent_requests).
  const int max_body_mb = cfg.max_body_mb;
  int max_body_mem_mb = cfg.max_body_mem_mb;
  if (max_body_mem_mb > max_body_mb) max_body_mem_mb = max_body_mb;

  TOCR_LOG_INFO("HTTP server starting", "port", port, "io_threads", io_threads,
           "work_threads", work_threads, "pool_size", dispatcher->worker_count(),
           "body_cap_mb_drogon", max_body_mb, "body_cap_mb_nginx", max_body_mb,
           "body_mem_mb", max_body_mem_mb);

  // Background readiness refresher: periodically re-runs the GPU probe (off the
  // gRPC CQ poller) so the cached verdict gRPC Health reads stays fresh even for
  // deployments that probe ONLY gRPC Health and never hit HTTP /health/ready.
  // The probe's own 5s TTL gates the real GPU work, so this just guarantees a
  // wake within that window. Stopped + joined before gRPC/dispatcher teardown.
  std::atomic<bool> refresher_stop{false};
  std::thread readiness_refresher([&readiness, &refresher_stop]() {
    while (!refresher_stop.load(std::memory_order_acquire)) {
      (void)readiness();
      for (int i = 0; i < 30 && !refresher_stop.load(std::memory_order_acquire); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });

  // Listener + body-size config + graceful-shutdown signal handlers + run().
  // Blocks until app().quit() (the drain thread) returns.
  bootstrap::run_http_server(cfg, io_threads, work_pool);

  refresher_stop.store(true, std::memory_order_release);
  readiness_refresher.join();
  bootstrap::shutdown_grpc_after_run(grpc_handle.server.get());
  return 0;
}
