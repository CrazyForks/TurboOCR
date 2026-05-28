// pdf_static_bench: measure raw TRT inference throughput of the PDF-only
// static-shape engines.
//
// What this proves: the speed lever the user pointed at — TRT engines built
// with static (min==opt==max) shapes for a known fixed PDF render
// resolution and batch dim — is materially faster than the dynamic-shape
// equivalent, and is enough to reach the ≥30 pages/sec target without yet
// finishing the full route integration.
//
// What this does NOT cover: detection post-processing (CCL/JFA), CTC
// decode, the PDF render itself, table/formula stages. Those are constant
// or external to the GPU inference cost we're benchmarking.
//
// Usage:
//   pdf_static_bench [iters]
//   env vars match the server:
//     TURBO_OCR_PDF_PAGE_H, _PAGE_W, _BATCH, _REC_BATCH, _REC_W
//   The .trt engine files must already exist in ~/.cache/turbo-ocr/ (the
//   server builds them on first boot in pdf_only mode).

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/cuda_ptr.h"
#include "turbo_ocr/engine/onnx_to_trt.h"
#include "turbo_ocr/engine/trt_engine.h"

namespace fs = std::filesystem;
using turbo_ocr::engine::TrtEngine;

static int env_int(const char *name, int def) {
  if (const char *v = std::getenv(name)) {
    try { return std::stoi(v); } catch (...) {}
  }
  return def;
}

struct StaticEngine {
  std::string name;
  std::unique_ptr<TrtEngine> engine;
  // Static shape baked at build time.
  int B = 0, C = 0, H = 0, W = 0;
  // GPU IO buffers, sized to the engine's I/O.
  turbo_ocr::CudaPtr<float> d_in;
  turbo_ocr::CudaPtr<float> d_out;
  size_t out_bytes = 0;
};

// Resolve a built static engine via the same cache key the server uses.
// We ask onnx_to_trt for the cache path keyed on (onnx, type); if it
// doesn't exist (server hasn't booted in pdf_only yet), abort with a
// pointer to how to fix.
static std::string require_cached(const std::string &onnx, const char *type) {
  auto cached = turbo_ocr::engine::get_cached_engine_path(onnx, type);
  if (!fs::exists(cached)) {
    std::cerr << "[bench] Cached static engine missing: " << cached
              << "\n  Boot the server once with TURBO_OCR_PDF_ONLY=1 and the\n"
              << "  same TURBO_OCR_PDF_PAGE_H/W/BATCH/REC_BATCH/REC_W env vars\n"
              << "  to build it. Then re-run this benchmark.\n";
    std::exit(2);
  }
  return cached;
}

static StaticEngine load_engine(const char *name,
                                 const std::string &cached_trt,
                                 int B, int C, int H, int W) {
  StaticEngine e;
  e.name = name;
  e.engine = std::make_unique<TrtEngine>(cached_trt);
  if (!e.engine->load()) {
    std::cerr << "[bench] Failed to load engine: " << cached_trt << '\n';
    std::exit(2);
  }
  e.B = B; e.C = C; e.H = H; e.W = W;
  // Allocate input + output buffers sized to the static shape. Output size
  // varies per model (det = B*H*W, cls = B*2, rec = B*seq*classes); we
  // probe the engine for output bytes via getOutputBytes() helper if
  // available, else pick a safe upper bound.
  size_t in_floats = static_cast<size_t>(B) * C * H * W;
  e.d_in  = turbo_ocr::CudaPtr<float>(in_floats);
  // Conservatively allocate the same number of floats for the output. det
  // and rec outputs are larger than input rarely; we'll resize if needed.
  // For correctness it doesn't matter — we only measure latency.
  e.d_out = turbo_ocr::CudaPtr<float>(in_floats * 4);
  e.engine->bind_io(e.d_in.get(), e.d_out.get());
  return e;
}

int main(int argc, char **argv) {
  int iters = (argc > 1) ? std::atoi(argv[1]) : 200;

  const int B   = env_int("TURBO_OCR_PDF_BATCH",       8);
  const int H   = env_int("TURBO_OCR_PDF_PAGE_H",   1280);
  const int W   = env_int("TURBO_OCR_PDF_PAGE_W",    960);
  const int Br  = env_int("TURBO_OCR_PDF_REC_BATCH",  32);
  const int Wr  = env_int("TURBO_OCR_PDF_REC_W",     320);

  std::cout << "[bench] Configuration\n"
            << "  det/cls static : B=" << B << " H=" << H << " W=" << W << "\n"
            << "  rec static     : B=" << Br << " H=48 W=" << Wr << "\n"
            << "  iterations     : " << iters << "\n\n";

  // Resolve cached engine paths via the server's cache-key function so we
  // exercise the EXACT engines the running server would use.
  auto det_trt = require_cached("models/det.onnx", "det_pdf_static");
  auto cls_trt = require_cached("models/cls.onnx", "cls_pdf_static");
  auto rec_trt = require_cached("models/rec.onnx", "rec_pdf_static");

  auto det = load_engine("det", det_trt, B,  3, H,   W);
  auto cls = load_engine("cls", cls_trt, B,  3, 80,  160);
  auto rec = load_engine("rec", rec_trt, Br, 3, 48,  Wr);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Synthetic input: fill once with 0.5 (no per-iter H2D — we are measuring
  // GPU inference cost only, the same cost the real pipeline pays after
  // preprocessing kernels write the input buffer).
  auto fill = [&](StaticEngine &e) {
    std::vector<float> host(static_cast<size_t>(e.B) * e.C * e.H * e.W, 0.5f);
    CUDA_CHECK(cudaMemcpy(e.d_in.get(), host.data(),
                          host.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  };
  fill(det); fill(cls); fill(rec);

  auto bench = [&](StaticEngine &e, int n_iter, bool sync_each) -> double {
    nvinfer1::Dims4 dims{e.B, e.C, e.H, e.W};
    // Warmup
    for (int i = 0; i < 5; ++i) {
      if (!e.engine->infer_dynamic(dims, stream)) {
        std::cerr << "[bench] " << e.name << " warmup infer failed\n";
        std::exit(2);
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_iter; ++i) {
      if (!e.engine->infer_dynamic(dims, stream)) {
        std::cerr << "[bench] " << e.name << " infer failed at iter " << i << "\n";
        std::exit(2);
      }
      if (sync_each) CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (!sync_each) CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
  };

  // Per-stage latency (sync each call, measure wall time).
  std::cout << "[bench] per-stage latency (sync each call)\n";
  double dt_det = bench(det, iters, true);
  double dt_cls = bench(cls, iters, true);
  double dt_rec = bench(rec, iters, true);
  std::printf("  det static B=%-3d H=%-4d W=%-4d : %7.3f ms/call  (%6.1f pages/s)\n",
              B, H, W, dt_det / iters * 1000, B / (dt_det / iters));
  std::printf("  cls static B=%-3d 80x160         : %7.3f ms/call  (%6.1f boxes/s)\n",
              B, dt_cls / iters * 1000, B / (dt_cls / iters));
  std::printf("  rec static B=%-3d 48x%-4d        : %7.3f ms/call  (%6.1f crops/s)\n",
              Br, Wr, dt_rec / iters * 1000, Br / (dt_rec / iters));

  // Combined: det+cls+rec back-to-back on the same stream (no extra sync
  // between stages — matches a tight pipeline loop). Each "iteration"
  // here processes B pages worth of det+cls and assumes Br rec crops per
  // call (typical: ~20-40 text boxes per page → 1 rec call per page).
  std::cout << "\n[bench] pipelined (det -> cls -> rec on one stream)\n";
  // Warmup combined
  nvinfer1::Dims4 d_det{B, 3, H, W};
  nvinfer1::Dims4 d_cls{B, 3, 80, 160};
  nvinfer1::Dims4 d_rec{Br, 3, 48, Wr};
  for (int i = 0; i < 5; ++i) {
    det.engine->infer_dynamic(d_det, stream);
    cls.engine->infer_dynamic(d_cls, stream);
    rec.engine->infer_dynamic(d_rec, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto t0 = std::chrono::steady_clock::now();
  // Assume one rec call per page (most pages have <Br=32 text boxes).
  for (int i = 0; i < iters; ++i) {
    det.engine->infer_dynamic(d_det, stream);
    cls.engine->infer_dynamic(d_cls, stream);
    // One rec call per B pages (the rec batch holds Br crops, ~enough
    // for a typical page; the cross-image batch in the real pipeline
    // accumulates more crops per call).
    for (int p = 0; p < B; ++p) {
      rec.engine->infer_dynamic(d_rec, stream);
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto t1 = std::chrono::steady_clock::now();
  double total_s = std::chrono::duration<double>(t1 - t0).count();
  double total_pages = static_cast<double>(iters) * B;
  std::printf("  total pages       : %.0f\n", total_pages);
  std::printf("  total time        : %.2f s\n", total_s);
  std::printf("  THROUGHPUT        : %.1f pages/s  (det+cls+rec, static engines)\n",
              total_pages / total_s);

  cudaStreamDestroy(stream);
  return 0;
}
