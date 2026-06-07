// pdf_static_bench: measure raw TRT inference throughput of the PDF-only
// static-shape DET engine.
//
// What this proves: the speed lever of pdf_only hybrid mode — a det TRT
// engine built with a static (min==opt==max) shape for a known fixed PDF
// render resolution and batch dim — is materially faster than the
// dynamic-shape equivalent.
//
// What this does NOT cover: detection post-processing (CCL/JFA), the
// dynamic 5-bucket rec (unchanged vs the normal pipeline in hybrid mode),
// CTC decode, the PDF render itself, table/formula stages. Those are
// constant or external to the GPU inference cost we're benchmarking.
//
// Usage:
//   pdf_static_bench [iters]
//   env vars match the server:
//     TURBO_OCR_PDF_PAGE_H, _PAGE_W, _BATCH
//   The .trt engine file must already exist in ~/.cache/turbo-ocr/ (the
//   server builds it on first boot in pdf_only mode).

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
#include "turbo_ocr/server/env_utils.h"

namespace fs = std::filesystem;
using turbo_ocr::engine::TrtEngine;

// Resolve the built static engine via the same cache key the server uses.
// If it doesn't exist (server hasn't booted in pdf_only yet), abort with a
// pointer to how to fix.
static std::string require_cached(const std::string &onnx, const char *type) {
  auto cached = turbo_ocr::engine::get_cached_engine_path(onnx, type);
  if (!fs::exists(cached)) {
    std::cerr << "[bench] Cached static engine missing: " << cached
              << "\n  Boot the server once with TURBO_OCR_PDF_ONLY=1 and the\n"
              << "  same TURBO_OCR_PDF_PAGE_H/W/BATCH env vars to build it.\n"
              << "  Then re-run this benchmark.\n";
    std::exit(2);
  }
  return cached;
}

int main(int argc, char **argv) {
  int iters = (argc > 1) ? std::atoi(argv[1]) : 200;

  const int B = turbo_ocr::server::env_int("TURBO_OCR_PDF_BATCH",  8,    1,  8);
  const int H = turbo_ocr::server::env_int("TURBO_OCR_PDF_PAGE_H", 1280, 32, 4096);
  const int W = turbo_ocr::server::env_int("TURBO_OCR_PDF_PAGE_W", 960,  32, 4096);

  std::cout << "[bench] Configuration\n"
            << "  det static : B=" << B << " H=" << H << " W=" << W << "\n"
            << "  iterations : " << iters << "\n\n";

  // Resolve the cached engine path via the server's cache-key function so
  // we exercise the EXACT engine the running server would use. The cache
  // key includes the onnx path/size/mtime, so honor the same DET_ONNX env
  // the server resolves its det model from.
  const char *det_onnx_env = std::getenv("DET_ONNX");
  const std::string det_onnx = det_onnx_env ? det_onnx_env : "models/det.onnx";
  if (!fs::exists(det_onnx)) {
    std::cerr << "[bench] det ONNX not found: " << det_onnx
              << " (set DET_ONNX or run from the repo root)\n";
    return 2;
  }
  auto det_trt = require_cached(det_onnx, "det_pdf_static");

  TrtEngine engine(det_trt);
  if (!engine.load()) {
    std::cerr << "[bench] Failed to load engine: " << det_trt << '\n';
    return 2;
  }

  // Allocate input + output buffers sized to the static shape. The det
  // output is one prob value per input pixel; sizing the output at the
  // input float count is therefore a safe upper bound.
  const size_t in_floats = static_cast<size_t>(B) * 3 * H * W;
  turbo_ocr::CudaPtr<float> d_in(in_floats);
  turbo_ocr::CudaPtr<float> d_out(in_floats);
  engine.bind_io(d_in.get(), d_out.get());

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Synthetic input: fill once with 0.5 (no per-iter H2D — we are measuring
  // GPU inference cost only, the same cost the real pipeline pays after
  // preprocessing kernels write the input buffer).
  {
    std::vector<float> host(in_floats, 0.5f);
    CUDA_CHECK(cudaMemcpy(d_in.get(), host.data(),
                          host.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  nvinfer1::Dims4 dims{B, 3, H, W};

  // Warmup
  for (int i = 0; i < 5; ++i) {
    if (!engine.infer_dynamic(dims, stream)) {
      std::cerr << "[bench] det warmup infer failed\n";
      return 2;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Per-call latency (sync each call, measure wall time).
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    if (!engine.infer_dynamic(dims, stream)) {
      std::cerr << "[bench] det infer failed at iter " << i << "\n";
      return 2;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  auto t1 = std::chrono::steady_clock::now();
  double dt_sync = std::chrono::duration<double>(t1 - t0).count();
  std::printf("[bench] per-call latency (sync each call)\n");
  std::printf("  det static B=%-3d H=%-4d W=%-4d : %7.3f ms/call  (%6.1f pages/s)\n",
              B, H, W, dt_sync / iters * 1000, B / (dt_sync / iters));

  // Pipelined: back-to-back enqueues on one stream, single final sync —
  // matches a tight batch loop where the next chunk enqueues while the
  // previous one drains.
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    if (!engine.infer_dynamic(dims, stream)) {
      std::cerr << "[bench] det infer failed at iter " << i << "\n";
      return 2;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  t1 = std::chrono::steady_clock::now();
  double dt_pipe = std::chrono::duration<double>(t1 - t0).count();
  double total_pages = static_cast<double>(iters) * B;
  std::printf("\n[bench] pipelined (back-to-back enqueues, one sync)\n");
  std::printf("  total pages : %.0f\n", total_pages);
  std::printf("  total time  : %.2f s\n", dt_pipe);
  std::printf("  THROUGHPUT  : %.1f pages/s  (det static engine)\n",
              total_pages / dt_pipe);

  cudaStreamDestroy(stream);
  return 0;
}
