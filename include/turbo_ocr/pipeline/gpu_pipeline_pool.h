#pragma once

#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "turbo_ocr/pipeline/ocr_pipeline.h"
#include "turbo_ocr/pipeline/pipeline_pool.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"

// TimeoutError now lives in turbo_ocr/common/errors.h (included above) so the
// CPU build sees it too.

namespace turbo_ocr::pipeline {

// Immutable recipe for (re)building one GpuPipelineEntry. Captured once at
// pool construction so a watchdog can rebuild a wedged entry without threading
// every model path back through the call site. All fields are owned by value
// so the spec safely outlives the request that triggered the rebuild.
struct PipelineBuildSpec {
  std::string det_model;
  std::string rec_model;
  std::string rec_dict;
  std::string cls_model;
  std::string layout_model;
  int pdf_only_batch = 0;
  int pdf_only_page_h = 0;
  int pdf_only_page_w = 0;
  std::string doc_ori_model;
  DetInferConfig det_cfg{turbo_ocr::detection::kDetResizeDefault,
                         turbo_ocr::detection::kDbDefaults};
};

/// OcrPipeline + its dedicated CUDA stream + its own nvJPEG decoder, managed
/// as a single poolable unit. One per dispatcher worker thread.
struct GpuPipelineEntry {
  std::unique_ptr<OcrPipeline> pipeline;
  cudaStream_t stream = nullptr;
  // Lazily constructed on the worker thread so the nvJPEG handle binds to
  // the same primary context that owns `stream` and `pipeline`.
  std::unique_ptr<decode::NvJpegDecoder> nvjpeg;

  GpuPipelineEntry() = default;

  GpuPipelineEntry(std::unique_ptr<OcrPipeline> p, cudaStream_t s)
      : pipeline(std::move(p)), stream(s) {}

  ~GpuPipelineEntry() noexcept {
    nvjpeg.reset();
    if (stream)
      cudaStreamDestroy(stream);
  }

  decode::NvJpegDecoder &get_nvjpeg() {
    if (!nvjpeg) nvjpeg = std::make_unique<decode::NvJpegDecoder>();
    return *nvjpeg;
  }

  // Rebuild a wedged entry in place: tear down the old pipeline, stream, and
  // nvJPEG handle, then construct a fresh OcrPipeline + stream from `spec` and
  // re-warm it. Must run on the worker thread that owns this entry (so the new
  // CUDA resources bind to that thread's primary context, like get_nvjpeg).
  // Throws on init/load failure, leaving the entry in a torn-down (pipeline ==
  // nullptr) state; the caller treats that as a still-dead slot. A wedged
  // stream may refuse cudaStreamDestroy with a sticky error — that's tolerated
  // here (the GPU context itself is poisoned by then, handled separately).
  void recycle(const PipelineBuildSpec &spec) {
    nvjpeg.reset();
    pipeline.reset();
    if (stream) {
      cudaStreamDestroy(stream); // best effort: may fail on a poisoned context
      stream = nullptr;
    }

    auto fresh = std::make_unique<OcrPipeline>();
    if (!fresh->init(spec.det_model, spec.rec_model, spec.rec_dict,
                     spec.cls_model, spec.det_cfg))
      throw turbo_ocr::ModelLoadError("[Recycle] Failed to re-init GPU pipeline");
    if (spec.pdf_only_batch > 0)
      fresh->enable_pdf_only_mode(spec.pdf_only_batch, spec.pdf_only_page_h,
                                  spec.pdf_only_page_w);
    if (!spec.layout_model.empty() && !fresh->load_layout_model(spec.layout_model))
      throw turbo_ocr::ModelLoadError("[Recycle] Failed to reload layout model");
    if (!spec.doc_ori_model.empty())
      (void)fresh->load_doc_ori_model(spec.doc_ori_model); // soft-disable on fail

    cudaStream_t fresh_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&fresh_stream, cudaStreamNonBlocking));
    pipeline = std::move(fresh);
    stream = fresh_stream;
    pipeline->warmup_gpu(stream);
  }

  GpuPipelineEntry(GpuPipelineEntry &&o) noexcept
      : pipeline(std::move(o.pipeline)), stream(o.stream),
        nvjpeg(std::move(o.nvjpeg)) {
    o.stream = nullptr;
  }
  GpuPipelineEntry &operator=(GpuPipelineEntry &&o) noexcept {
    if (this != &o) {
      nvjpeg.reset();
      if (stream) cudaStreamDestroy(stream);
      pipeline = std::move(o.pipeline);
      stream = o.stream;
      nvjpeg = std::move(o.nvjpeg);
      o.stream = nullptr;
    }
    return *this;
  }
  GpuPipelineEntry(const GpuPipelineEntry &) = delete;
  GpuPipelineEntry &operator=(const GpuPipelineEntry &) = delete;
};

/// Convenience alias — a PipelinePool of GpuPipelineEntry.
using GpuPipelinePool = PipelinePool<GpuPipelineEntry>;

/// Factory: create, init, warmup GPU pipelines and return a pool.
/// `layout_model` is an optional TRT engine path — pass "" to disable the
/// layout stage entirely (zero added cost at runtime).
[[nodiscard]] inline std::unique_ptr<GpuPipelinePool> make_gpu_pipeline_pool(
    int pool_size, const std::string &det_model, const std::string &rec_model,
    const std::string &rec_dict, const std::string &cls_model = "",
    const std::string &layout_model = "",
    const DetInferConfig &det_cfg = {turbo_ocr::detection::kDetResizeDefault,
                                     turbo_ocr::detection::kDbDefaults}) {

  if (pool_size <= 0) [[unlikely]]
    throw std::invalid_argument(
        std::format("[Pool] Invalid pool_size={}, must be > 0", pool_size));

  std::vector<std::unique_ptr<GpuPipelineEntry>> entries;
  for (int i = 0; i < pool_size; ++i) {
    auto pipeline = std::make_unique<OcrPipeline>();
    if (!pipeline->init(det_model, rec_model, rec_dict, cls_model, det_cfg)) {
      std::cerr << std::format("[Pool] Failed to init GPU pipeline {} of {}", i, pool_size) << '\n';
      continue;
    }
    if (!layout_model.empty()) {
      if (!pipeline->load_layout_model(layout_model)) {
        // Fail hard: mixing layout-on / layout-off pipelines in the same
        // pool would make response shape non-deterministic depending on
        // which handle the request happened to acquire.
        throw turbo_ocr::ModelLoadError(std::format(
            "[Pool] Failed to load layout model for pipeline {} of {}",
            i, pool_size));
      }
    }
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    entries.push_back(std::make_unique<GpuPipelineEntry>(std::move(pipeline), stream));
  }

  if (entries.empty()) [[unlikely]]
    throw turbo_ocr::ModelLoadError(
        std::format("[Pool] All {} GPU pipelines failed to initialize", pool_size));

  std::cout << std::format("Warming up {} pipelines...", entries.size()) << '\n';
  for (auto &e : entries) {
    e->pipeline->warmup_gpu(e->stream);
  }
  std::cout << "Pipeline warmup complete." << '\n';

  return std::make_unique<GpuPipelinePool>(std::move(entries));
}

} // namespace turbo_ocr::pipeline
