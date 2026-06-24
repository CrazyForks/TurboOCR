#pragma once

#include <memory>
#include <string>

#include <NvInfer.h>

// Process-global TensorRT engine cache.
//
// Every GPU pool worker builds its own OcrPipeline, and each pipeline's
// recognizers (det/rec/cls/layout/table/formula) previously deserialized their
// OWN ICudaEngine from the same .trt file — N× weight duplication in VRAM for a
// pool of size N. A deserialized ICudaEngine is thread-safe to share across
// threads; only IExecutionContext must be per-thread. So we deserialize each
// .trt blob exactly once (keyed by file path) behind a shared IRuntime, hand
// every caller the SHARED engine via shared_ptr, and let each caller create its
// own per-thread IExecutionContext.
//
// Lifetime: the cache holds weak_ptr<ICudaEngine> and hands out shared_ptr to
// callers, so an engine stays alive exactly as long as some caller still holds
// it. The per-instance execution contexts are owned by the individual
// recognizer/TrtEngine instances; while any of those instances live they also
// keep the shared engine alive (the engine outlives every context created from
// it). Once the last sharing instance dies the engine is freed and its cache
// entry's weak_ptr expires; a later request for the same path re-deserializes.

namespace turbo_ocr::engine {

// Deserialize the engine at `trt_path` once and cache it, or return the already
// cached shared engine on subsequent calls with the same path. Thread-safe.
// Returns nullptr on read/deserialize failure (no entry is cached on failure).
//
// The returned engine is shared — callers MUST create their own
// createExecutionContext() and never share that context across threads.
[[nodiscard]] std::shared_ptr<nvinfer1::ICudaEngine>
get_or_load_engine(const std::string &trt_path);

} // namespace turbo_ocr::engine
