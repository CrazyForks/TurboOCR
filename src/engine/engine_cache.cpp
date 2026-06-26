#include "turbo_ocr/engine/engine_cache.h"

#include <fstream>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace turbo_ocr::engine {

namespace {

class CacheLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity != Severity::kINFO)
      std::cerr << "[TRT] " << msg << '\n';
  }
};

// Function-local statics: the runtime and the cache map are constructed on
// first use and destroyed in reverse order at process teardown — the map
// (engines) before the runtime. TensorRT requires the IRuntime to outlive every
// engine deserialized from it; each cached engine's shared_ptr also captures the
// shared runtime in its deleter (see below), so even an engine released after
// the map is cleared keeps the runtime alive until that engine is gone.
struct CacheState {
  std::mutex mu;
  CacheLogger logger;
  std::shared_ptr<nvinfer1::IRuntime> runtime;
  std::unordered_map<std::string, std::weak_ptr<nvinfer1::ICudaEngine>> engines;
};

CacheState &state() {
  static CacheState s;
  return s;
}

std::vector<char> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.good())
    return {};
  const std::streampos pos = f.tellg();
  if (pos < 0)
    return {};
  std::vector<char> buf(static_cast<std::size_t>(pos));
  f.seekg(0, std::ios::beg);
  f.read(buf.data(), pos);
  return buf;
}

} // namespace

// Cross-pipeline engine sharing is OFF by default.
//
// Sharing ONE deserialized ICudaEngine across all pool workers and giving each
// worker its own IExecutionContext is the textbook-"supported" TensorRT pattern
// — but in practice, concurrent enqueueV3() on multiple contexts of the SAME
// engine corrupts recognition output under load: detection still finds the text
// boxes, yet rec decodes them to blank (empty/partial text on ~0.5-0.8% of
// requests at concurrency ~150, the exact "silent text loss" failure). Private
// per-worker engines eliminate the shared mutable engine state entirely AND run
// ~2x faster (no contention on shared engine internals). Each pool stage loads a
// distinct .trt path, so the cache deduped nothing WITHIN a pipeline anyway —
// it only ever shared ACROSS pipelines, which is precisely the unsafe case.
//
// TURBO_OCR_SHARE_ENGINES=1 restores the old shared-engine behaviour; only safe
// when pool concurrency is 1 (e.g. a single-worker, VRAM-constrained deploy).
[[nodiscard]] static bool engine_sharing_enabled() {
  static const bool v = std::getenv("TURBO_OCR_SHARE_ENGINES") != nullptr;
  return v;
}

std::shared_ptr<nvinfer1::ICudaEngine>
get_or_load_engine(const std::string &trt_path) {
  CacheState &s = state();
  std::lock_guard<std::mutex> lock(s.mu);

  const bool share = engine_sharing_enabled();

  if (share) {
  if (auto it = s.engines.find(trt_path); it != s.engines.end()) {
    if (auto eng = it->second.lock())
      return eng;
    // weak_ptr expired (last sharing instance died): drop the dead entry so the
    // map can't accumulate stale keys, then re-deserialize below.
    s.engines.erase(it);
  }
  }

  if (!s.runtime) {
    s.runtime.reset(nvinfer1::createInferRuntime(s.logger));
    if (!s.runtime) {
      std::cerr << "[TRT] Failed to create runtime for: " << trt_path << '\n';
      return nullptr;
    }
  }

  const std::vector<char> blob = read_file(trt_path);
  if (blob.empty()) {
    std::cerr << "[TRT] Error loading engine file: " << trt_path << '\n';
    return nullptr;
  }

  nvinfer1::ICudaEngine *raw =
      s.runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (!raw) {
    std::cerr << "[TRT] Failed to deserialize engine: " << trt_path << '\n';
    return nullptr;
  }

  // Capture the shared runtime in the deleter so the runtime is guaranteed to
  // outlive this engine no matter the static-destruction order.
  std::shared_ptr<nvinfer1::ICudaEngine> eng(
      raw, [rt = s.runtime](nvinfer1::ICudaEngine *p) { delete p; });

  if (share)
    s.engines[trt_path] = eng;
  return eng;
}

} // namespace turbo_ocr::engine
