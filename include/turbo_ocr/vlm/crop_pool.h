#pragma once

// Global async VLM crop pool.
//
// Replaces the per-VLMFormula / per-VLMTable thread-per-crop model with a
// process-global pool that keeps up to VLM_GLOBAL_CONCURRENCY (default 50)
// CURL easy handles in flight simultaneously across ALL pipeline instances.
//
// Usage:
//   auto fut = VLMCropPool::instance().submit(png_bytes, prompt, model,
//                                             max_tokens, timeout_s);
//   std::string result = fut.get();   // blocks only the calling thread
//
// The pool owns a single worker thread that polls curl_multi, drains the
// completion queue, and fulfils promises. Application threads submit and
// wait on futures independently — no cross-page serialisation.
//
// Env vars:
//   VLM_GLOBAL_CONCURRENCY  (default 50)  max simultaneous CURL handles
//   VLM_PNG_THREADS         (default 4)   per-page PNG encode workers (unused here, exposed for callers)

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>

namespace turbo_ocr::vlm {

struct CropRequest {
    std::vector<uint8_t>               png_bytes;
    std::string                        prompt;
    std::string                        model;
    int                                max_tokens  = 512;
    int                                timeout_s   = 30;
    std::string                        base_url;          // includes http://host:port, no trailing slash
    std::promise<std::string>          result;
};

class VLMCropPool {
public:
    // Process-global singleton.
    static VLMCropPool &instance();

    // Submit one crop for async processing. Never blocks the caller beyond
    // a brief queue-lock acquisition. Returns a future that resolves to the
    // extracted text/LaTeX/OTSL string, or empty on error.
    std::future<std::string> submit(std::vector<uint8_t> png_bytes,
                                    std::string          prompt,
                                    std::string          model,
                                    int                  max_tokens,
                                    int                  timeout_s,
                                    std::string          base_url);

    // Graceful shutdown: called by destructor. Waits for in-flight ops.
    void shutdown();

    int max_concurrency() const noexcept { return max_concurrency_; }

    VLMCropPool(const VLMCropPool &) = delete;
    VLMCropPool &operator=(const VLMCropPool &) = delete;

private:
    VLMCropPool();
    ~VLMCropPool();

    void worker_loop();

    // Per-handle state kept alive during the async operation.
    struct HandleCtx {
        CURL                      *easy      = nullptr;
        struct curl_slist         *headers   = nullptr;
        std::string                post_body;  // JSON body (kept alive for CURL)
        std::string                response;   // accumulates write callback data
        std::promise<std::string>  result;
        VLMCropPool               *pool       = nullptr;
    };

    static size_t write_cb(char *ptr, size_t sz, size_t nmemb, void *ud);
    void complete_handle(CURL *easy, CURLcode rc);
    void try_dispatch();  // pop from queue_ and add to multi if capacity allows

    int                          max_concurrency_;
    CURLM                       *multi_           = nullptr;
    std::thread                  worker_;
    std::atomic<bool>            stop_{false};

    // Pending requests waiting for a free slot.
    std::mutex                   queue_mu_;
    std::condition_variable      queue_cv_;
    std::queue<std::unique_ptr<CropRequest>> queue_;

    // Count of handles currently in the multi handle.
    std::atomic<int>             in_flight_{0};

    // Pipe fd pair for waking the worker thread from submit().
    int wake_rd_ = -1;
    int wake_wr_ = -1;
};

} // namespace turbo_ocr::vlm
