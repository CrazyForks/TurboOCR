#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <format>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

#include "turbo_ocr/pipeline/gpu_pipeline_pool.h"

namespace turbo_ocr::pipeline {

/// Wait for a future with a deadline. Returns the result if it resolves in
/// time, otherwise throws turbo_ocr::TimeoutError and ABANDONS the future:
/// the underlying task may still be running and writing into whatever it
/// captured. Callers MUST therefore submit work that owns its inputs by
/// value, so a timed-out task can safely outlive the request handler that
/// launched it. `timeout_ms <= 0` waits unbounded (legacy behaviour).
template <typename T>
T get_with_timeout(std::future<T> &future, long timeout_ms) {
  if (timeout_ms <= 0) return future.get();
  if (future.wait_for(std::chrono::milliseconds(timeout_ms)) ==
      std::future_status::timeout)
    throw turbo_ocr::TimeoutError(std::format(
        "Inference exceeded the {} ms request deadline", timeout_ms));
  return future.get();
}

/// Work-queue dispatcher that keeps GPU pipelines permanently busy.
///
/// Each worker thread owns one GpuPipelineEntry and pulls tasks from a
/// shared FIFO queue.  Unlike the acquire/release PipelinePool pattern,
/// the GPU never idles waiting for HTTP round-trip overhead — while one
/// request's response is being serialised and sent, the worker is already
/// processing the next queued image.
class PipelineDispatcher {
public:
  /// `spec` records how each entry was built so a wedged entry can be rebuilt
  /// on its owning worker thread (see request_recycle). Pass {} to disable
  /// recycling (request_recycle becomes a no-op).
  explicit PipelineDispatcher(std::vector<std::unique_ptr<GpuPipelineEntry>> entries,
                              PipelineBuildSpec spec = {})
      : spec_(std::move(spec)) {
    size_t n = entries.size();
    workers_.reserve(n);
    recycle_flags_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      auto flag = std::make_shared<std::atomic<bool>>(false);
      recycle_flags_.push_back(flag);
      workers_.emplace_back([this, flag, entry = std::move(entries[i])]() mutable {
        while (true) {
          WorkFn work;
          {
            std::unique_lock lock(mutex_);
            // Predicate form of wait() is atomic w.r.t. notify only when
            // the notifier holds the same mutex around the state change.
            // The dtor takes mutex_ before flipping stop_ and calling
            // notify_all() — see ~PipelineDispatcher.
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
              if (stop_) return;
              continue;
            }
            work = std::move(queue_.front());
            queue_.pop();
          }
          // Recycle a suspect entry before reusing it. Only this worker may
          // touch its entry, so the rebuild is race-free w.r.t. inference.
          // A still-running timed-out task on the OLD stream is what made the
          // entry suspect; the new stream/pipeline is independent of it.
          if (flag->exchange(false))
            maybe_recycle_(*entry);
          work(*entry);
        }
      });
    }
  }

  ~PipelineDispatcher() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &w : workers_)
      if (w.joinable()) w.join();
  }

  PipelineDispatcher(const PipelineDispatcher &) = delete;
  PipelineDispatcher &operator=(const PipelineDispatcher &) = delete;

  /// Submit work that runs on a GPU worker thread.  Returns a future
  /// whose value is whatever the callable returns.
  ///
  /// The callable signature must be:  R fn(GpuPipelineEntry &)
  template <typename F>
  auto submit(F &&fn) -> std::future<std::invoke_result_t<F, GpuPipelineEntry &>> {
    using R = std::invoke_result_t<F, GpuPipelineEntry &>;
    auto task = std::make_shared<std::packaged_task<R(GpuPipelineEntry &)>>(
        std::forward<F>(fn));
    auto future = task->get_future();
    {
      std::unique_lock lock(mutex_);
      if (queue_.size() >= max_queue_depth_)
        throw turbo_ocr::PoolExhaustedError(
            "Server at capacity (GPU queue full). Use persistent connections "
            "(HTTP keep-alive) instead of opening a new connection per request.");
      queue_.push([task = std::move(task)](GpuPipelineEntry &e) { (*task)(e); });
    }
    cv_.notify_one();
    return future;
  }

  /// Submit work and wait for it with a deadline. On timeout, throws
  /// turbo_ocr::TimeoutError and abandons the still-queued/running task.
  ///
  /// SAFETY: because the task may outlive this call, `fn` MUST capture every
  /// input it touches BY VALUE (no references/pointers into the caller's
  /// stack or request buffers). The result type R must likewise be safe to
  /// discard if the deadline elapses. `timeout_ms <= 0` waits unbounded.
  ///
  /// The callable signature must be:  R fn(GpuPipelineEntry &)
  template <typename F>
  auto submit_for(long timeout_ms, F &&fn)
      -> std::invoke_result_t<F, GpuPipelineEntry &> {
    auto future = submit(std::forward<F>(fn));
    return get_with_timeout(future, timeout_ms);
  }

  [[nodiscard]] size_t worker_count() const noexcept { return workers_.size(); }

  /// Flag worker `worker_index`'s entry as wedged. Its owning worker rebuilds
  /// the OcrPipeline + CUDA stream from the build spec before its next task,
  /// rather than leaking the slot forever. Safe to call from any thread (e.g.
  /// a watchdog) and idempotent. No-op when the dispatcher was constructed
  /// without a build spec, or for an out-of-range index.
  void request_recycle(size_t worker_index) noexcept {
    if (!recycle_enabled_ || worker_index >= recycle_flags_.size()) return;
    recycle_flags_[worker_index]->store(true);
    // Wake an idle worker so a slot with no queued work still recycles. The
    // owning worker re-checks its flag after every wakeup.
    cv_.notify_all();
  }

private:
  using WorkFn = std::function<void(GpuPipelineEntry &)>;

  // Rebuild `entry` from the stored spec; on failure log and leave the entry
  // dead (pipeline == nullptr) — a subsequent task throws rather than touching
  // a half-built pipeline, and the watchdog can request another recycle.
  void maybe_recycle_(GpuPipelineEntry &entry) noexcept {
    try {
      entry.recycle(spec_);
    } catch (const std::exception &e) {
      std::cerr << std::format("[Dispatcher] entry recycle failed: {}\n",
                               e.what());
    }
  }

  std::queue<WorkFn> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<std::thread> workers_;
  std::vector<std::shared_ptr<std::atomic<bool>>> recycle_flags_;
  PipelineBuildSpec spec_;
  bool recycle_enabled_ = !spec_.det_model.empty();
  bool stop_{false};  // guarded by mutex_
  size_t max_queue_depth_ = 4096;
};

/// Factory: create, init, warmup GPU pipelines and wrap in a dispatcher.
/// `pdf_only_batch > 0` enables the PDF-only hybrid mode on every pipeline
/// (static batched det at {batch, 3, page_h, page_w}; values from the
/// validated ServerConfig) before warmup, so the static-engine warmup path
/// is taken instead of the dynamic-shape one.
// Build + warm exactly one pipeline (det/rec/cls + optional layout/doc_ori +
// pdf_only). Returns nullptr on a non-fatal init failure (logged).
[[nodiscard]] inline std::unique_ptr<GpuPipelineEntry> build_one_pipeline(
    int idx, const std::string &det_model, const std::string &rec_model,
    const std::string &rec_dict, const std::string &cls_model,
    const std::string &layout_model, int pdf_only_batch, int pdf_only_page_h,
    int pdf_only_page_w, const std::string &doc_ori_model,
    const DetInferConfig &det_cfg) {
  auto pipeline = std::make_unique<OcrPipeline>();
  if (!pipeline->init(det_model, rec_model, rec_dict, cls_model, det_cfg)) {
    std::cerr << std::format("[Dispatcher] Failed to init GPU pipeline {}\n", idx);
    return nullptr;
  }
  if (pdf_only_batch > 0)
    pipeline->enable_pdf_only_mode(pdf_only_batch, pdf_only_page_h,
                                   pdf_only_page_w);
  if (!layout_model.empty() && !pipeline->load_layout_model(layout_model))
    throw turbo_ocr::ModelLoadError(std::format(
        "[Dispatcher] Failed to load layout model for pipeline {}", idx));
  if (!doc_ori_model.empty())
    (void)pipeline->load_doc_ori_model(doc_ori_model); // soft-disable on fail
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto entry = std::make_unique<GpuPipelineEntry>(std::move(pipeline), stream);
  entry->pipeline->warmup_gpu(entry->stream); // grows buffers + bakes graphs
  return entry;
}

[[nodiscard]] inline std::unique_ptr<PipelineDispatcher> make_pipeline_dispatcher(
    int pool_size, const std::string &det_model, const std::string &rec_model,
    const std::string &rec_dict, const std::string &cls_model = "",
    const std::string &layout_model = "",
    int pdf_only_batch = 0, int pdf_only_page_h = 0, int pdf_only_page_w = 0,
    const std::string &doc_ori_model = "",
    const DetInferConfig &det_cfg = {turbo_ocr::detection::kDetResizeDefault,
                                     turbo_ocr::detection::kDbDefaults}) {

  if (pool_size <= 0) [[unlikely]]
    throw std::invalid_argument(
        std::format("[Dispatcher] Invalid pool_size={}, must be > 0", pool_size));

  std::vector<std::unique_ptr<GpuPipelineEntry>> entries;
  for (int i = 0; i < pool_size; ++i) {
    if (auto e = build_one_pipeline(i, det_model, rec_model, rec_dict,
                                    cls_model, layout_model, pdf_only_batch,
                                    pdf_only_page_h, pdf_only_page_w,
                                    doc_ori_model, det_cfg))
      entries.push_back(std::move(e));
  }

  if (entries.empty()) [[unlikely]]
    throw turbo_ocr::ModelLoadError(std::format(
        "[Dispatcher] All {} GPU pipelines failed to initialize", pool_size));

  std::cout << std::format("Pipeline warmup complete ({} pipelines).\n",
                           entries.size());
  // Carry the build recipe so a watchdog can rebuild a wedged entry in place.
  PipelineBuildSpec spec{det_model,        rec_model,      rec_dict,
                         cls_model,         layout_model,   pdf_only_batch,
                         pdf_only_page_h,   pdf_only_page_w, doc_ori_model,
                         det_cfg};
  return std::make_unique<PipelineDispatcher>(std::move(entries),
                                              std::move(spec));
}

} // namespace turbo_ocr::pipeline
