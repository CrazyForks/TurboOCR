#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "turbo_ocr/classification/doc_orientation.h"
#include "turbo_ocr/classification/paddle_cls.h"
#include "turbo_ocr/common/timing.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/detection/paddle_det.h"
#include "turbo_ocr/layout/paddle_layout.h"
#include "turbo_ocr/pipeline/i_ocr_pipeline.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/recognition/paddle_rec.h"
#include "turbo_ocr/router/routing_plan.h"

namespace turbo_ocr::router  { class CuaRouter; }
namespace turbo_ocr::table   { class TableStage; class NemotronTableStruct; }
namespace turbo_ocr::formula { class IFormulaRecognizer; }

namespace turbo_ocr::pipeline {

class OcrPipeline : public IOcrPipeline {
public:
  OcrPipeline();
  ~OcrPipeline() noexcept override;

  [[nodiscard]] bool init(const std::string &det_model, const std::string &rec_model,
                          const std::string &rec_dict,
                          const std::string &cls_model = "",
                          const DetInferConfig &det_cfg =
                              {turbo_ocr::detection::kDetResizeDefault,
                               turbo_ocr::detection::kDbDefaults}) override;

  // Optionally load a PP-DocLayoutV3 model. Must be called after init().
  // Call once per pipeline. After a successful call, run_with_layout(...)
  // returns both text results and layout detections in one struct.
  // Plain run(...) continues to return text only (layout is computed and
  // discarded to keep a stable API for non-layout callers).
  [[nodiscard]] bool load_layout_model(const std::string &layout_trt_path);

  // Single-pass Nemotron table-structure backend (builds a TRT FP16 engine
  // from the ONNX). When loaded, table regions are recognized by this model
  // and cells are filled from the page's text-OCR results.
  [[nodiscard]] bool load_table_struct_model(const std::string &onnx_path);

  // Load the CUA router + table-stage + formula engines in one call.
  // Pass an empty path for any optional component (e.g. wireless table
  // engines, or the formula engine entirely) to skip it — the loader
  // returns true as long as the *required* parts (router + wired table)
  // initialised. Streams and events for the new modalities are created
  // lazily inside this call; pipelines that never invoke it allocate
  // zero new CUDA resources (required by plan 04 §7).
  [[nodiscard]] bool
  load_router_models(const std::string &table_cls_trt,
                     const std::string &cell_wired_trt,
                     const std::string &cell_wireless_trt,
                     const std::string &slanext_wired_trt,
                     const std::string &slanext_wireless_trt,
                     const std::string &formula_onnx,
                     const std::string &formula_tokenizer_json);

  // PDF-only hybrid mode: static batched det at the fixed render size +
  // batched layout; rec stays the dynamic 5-bucket engine and cls is
  // skipped entirely (rendered PDF pages are upright by construction).
  // Call once after init() with the validated ServerConfig values —
  // the pipeline and its stages never read TURBO_OCR_PDF_* env directly,
  // so the engine profile and the runtime path can't disagree.
  void enable_pdf_only_mode(int batch, int page_h, int page_w);

  // Load the document-orientation model (PP-LCNet_x1_0_doc_ori). Optional;
  // when loaded, detect_orientation() powers /ocr/pdf?autorotate=1. Must be
  // called after init(). Returns false on load failure (caller logs + treats
  // autorotate as unavailable).
  [[nodiscard]] bool load_doc_ori_model(const std::string &doc_ori_trt_path);
  [[nodiscard]] bool has_doc_ori() const noexcept { return use_doc_ori_; }

  // Detect a rendered page's clockwise rotation (0/90/180/270). Returns 0 when
  // the model isn't loaded or confidence is low. Used by the autorotate path
  // to de-rotate the page before OCR + image encode.
  [[nodiscard]] int detect_orientation(const cv::Mat &bgr, cudaStream_t stream);

  // IOcrPipeline interface — delegates to stream-aware overloads with stream=0
  void warmup() override { warmup_gpu(0); }
  [[nodiscard]] std::vector<OCRResultItem> run(const cv::Mat &img) override { return run(img, 0); }

  // GPU-specific: Pre-allocate all buffers and run a dummy inference to warm up
  // TensorRT engines and lazy GPU allocations.
  void warmup_gpu(cudaStream_t stream);

  [[nodiscard]] std::vector<OCRResultItem> run(const cv::Mat &img, cudaStream_t stream);

  // Run OCR on an image already in GPU memory (skips CPU→GPU upload).
  // The GpuImage must contain BGR8 data with the given pitch/step.
  // The caller retains ownership of the GPU buffer.
  [[nodiscard]] std::vector<OCRResultItem> run(GpuImage gpu_img, cudaStream_t stream = 0);

  // Text + layout in one struct. Returns an empty `layout` vector when the
  // pipeline was never loaded with a layout model, so this overload is
  // always safe to call. Preferred over `run()` for HTTP/gRPC paths that
  // surface layout results to clients — it removes the need for a
  // side-channel getter on the pipeline object.
  // Layout is OPT-IN per call: `want_layout=false` (default) runs only
  // det/cls/rec; `want_layout=true` additionally runs layout if the model
  // is loaded. When the pipeline has no layout model, the flag has no
  // effect (the output `.layout` is always empty). Callers (HTTP/gRPC
  // routes) parse `?layout=1` from the request and pass it through.
  [[nodiscard]] OcrPipelineResult run_with_layout(const cv::Mat &img,
                                                   cudaStream_t stream,
                                                   bool want_layout = false,
                                                   bool want_reading_order = false);
  [[nodiscard]] OcrPipelineResult run_with_layout(GpuImage gpu_img,
                                                   cudaStream_t stream = 0,
                                                   bool want_layout = false,
                                                   bool want_reading_order = false);

  // Layout-only path: upload the image, run the PP-DocLayoutV3 inference,
  // collect the boxes, and return. Skips detection, angle classification,
  // and recognition entirely — no CTC decode, no rec_stream synchronisation,
  // no wasted inference.
  //
  // Used by /ocr/pdf mode=geometric and mode=auto (native-text pages) when
  // ENABLE_LAYOUT=1: the page's text comes from the PDFium text layer and
  // only the visual `layout` array still needs the rendered image. Returns
  // an OcrPipelineResult with empty `.results` (caller fills from the text
  // layer) and populated `.layout` (or empty when the pipeline has no
  // layout model, in which case this method is a no-op and still safe).
  [[nodiscard]] OcrPipelineResult run_layout_only(const cv::Mat &img,
                                                   cudaStream_t stream);

  // Ensure the GPU upload buffer can hold an image of the given size.
  // Returns {d_img_buf_, d_img_pitch_} after grow-only reallocation.
  // Useful for callers that want to decode directly into the pipeline's GPU buffer.
  [[nodiscard]] std::pair<void *, size_t> ensure_gpu_buf(int rows, int cols);

  // Batch: process multiple images, return text results per image.
  // Delegates to run_batch_with_layout with want_layout=false, so
  // text-only batches pay zero layout/router cost.
  [[nodiscard]] std::vector<std::vector<OCRResultItem>>
  run_batch(const std::vector<cv::Mat> &imgs, cudaStream_t stream = 0);

  // Batch counterpart of run_with_layout: batched det (static-shape in
  // pdf_only mode), cross-image batched rec, and — when want_layout —
  // layout (one batched TRT execute in pdf_only mode) + CUA router per
  // page. Returns full per-page results: text, layout, tables, formulas,
  // and reading order when requested. Callers should chunk at most
  // kMaxBatchImages images per call (extra images are dropped with a
  // warning, same contract as run_batch).
  [[nodiscard]] std::vector<OcrPipelineResult>
  run_batch_with_layout(const std::vector<cv::Mat> &imgs,
                        cudaStream_t stream = 0,
                        bool want_layout = false,
                        bool want_reading_order = false);

private:
  std::unique_ptr<detection::PaddleDet> det_;
  std::unique_ptr<classification::PaddleCls> cls_;
  std::unique_ptr<recognition::PaddleRec> rec_;
  std::unique_ptr<layout::PaddleLayout> layout_;
  std::unique_ptr<classification::DocOrientation> doc_ori_;

  bool use_cls_ = false;
  bool use_layout_ = false;
  bool use_doc_ori_ = false;

  // PDF-only hybrid mode (see enable_pdf_only_mode). batch/page dims are
  // the static det engine's only accepted profile shape.
  bool pdf_only_ = false;
  int pdf_batch_ = 0;
  int pdf_page_h_ = 0;
  int pdf_page_w_ = 0;

  // Shared GPU upload: wait for previous rec, toggle double-buffer, grow-only
  // realloc, pinned staging memcpy, async H2D. Used by run_with_layout and
  // run_layout_only.
  GpuImage upload_image(const cv::Mat &img, cudaStream_t stream,
                        PipelineTimer &timer);

  // Dedicated stream for recognition — allows det on the caller's stream to
  // overlap with rec on this stream across consecutive requests.
  cudaStream_t rec_stream_ = nullptr;

  // Event recorded on rec_stream_ after recognition GPU work is launched.
  // The next run() waits on this before reusing the image buffer that rec
  // might still be reading.  This enables det/rec overlap across calls:
  //   Call N:    [...det_N on stream...] → launch rec_N on rec_stream_ (async)
  //   Call N+1:  wait rec_event_ → [...det_{N+1} overlaps with tail of rec_N...]
  cudaEvent_t  rec_event_  = nullptr;

  // Event recorded on the caller's stream after det+cls finish.
  // rec_stream_ waits on this before launching recognition, ensuring proper
  // data dependency without a full stream synchronize.
  cudaEvent_t  det_event_  = nullptr;

  // Dedicated stream for optional layout detection. Layout reads gpu_img
  // (written by det's preprocess) and runs independently of cls/rec, so
  // running it on its own stream lets it overlap with rec on rec_stream_
  // and with cls on the caller's stream. Only allocated when layout is
  // enabled via load_layout_model().
  cudaStream_t layout_stream_ = nullptr;

  // Event recorded on the caller's stream right after detection, BEFORE cls.
  // layout_stream_ waits on this so layout can start as soon as det is done
  // without also waiting for cls/rec. Only used when layout is enabled.
  cudaEvent_t  det_only_event_ = nullptr;

  // Double-buffered GPU upload buffers (grow-only).
  // Alternating between two buffers lets recognition read the previous image
  // on rec_stream_ while the current image is uploaded + detected on the
  // caller's stream without a data race.
  int cur_img_buf_ = 0; // toggles 0/1 each run() call
  struct ImgBuf {
    void *d_buf = nullptr;
    size_t pitch = 0;
    int cap_rows = 0;
    int cap_cols = 0;
  };
  ImgBuf img_bufs_[2];

  // Pinned host staging buffer for truly async CPU->GPU uploads
  void *h_pinned_buf_ = nullptr;
  size_t h_pinned_size_ = 0;

  // Reusable buffers for selective angle classification
  std::vector<int> vertical_box_indices_;
  std::vector<Box> vertical_boxes_buf_;

  // Pre-allocated batch image GPU buffers (avoid cudaMalloc per batch image)
  static constexpr int kMaxBatchImages = 8;
  struct BatchImgBuf {
    void *d_buf = nullptr;
    size_t pitch = 0;
    int cap_rows = 0, cap_cols = 0;
  };
  BatchImgBuf batch_img_bufs_[kMaxBatchImages];

  // ---- CUA router + table/formula stages (lazy-allocated) ----------------
  std::unique_ptr<router::CuaRouter>   router_;
  std::unique_ptr<table::TableStage>   table_stage_;
  std::unique_ptr<table::NemotronTableStruct> table_struct_;
  std::unique_ptr<formula::IFormulaRecognizer> formula_;

  cudaStream_t table_stream_       = nullptr;
  cudaStream_t formula_stream_     = nullptr;
  cudaEvent_t  table_done_event_   = nullptr;
  cudaEvent_t  formula_done_event_ = nullptr;

  // Reusable per-call routing plan — member, not local, to avoid the
  // per-page heap churn of the inner vectors.
  router::RoutingPlan plan_;

  // Apply the router + dispatch table/formula stages on a single page.
  // No-op (returns immediately) unless `router_` is loaded AND `out.layout`
  // is non-empty — text-only callers see zero new CUDA API calls.
  void dispatch_router_(OcrPipelineResult &out,
                        const GpuImage &gpu_img,
                        const std::vector<Box> &boxes,
                        PipelineTimer &timer);

  // Phase 4 of run_batch_with_layout: layout (one batched TRT execute in
  // pdf_only mode, per-image otherwise), CUA router dispatch, and reading-
  // order assignment. `image_crops[i].img` is page i on GPU; `.boxes` are
  // the det boxes recognition ran on. Caller guarantees layout_ is loaded.
  void run_batch_layout_stage_(
      const std::vector<recognition::PaddleRec::ImageCrops> &image_crops,
      bool want_reading_order,
      cudaStream_t stream,
      std::vector<OcrPipelineResult> &outs);
};

} // namespace turbo_ocr::pipeline
