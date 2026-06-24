#include "turbo_ocr/detection/paddle_det.h"
#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/detection/det_postprocess.h"

#include "turbo_ocr/common/errors.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ranges>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

using turbo_ocr::engine::TrtEngine;

namespace turbo_ocr::detection {

bool PaddleDet::load_model(const std::string& model_path, const DetResizeParams& resize,
                           const DbParams& db) {
  engine_ = std::make_unique<TrtEngine>(model_path);
  if (!engine_->load())
    return false;
  return init_buffers(resize, db);
}

bool PaddleDet::init_buffers(const DetResizeParams& resize, const DbParams& db) {
  // Per-model resize policy with env overrides layered on (env wins). The
  // engine builder must size its TRT profile MAX off the same effective
  // max-side or the runtime and engine silently disagree.
  resize_ = read_det_resize(resize);
  kMaxSideLen_ = effective_det_max_side(resize_);

  size_t max_pixels = static_cast<size_t>(kMaxSideLen_) * kMaxSideLen_;
  input_size_ = max_pixels * 3 * sizeof(float);
  output_size_ = max_pixels * sizeof(float);

  d_input_ = CudaPtr<float>(max_pixels * 3);
  d_output_ = CudaPtr<float>(max_pixels);

  // Pre-allocate bitmap buffer (kMaxSideLen_ x kMaxSideLen_ uint8)
  d_bitmap_buf_ = CudaPtr<uint8_t>(max_pixels);

  // Pre-allocate batch buffers (kMaxBatchSize x max_pixels)
  batch_input_size_ = static_cast<size_t>(kMaxBatchSize) * max_pixels * 3 * sizeof(float);
  batch_output_size_ = static_cast<size_t>(kMaxBatchSize) * max_pixels * sizeof(float);
  d_batch_input_ = CudaPtr<float>(static_cast<size_t>(kMaxBatchSize) * max_pixels * 3);
  d_batch_output_ = CudaPtr<float>(static_cast<size_t>(kMaxBatchSize) * max_pixels);
  d_batch_bitmap_ = CudaPtr<uint8_t>(static_cast<size_t>(kMaxBatchSize) * max_pixels);

  // Device arrays for batched kernel launch parameters
  d_batch_src_ptrs_ = CudaPtr<void *>(kMaxBatchSize);
  d_batch_src_steps_ = CudaPtr<int>(kMaxBatchSize);
  d_batch_src_heights_ = CudaPtr<int>(kMaxBatchSize);
  d_batch_src_widths_ = CudaPtr<int>(kMaxBatchSize);
  // Pinned host staging for async copy (avoids pageable fallback)
  h_batch_src_ptrs_ = CudaHostPtr<void *>(kMaxBatchSize);
  h_batch_src_steps_ = CudaHostPtr<int>(kMaxBatchSize);
  h_batch_src_heights_ = CudaHostPtr<int>(kMaxBatchSize);
  h_batch_src_widths_ = CudaHostPtr<int>(kMaxBatchSize);

  // Set working pointers to the single-image buffers
  cur_output_ = d_output_.get();
  cur_bitmap_ = d_bitmap_buf_.get();

  // Bind I/O pointers once for single-image path (never change)
  engine_->bind_io(d_input_.get(), d_output_.get());

  // DB params: this model's base + env overrides (read_db_params), same
  // det_config.h source as the CPU detector. GPU_BOX_THRESH / GPU_UNCLIP_SCALE
  // remain as overrides layered on top.
  const DbParams eff_db = read_db_params(db);
  db_thresh_ = eff_db.thresh;
  box_thresh_ = eff_db.box_thresh;
  unclip_ratio_ = eff_db.unclip_ratio;

  // GPU CCL mode: 0=CPU contours, 1=GPU CCL+per-ROI findContours (default),
  // 2=all-GPU JFA per-component Euclidean unclip
  if (const char *env = std::getenv("GPU_CCL"))
    gpu_ccl_mode_ = std::atoi(env);
  // box_thresh_ + unclip_scale_ apply to all three modes (0/1/2).
  if (const char *env = std::getenv("GPU_BOX_THRESH"))
    box_thresh_ = (float)std::atof(env);
  if (const char *env = std::getenv("GPU_UNCLIP_SCALE"))
    unclip_scale_ = (float)std::atof(env);

  if (gpu_ccl_mode_ > 0) {
    // Pre-allocate ALL GPU CCL buffers (no per-request alloc)
    d_ccl_labels_ = CudaPtr<int>(max_pixels);
    d_ccl_compact_ids_ = CudaPtr<int>(max_pixels);
    d_ccl_id_counter_ = CudaPtr<int>(1);
    // 2x kMaxGpuComponents: first half for per-component bboxes, second half for filtered output
    d_ccl_bboxes_ = CudaPtr<turbo_ocr::kernels::GpuDetBox>(
        turbo_ocr::kernels::kMaxGpuComponents * 2);
    d_ccl_num_boxes_ = CudaPtr<int>(1);

    // Pinned host memory for result transfer
    h_ccl_boxes_ = CudaHostPtr<turbo_ocr::kernels::GpuDetBox>(
        turbo_ocr::kernels::kMaxGpuComponents);
    h_bitmap_ = CudaHostPtr<uint8_t>(max_pixels);
    h_bitmap_pixels_ = max_pixels;

    // JFA (Jump Flooding) per-component label expansion
    d_jfa_labels_ = CudaPtr<uint32_t>(max_pixels);
    d_jfa_seeds_ = CudaPtr<uint32_t>(max_pixels);
    d_jfa_seeds_alt_ = CudaPtr<uint32_t>(max_pixels);
    d_expand_per_comp_ = CudaPtr<float>(turbo_ocr::kernels::kMaxGpuComponents);
    h_exp_boxes_ = CudaHostPtr<turbo_ocr::kernels::GpuDetBox>(
        turbo_ocr::kernels::kMaxGpuComponents);
  }

  return true;
}

// GPU CCL path: connected component labeling on GPU, then extract real contours
// from bitmap within each component's bbox for accurate unclip polygons.
std::vector<Box>
PaddleDet::run_gpu_ccl(int resize_h, int resize_w,
                        int orig_h, int orig_w,
                        cudaStream_t stream) {
  float ratio_h = static_cast<float>(resize_h) / orig_h;
  float ratio_w = static_cast<float>(resize_w) / orig_w;

  int h_num_boxes = 0;
  turbo_ocr::kernels::cuda_gpu_ccl_detect(
      cur_bitmap_, cur_output_, resize_w, resize_h,
      box_thresh_,
      d_ccl_labels_.get(), d_ccl_compact_ids_.get(), d_ccl_id_counter_.get(),
      d_ccl_bboxes_.get(), d_ccl_num_boxes_.get(),
      h_ccl_boxes_.get(), &h_num_boxes, stream);

  std::vector<Box> boxes;
  if (h_num_boxes == 0)
    return boxes;

  // Download ONLY the bitmap (not pred_map -- GPU CCL already computed scores).
  // We need the bitmap for per-ROI findContours to get accurate polygon contours.
  const size_t bitmap_pixels = static_cast<size_t>(resize_h) * resize_w;
  if (bitmap_pixels > h_bitmap_pixels_) [[unlikely]] {
    h_bitmap_ = CudaHostPtr<uint8_t>(bitmap_pixels);
    h_bitmap_pixels_ = bitmap_pixels;
  }
  cv::Mat bitmap(resize_h, resize_w, CV_8UC1, h_bitmap_.get());
  CUDA_CHECK(cudaMemcpyAsync(bitmap.data, cur_bitmap_, resize_w * resize_h,
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  // No pred_map download -- use gb.score from GPU CCL instead of box_score_fast

  boxes.reserve(h_num_boxes);

  // For each GPU-detected component, extract the real contour from the bitmap
  // within the bbox region, then feed to the same unclip pipeline as CPU path.
  for (int i = 0; i < h_num_boxes; i++) {
    const auto &gb = h_ccl_boxes_.get()[i];

    int bw = gb.xmax - gb.xmin + 1;
    int bh = gb.ymax - gb.ymin + 1;
    if (bw < 3 || bh < 3)
      continue;

    // Extract the small bitmap ROI for this component's bbox
    // Pad by 1 pixel to ensure findContours can find closed contours at edges
    int roi_x = std::max(0, gb.xmin - 1);
    int roi_y = std::max(0, gb.ymin - 1);
    int roi_x2 = std::min(resize_w - 1, gb.xmax + 1);
    int roi_y2 = std::min(resize_h - 1, gb.ymax + 1);
    int roi_w = roi_x2 - roi_x + 1;
    int roi_h = roi_y2 - roi_y + 1;

    // Clone the ROI since findContours may modify the source image
    cv::Mat roi = bitmap(cv::Rect(roi_x, roi_y, roi_w, roi_h)).clone();

    // Find contours within this small ROI (~50x20 pixels, negligible cost)
    ccl_roi_contours_buf_.clear();
    cv::findContours(roi, ccl_roi_contours_buf_, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    if (ccl_roi_contours_buf_.empty())
      continue;

    // Pick the largest contour in the ROI (should be the component itself)
    const auto &best_contour = (ccl_roi_contours_buf_.size() == 1)
      ? ccl_roi_contours_buf_[0]
      : *std::ranges::max_element(ccl_roi_contours_buf_, {}, [](const std::vector<cv::Point> &c) {
          return cv::contourArea(c);
        });

    if (best_contour.size() <= 2)
      continue;

    // Shift contour from ROI-local coords to global bitmap coords
    ccl_contour_buf_.clear();
    ccl_contour_buf_.reserve(best_contour.size());
    for (const auto &pt : best_contour)
      ccl_contour_buf_.push_back(cv::Point(pt.x + roi_x, pt.y + roi_y));

    // Use GPU CCL score (already filtered by box_thresh_ in the GPU kernel)
    // Skip box_score_fast — saves downloading pred_map (2.4MB) entirely

    float ssid = 0;
    (void)get_mini_boxes(ccl_contour_buf_, ssid);
    if (ssid < kMinBoxSide)
      continue;

    auto unclipped = unclip(ccl_contour_buf_, unclip_ratio_ * unclip_scale_);
    if (unclipped.size() < 3)
      continue;

    float ssid2 = 0;
    auto box = get_mini_boxes(unclipped, ssid2);
    if (ssid2 < kMinUnclippedSide)
      continue;

    // Scale back to original image
    for (int k = 0; k < 4; ++k) {
      box[k][0] = std::clamp(static_cast<int>(std::round(box[k][0] / ratio_w)), 0, orig_w - 1);
      box[k][1] = std::clamp(static_cast<int>(std::round(box[k][1] / ratio_h)), 0, orig_h - 1);
    }

    // Filter tiny boxes
    int rw = static_cast<int>(std::sqrt(((box[0][0] - box[1][0]) * (box[0][0] - box[1][0])) +
                                        ((box[0][1] - box[1][1]) * (box[0][1] - box[1][1]))));
    int rh = static_cast<int>(std::sqrt(((box[0][0] - box[3][0]) * (box[0][0] - box[3][0])) +
                                        ((box[0][1] - box[3][1]) * (box[0][1] - box[3][1]))));
    if (rw <= 3 || rh <= 3)
      continue;

    boxes.push_back(box);
  }

  return boxes;
}

// GPU CCL + JFA per-component Euclidean unclip (all-GPU).
// 1. CCL on original → compact_ids + bboxes + moments
// 2. JFA propagates nearest-foreground coords (unsigned SDF)
// 3. Expand: pixels within `expand` distance assigned to nearest component
//    via compact_ids lookup → no component merging (Voronoi boundary)
// 4. GPU bbox extraction: one block per component scans expanded labels
// 5. Copy expanded bboxes → scale → filter → output
std::vector<Box>
PaddleDet::run_gpu_ccl_fast(int resize_h, int resize_w,
                              int orig_h, int orig_w,
                              cudaStream_t stream) {
  float ratio_h = static_cast<float>(resize_h) / orig_h;
  float ratio_w = static_cast<float>(resize_w) / orig_w;

  // Step 1: CCL on original mask → compact IDs + original bboxes
  int h_num_boxes = 0;
  int h_num_total = 0;
  turbo_ocr::kernels::cuda_gpu_ccl_detect(
      cur_bitmap_, cur_output_, resize_w, resize_h,
      box_thresh_,
      d_ccl_labels_.get(), d_ccl_compact_ids_.get(), d_ccl_id_counter_.get(),
      d_ccl_bboxes_.get(), d_ccl_num_boxes_.get(),
      h_ccl_boxes_.get(), &h_num_boxes, stream, &h_num_total);

  std::vector<Box> boxes;
  if (h_num_boxes == 0) return boxes;

  // Process all PRE-filter compact_ids — that's what compact_ids[] stores.
  // Score+size filter is applied inside compute_expand_per_comp_kernel.
  using turbo_ocr::kernels::GpuDetBox;
  using turbo_ocr::kernels::kMaxGpuComponents;
  int num_slots = std::min(h_num_total, (int)kMaxGpuComponents);
  if (num_slots == 0) return boxes;

  // Step 2: Per-component expand distance from CCL bboxes (Clipper-equivalent
  // area*ratio/perim). Indexed by PRE-filter compact_id. kMaxExpand is the
  // global cutoff; it also bounds the JFA jump range below — keep them equal.
  constexpr float kMaxExpand = 24.0f;
  turbo_ocr::kernels::cuda_compute_expand_per_comp(
      d_ccl_bboxes_.get(), num_slots, unclip_ratio_ * unclip_scale_, /*min*/ 2.0f, kMaxExpand,
      box_thresh_, d_expand_per_comp_.get(), stream);

  // Step 3: JFA + per-component label expansion (variable cutoff per component).
  // Pass kMaxExpand so JFA bounds its jump range to it (no pixel beyond it survives).
  turbo_ocr::kernels::cuda_jfa_expand_labels(
      cur_bitmap_, d_ccl_compact_ids_.get(), d_expand_per_comp_.get(),
      d_jfa_labels_.get(), resize_w, resize_h, kMaxExpand,
      d_jfa_seeds_.get(), d_jfa_seeds_alt_.get(), stream);

  // Step 4: GPU bbox extraction over expanded region. Launcher inits sentinels
  // for atomic scatter, so no pre-memset needed.
  GpuDetBox *exp_bboxes = d_ccl_bboxes_.get() + kMaxGpuComponents;
  turbo_ocr::kernels::cuda_jfa_extract_bboxes(
      d_jfa_labels_.get(), resize_w, resize_h,
      exp_bboxes, num_slots, stream);

  // Step 5: Copy expanded bboxes to host via the pre-allocated pinned buffer.
  CUDA_CHECK(cudaMemcpyAsync(h_exp_boxes_.get(), exp_bboxes,
      num_slots * sizeof(GpuDetBox), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Filter, scale, output. pixel_count==0 means slot was empty or filtered out.
  boxes.reserve(h_num_boxes);
  for (int i = 0; i < num_slots; i++) {
    const auto &eb = h_exp_boxes_.get()[i];
    if (eb.pixel_count < 9) continue;
    int bw = eb.xmax - eb.xmin + 1, bh = eb.ymax - eb.ymin + 1;
    if (bw < kMinUnclippedSide || bh < kMinUnclippedSide) continue;

    Box box;
    box[0] = {std::clamp(static_cast<int>(std::round(eb.xmin / ratio_w)), 0, orig_w - 1),
              std::clamp(static_cast<int>(std::round(eb.ymin / ratio_h)), 0, orig_h - 1)};
    box[1] = {std::clamp(static_cast<int>(std::round(eb.xmax / ratio_w)), 0, orig_w - 1),
              std::clamp(static_cast<int>(std::round(eb.ymin / ratio_h)), 0, orig_h - 1)};
    box[2] = {std::clamp(static_cast<int>(std::round(eb.xmax / ratio_w)), 0, orig_w - 1),
              std::clamp(static_cast<int>(std::round(eb.ymax / ratio_h)), 0, orig_h - 1)};
    box[3] = {std::clamp(static_cast<int>(std::round(eb.xmin / ratio_w)), 0, orig_w - 1),
              std::clamp(static_cast<int>(std::round(eb.ymax / ratio_h)), 0, orig_h - 1)};
    if (std::abs(box[1][0]-box[0][0]) <= 3 || std::abs(box[3][1]-box[0][1]) <= 3) continue;
    boxes.push_back(box);
  }
  return boxes;
}

// CPU fallback path (original findContours)
std::vector<Box>
PaddleDet::run_cpu_contours(int resize_h, int resize_w,
                             int orig_h, int orig_w,
                             cudaStream_t stream) {
  // Download raw probability map for score filtering
  cv::Mat pred_map(resize_h, resize_w, CV_32F);
  CUDA_CHECK(cudaMemcpyAsync(pred_map.data, cur_output_,
                              resize_h * resize_w * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));

  cv::Mat bitmap(resize_h, resize_w, CV_8UC1);
  CUDA_CHECK(cudaMemcpyAsync(bitmap.data, cur_bitmap_, resize_w * resize_h,
                              cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  return extract_boxes_from_bitmap(pred_map, bitmap, orig_h, orig_w, resize_h, resize_w,
                                   box_thresh_, unclip_ratio_ * unclip_scale_, kMinBoxSide,
                                   kMinUnclippedSide, shifted_buf_, mask_buf_, contours_buf_,
                                   hierarchy_buf_);
}

std::vector<Box>
PaddleDet::run(const GpuImage &gpu_img, int orig_h, int orig_w,
               cudaStream_t stream) {
  // PDF-only static mode: the engine accepts only {B, 3, H, W}, so a
  // dynamic single-image execute would be rejected by TRT. Route through
  // the padded batched path instead — correct for every single-image
  // caller (health probe, /ocr, per-page fallbacks); throughput-critical
  // callers batch pages and call run_batch directly.
  if (pdf_static_batch_ > 0)
    return std::move(run_batch({gpu_img}, {{orig_h, orig_w}}, stream)[0]);

  auto [resize_h, resize_w] = compute_det_resize(orig_h, orig_w, resize_);

  // Reset working pointers to single-image buffers (in case batch mode changed them)
  cur_output_ = d_output_.get();
  cur_bitmap_ = d_bitmap_buf_.get();

  // 1+2. Fused resize + normalize + CHW (single kernel, no intermediate buffer)
  turbo_ocr::kernels::cuda_fused_resize_normalize_det(gpu_img, d_input_.get(), resize_w,
                                                resize_h, stream);

  // 3. Inference (dynamic H,W) -- I/O already bound in load_model
  nvinfer1::Dims4 input_dims{1, 3, resize_h, resize_w};
  if (!engine_->infer_dynamic(input_dims, stream)) {
    throw turbo_ocr::InferenceError("Detection TRT inference failed");
  }

  // 4. Threshold on GPU for bitmap
  turbo_ocr::kernels::cuda_threshold_to_u8(cur_output_, cur_bitmap_, resize_w, resize_h, db_thresh_,
                                           stream);

  // 5. Choose contour extraction path
  if (gpu_ccl_mode_ == 2) {
    return run_gpu_ccl_fast(resize_h, resize_w, orig_h, orig_w, stream);
  } else if (gpu_ccl_mode_ == 1) {
    return run_gpu_ccl(resize_h, resize_w, orig_h, orig_w, stream);
  } else {
    return run_cpu_contours(resize_h, resize_w, orig_h, orig_w, stream);
  }
}

// ============================================================================
// PDF-only static-engine mode
// ============================================================================
void PaddleDet::set_pdf_static_profile(int batch, int page_h, int page_w) {
  pdf_static_batch_ = batch;
  pdf_static_h_ = page_h;
  pdf_static_w_ = page_w;

  // init_buffers() sized the batch + CCL scratch for kMaxBatchSize ×
  // effective_det_max_side² pixels; grow everything indexed by per-image pixels
  // when the fixed PDF page exceeds that, so the static-shape execute can't
  // write past its buffers. Per-slot metadata arrays stay at kMaxBatchSize:
  // ServerConfig clamps pdf_batch to it.
  const size_t page_pixels = static_cast<size_t>(page_h) * page_w;
  const size_t max_pixels  = static_cast<size_t>(kMaxSideLen_) * kMaxSideLen_;
  if (page_pixels <= max_pixels)
    return;

  const size_t batch_px = static_cast<size_t>(kMaxBatchSize) * page_pixels;
  batch_input_size_  = batch_px * 3 * sizeof(float);
  batch_output_size_ = batch_px * sizeof(float);
  d_batch_input_  = CudaPtr<float>(batch_px * 3);
  d_batch_output_ = CudaPtr<float>(batch_px);
  d_batch_bitmap_ = CudaPtr<uint8_t>(batch_px);

  if (gpu_ccl_mode_ > 0) {
    // Per-image CCL/JFA scratch is indexed by the resize_h×resize_w of the
    // slice being post-processed — grow it to the fixed page size too.
    d_ccl_labels_      = CudaPtr<int>(page_pixels);
    d_ccl_compact_ids_ = CudaPtr<int>(page_pixels);
    d_jfa_labels_      = CudaPtr<uint32_t>(page_pixels);
    d_jfa_seeds_       = CudaPtr<uint32_t>(page_pixels);
    d_jfa_seeds_alt_   = CudaPtr<uint32_t>(page_pixels);
  }
}

bool PaddleDet::warmup_pdf_static(cudaStream_t stream) {
  if (pdf_static_batch_ <= 0)
    return false;
  // Bind the batched I/O slot — the cached static engine expects exactly
  // {B, 3, H, W}, which is what its profile was built with.
  engine_->bind_io(d_batch_input_.get(), d_batch_output_.get());
  const size_t in_floats = static_cast<size_t>(pdf_static_batch_) * 3 *
                           pdf_static_h_ * pdf_static_w_;
  // Zero-fill input on device; the actual values don't matter for warmup,
  // we just need TRT to JIT and lazy-alloc with this exact shape.
  CUDA_CHECK(cudaMemsetAsync(d_batch_input_.get(), 0,
                              in_floats * sizeof(float), stream));
  nvinfer1::Dims4 dims{pdf_static_batch_, 3, pdf_static_h_, pdf_static_w_};
  bool ok = engine_->infer_dynamic(dims, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  // Restore single-image binding so any future dynamic-shape callers find
  // the right slot. In pdf_only mode no one should call run() anyway.
  engine_->bind_io(d_input_.get(), d_output_.get());
  return ok;
}

// ============================================================================
// Batched detection: process N images in a single TRT inference call.
// All images are resized to the same target dimensions (max of the batch).
// ============================================================================
std::vector<std::vector<Box>>
PaddleDet::run_batch(const std::vector<GpuImage> &gpu_imgs,
                     const std::vector<std::pair<int,int>> &orig_dims,
                     cudaStream_t stream) {
  const int n = static_cast<int>(gpu_imgs.size());
  if (n == 0)
    return {};

  // PDF-only static-engine mode: the engine accepts ONLY {B, 3, H, W}.
  // We always submit exactly that, padding shorter inputs by repeating
  // slot 0 (the pad slots' output gets discarded after post-processing).
  const bool pdf_only = pdf_static_batch_ > 0;

  // Static mode with more pages than the profile's batch dim: chunk into
  // B-page executes instead of silently dropping the tail.
  if (pdf_only && n > pdf_static_batch_) {
    std::vector<std::vector<Box>> all;
    all.reserve(n);
    for (int off = 0; off < n; off += pdf_static_batch_) {
      const int end = std::min(off + pdf_static_batch_, n);
      std::vector<GpuImage> sub(gpu_imgs.begin() + off, gpu_imgs.begin() + end);
      std::vector<std::pair<int,int>> sub_dims(orig_dims.begin() + off,
                                                orig_dims.begin() + end);
      auto part = run_batch(sub, sub_dims, stream);
      for (auto &b : part)
        all.push_back(std::move(b));
    }
    return all;
  }

  // Fallback: single image → use optimized single path (only valid when
  // running with the dynamic engine; pdf_only's static engine can't take
  // a batch=1 shape since its profile is fixed at pdf_batch).
  if (n == 1 && !pdf_only) {
    auto boxes = run(gpu_imgs[0], orig_dims[0].first, orig_dims[0].second, stream);
    return {std::move(boxes)};
  }

  int batch_size;
  int resize_h, resize_w;
  struct PerImgInfo {
    int orig_h, orig_w;
    float ratio;
    int resize_h, resize_w;
  };
  std::vector<PerImgInfo> infos;

  if (pdf_only) {
    // Lock to the static engine's exact profile (set_pdf_static_profile,
    // fed from the validated ServerConfig). The CUDA preprocessing kernel
    // handles arbitrary input H/W → fixed output H/W, so callers can hand
    // us pages of any size and we'll resize-fit to the static shape; the
    // post-processing paths map boxes back to orig space via orig_dims.
    batch_size = pdf_static_batch_;
    resize_h   = pdf_static_h_;
    resize_w   = pdf_static_w_;
    infos.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      // Real input dims for slots 0..min(n,batch_size)-1; pad slots
      // beyond use slot 0's dims (the result is discarded later).
      // src is always in [0, n): n >= 1 here (n==0 returned above) and i >= 0,
      // so the ternary yields either i (when i < n) or 0 — never negative.
      int src = (i < n) ? i : 0;
      // cppcheck-suppress negativeContainerIndex // src in [0,n), see above
      int h = orig_dims[src].first;
      // cppcheck-suppress negativeContainerIndex // src in [0,n), see above
      int w = orig_dims[src].second;
      float ratio = std::min(static_cast<float>(resize_h) / h,
                              static_cast<float>(resize_w) / w);
      infos[i] = {h, w, ratio, resize_h, resize_w};
    }
  } else {
    // Clamp to max batch size
    batch_size = std::min(n, kMaxBatchSize);

    // --- Compute unified resize dimensions (max across batch, rounded to 32) ---
    int max_resize_h = 0, max_resize_w = 0;
    infos.resize(batch_size);

    for (int i = 0; i < batch_size; i++) {
      int h = orig_dims[i].first;
      int w = orig_dims[i].second;
      auto [rh, rw] = compute_det_resize(h, w, resize_);
      float ratio = std::min(static_cast<float>(rh) / h, static_cast<float>(rw) / w);
      infos[i] = {h, w, ratio, rh, rw};
      max_resize_h = std::max(max_resize_h, rh);
      max_resize_w = std::max(max_resize_w, rw);
    }

    // Use the unified (max) dimensions for all images in the batch
    resize_h = max_resize_h;
    resize_w = max_resize_w;
  }
  const int pixels_per_image = resize_h * resize_w;

  // --- 1. Upload per-image metadata to device ---
  // Use pre-allocated pinned buffers for truly async transfers.
  // For pdf_only with n < batch_size, pad slots beyond n with slot 0's
  // GpuImage so the resize kernel has a valid src — the output for
  // padded slots is discarded in the post-processing loop below.
  for (int i = 0; i < batch_size; i++) {
    // src in [0, n): n >= 1 (n==0 returned above), i >= 0, so the ternary is
    // never negative; gpu_imgs has size n == orig_dims.size() (caller builds
    // both in lockstep). The negativeContainerIndex below is a false positive.
    int src = (i < n) ? i : 0;
    // cppcheck-suppress negativeContainerIndex // src in [0,n), see above
    h_batch_src_ptrs_.get()[i]    = gpu_imgs[src].data;
    // cppcheck-suppress negativeContainerIndex // src in [0,n), see above
    h_batch_src_steps_.get()[i]   = static_cast<int>(gpu_imgs[src].step);
    // cppcheck-suppress negativeContainerIndex // src in [0,n), see above
    h_batch_src_heights_.get()[i] = gpu_imgs[src].rows;
    // cppcheck-suppress negativeContainerIndex // src in [0,n), see above
    h_batch_src_widths_.get()[i]  = gpu_imgs[src].cols;
  }

  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_ptrs_.get(), h_batch_src_ptrs_.get(),
                              batch_size * sizeof(void *),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_steps_.get(), h_batch_src_steps_.get(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_heights_.get(), h_batch_src_heights_.get(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_widths_.get(), h_batch_src_widths_.get(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice, stream));

  // --- 2. Batched fused resize + normalize + CHW ---
  turbo_ocr::kernels::cuda_batch_fused_resize_normalize_det(
      (const void *const *)d_batch_src_ptrs_.get(), d_batch_src_steps_.get(),
      d_batch_src_heights_.get(), d_batch_src_widths_.get(),
      d_batch_input_.get(), resize_w, resize_h, batch_size, stream);

  // --- 3. Single TRT inference call with batch=N ---
  // Temporarily rebind I/O to batch buffers
  engine_->bind_io(d_batch_input_.get(), d_batch_output_.get());

  nvinfer1::Dims4 input_dims{batch_size, 3, resize_h, resize_w};
  if (!engine_->infer_dynamic(input_dims, stream)) {
    // Restore single-image binding before throwing
    engine_->bind_io(d_input_.get(), d_output_.get());
    throw turbo_ocr::InferenceError("Batched detection TRT inference failed");
  }

  // Restore single-image I/O binding for future single-image calls
  engine_->bind_io(d_input_.get(), d_output_.get());

  // --- 4. Batched threshold (all images at once) ---
  turbo_ocr::kernels::cuda_batch_threshold_to_u8(d_batch_output_.get(), d_batch_bitmap_.get(),
                                                 resize_w, resize_h, batch_size, db_thresh_,
                                                 stream);

  // --- 5. Per-image post-processing (GPU CCL fast / CPU contours) ---
  // For each image slice in the batch output, run the appropriate path.
  // We temporarily alias cur_output_ / cur_bitmap_ to point at each image's
  // slice. Scope guard ensures they are restored even if an exception is thrown.
  // Output is sized to n (the real input count); the pdf_only pad slots
  // past n are processed but their results are dropped.
  const int real_n = pdf_only ? std::min(n, batch_size) : batch_size;
  std::vector<std::vector<Box>> all_boxes(real_n);

  for (int i = 0; i < real_n; i++) {
    const int orig_h = infos[i].orig_h;
    const int orig_w = infos[i].orig_w;

    // Alias working pointers to this image's slice.
    auto saved_out = cur_output_;
    auto saved_bmp = cur_bitmap_;
    auto restore = [&, saved_out, saved_bmp]() {
      cur_output_ = saved_out;
      cur_bitmap_ = saved_bmp;
    };
    struct ScopeGuard { decltype(restore) fn; ~ScopeGuard() noexcept { fn(); } } guard{restore};

    cur_output_ = d_batch_output_.get() + static_cast<size_t>(i) * pixels_per_image;
    cur_bitmap_ = d_batch_bitmap_.get() + static_cast<size_t>(i) * pixels_per_image;

    if (gpu_ccl_mode_ == 2) {
      all_boxes[i] = run_gpu_ccl_fast(resize_h, resize_w, orig_h, orig_w, stream);
    } else if (gpu_ccl_mode_ == 1) {
      all_boxes[i] = run_gpu_ccl(resize_h, resize_w, orig_h, orig_w, stream);
    } else {
      all_boxes[i] = run_cpu_contours(resize_h, resize_w, orig_h, orig_w, stream);
    }
  }

  // --- 6. Handle overflow: process remaining images via single-image path ---
  if (n > kMaxBatchSize) {
    all_boxes.resize(n);
    for (int i = kMaxBatchSize; i < n; i++) {
      all_boxes[i] = run(gpu_imgs[i], orig_dims[i].first, orig_dims[i].second, stream);
    }
  }

  return all_boxes;
}

} // namespace turbo_ocr::detection
