#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "turbo_ocr/decode/gpu_image.h"

namespace turbo_ocr::formula::kernels {

using decode::GpuImage;

// Sub-rect of a page GpuImage pointing at one formula crop.
struct FormulaSubRect {
  int x0;
  int y0;
  int w;
  int h;
};

// Constants mirroring `turbostruct-rs/crates/turbostruct-formula` and PaddleX
// UniMERNet. All three channels of `FORMULA_MEAN`/`FORMULA_STD` are identical,
// which lets us collapse the per-channel normalize and the BT.601 grayscale
// step into a single post-grayscale normalize (mathematically equivalent).
inline constexpr int   kFormulaH = 384;
inline constexpr int   kFormulaW = 384;
inline constexpr float kFormulaMean = 0.7931f;
inline constexpr float kFormulaStd  = 0.1738f;
// Foreground threshold over min-max normalized luminance (PaddleX uses
// `255 * (data < 200)`).
inline constexpr float kFormulaForegroundThreshold = 200.0f;

// Fully on-device fused preprocess for a batch of formula crops.
//
// Mirrors `preprocess_crop` from the Rust crate:
//   1. PIL "L" grayscale + min-max normalize + threshold < 200 -> tight bbox.
//   2. resize_short_then_thumbnail to 384 (preserve AR, cap at 384).
//   3. Center pad to 384x384 (constant 0 in byte domain).
//   4. Bilinear sample + per-channel normalize + BT.601 BGR->GRAY collapse,
//      producing a single f32 plane per crop.
//
// Buffer ownership (caller-allocated, all on the same `stream`):
//   d_subrects : [B]    FormulaSubRect, sub-rects into `page`.
//   d_bbox     : [B*4]  int32 scratch (cx, cy, cw, ch in sub-rect coords).
//   d_output   : [B*H*W] float32 output, shape (B, 1, 384, 384).
//
// `page` must be 3-channel BGR uint8 (matches OpenCV cv::Mat default and the
// rest of this codebase's GpuImage layout). The Rust crate consumes RGB but
// then swaps to BGR for sampling; PIL "L" applied to RGB equals BT.601 on
// BGR, so the luminance values match either way.
void cuda_fused_formula_pre(const GpuImage &page,
                            const FormulaSubRect *d_subrects,
                            int batch_size,
                            int32_t *d_bbox,
                            float *d_output,
                            cudaStream_t stream);

// Workspace size helper -- bytes of `d_bbox` scratch needed for `batch_size`
// crops. Kept here so callers can size a single arena buffer.
[[nodiscard]] inline constexpr std::size_t formula_bbox_scratch_bytes(int batch_size) noexcept {
  return static_cast<std::size_t>(batch_size) * 4 * sizeof(int32_t);
}

[[nodiscard]] inline constexpr std::size_t formula_output_bytes(int batch_size) noexcept {
  return static_cast<std::size_t>(batch_size) * kFormulaH * kFormulaW * sizeof(float);
}

} // namespace turbo_ocr::formula::kernels
