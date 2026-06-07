#include "turbo_ocr/formula/formula_kernels.h"

#include "turbo_ocr/common/cuda_check.h"

#include <climits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace turbo_ocr::formula::kernels {

namespace {

// PIL "L" weights == BT.601 weights with channels swapped. The luminance
// value is identical whether the source bytes are RGB or BGR, so we use
// BT.601 (B,G,R) here and let the caller treat page bytes as BGR (matches
// OpenCV cv::Mat default everywhere else in this codebase).
__device__ __forceinline__ int luminance_bgr(uint8_t b, uint8_t g, uint8_t r) {
  float L = 0.114f * static_cast<float>(b)
          + 0.587f * static_cast<float>(g)
          + 0.299f * static_cast<float>(r);
  L = fminf(fmaxf(L + 0.5f, 0.0f), 255.0f); // round + clamp like PIL "L"
  return static_cast<int>(L);
}

// One block per crop. 256 threads stride over the crop's sub-rect pixels in
// two passes; __syncthreads guarantees pass 1 (min/max) finishes before
// pass 2 (threshold bbox) starts. Per-crop work is small (<=1 Mpx) so the
// low-occupancy launch is fine -- the whole step is budgeted at <1 ms.
__global__ __launch_bounds__(256)
void formula_bbox_kernel(const uint8_t * __restrict__ page_data,
                          int page_step, int page_rows, int page_cols,
                          const FormulaSubRect * __restrict__ d_subrects,
                          int32_t * __restrict__ d_bbox) {
  int b = blockIdx.z;
  FormulaSubRect rect = d_subrects[b];

  // Clamp sub-rect to page bounds and guarantee non-empty.
  int sx0 = max(0, rect.x0);
  int sy0 = max(0, rect.y0);
  int sx1 = min(page_cols, rect.x0 + rect.w);
  int sy1 = min(page_rows, rect.y0 + rect.h);
  int sw = max(1, sx1 - sx0);
  int sh = max(1, sy1 - sy0);
  int total = sw * sh;

  __shared__ int s_min;
  __shared__ int s_max;
  __shared__ int s_xmin;
  __shared__ int s_ymin;
  __shared__ int s_xmax;
  __shared__ int s_ymax;
  __shared__ int s_any;

  int tid = threadIdx.x;
  if (tid == 0) {
    s_min = 255;
    s_max = 0;
    s_xmin = INT_MAX;
    s_ymin = INT_MAX;
    s_xmax = -1;
    s_ymax = -1;
    s_any = 0;
  }
  __syncthreads();

  // Pass 1: per-crop luminance min/max via shared-memory atomics.
  for (int i = tid; i < total; i += blockDim.x) {
    int dx = i % sw;
    int dy = i / sw;
    int gx = sx0 + dx;
    int gy = sy0 + dy;
    const uint8_t *p = page_data + gy * page_step + gx * 3;
    int li = luminance_bgr(__ldg(p), __ldg(p + 1), __ldg(p + 2));
    atomicMin(&s_min, li);
    atomicMax(&s_max, li);
  }
  __syncthreads();

  int min_v = s_min;
  int max_v = s_max;
  // Monochrome (matches PaddleX early return) -> full sub-rect bbox.
  if (min_v == max_v) {
    if (tid == 0) {
      d_bbox[b * 4 + 0] = 0;
      d_bbox[b * 4 + 1] = 0;
      d_bbox[b * 4 + 2] = sw;
      d_bbox[b * 4 + 3] = sh;
    }
    return;
  }

  // Pass 2: foreground mask via `min-max normalize then threshold at 200`
  // (`255 * (data < 200)` in PaddleX). bbox extrema accumulated via shared
  // atomicMin/Max.
  float span = static_cast<float>(max_v - min_v);
  for (int i = tid; i < total; i += blockDim.x) {
    int dx = i % sw;
    int dy = i / sw;
    int gx = sx0 + dx;
    int gy = sy0 + dy;
    const uint8_t *p = page_data + gy * page_step + gx * 3;
    int li = luminance_bgr(__ldg(p), __ldg(p + 1), __ldg(p + 2));
    float normed = (static_cast<float>(li - min_v) / span) * 255.0f;
    if (normed < kFormulaForegroundThreshold) {
      atomicMin(&s_xmin, dx);
      atomicMin(&s_ymin, dy);
      atomicMax(&s_xmax, dx);
      atomicMax(&s_ymax, dy);
      atomicAdd(&s_any, 1);
    }
  }
  __syncthreads();

  if (tid == 0) {
    if (s_any == 0) {
      // No foreground pixels found -> keep whole sub-rect (PaddleX fallback).
      d_bbox[b * 4 + 0] = 0;
      d_bbox[b * 4 + 1] = 0;
      d_bbox[b * 4 + 2] = sw;
      d_bbox[b * 4 + 3] = sh;
    } else {
      d_bbox[b * 4 + 0] = s_xmin;
      d_bbox[b * 4 + 1] = s_ymin;
      d_bbox[b * 4 + 2] = s_xmax - s_xmin + 1;
      d_bbox[b * 4 + 3] = s_ymax - s_ymin + 1;
    }
  }
}

// resize_short_then_thumbnail (PaddleX UniMERNetImgDecode): the shorter side
// scales to `target_short`, the longer side scales proportionally, then both
// sides are thumbnail-capped at `cap` (downscale-only).
__device__ __forceinline__ void resize_short_thumbnail(int cw, int ch,
                                                       int target_short,
                                                       int cap,
                                                       int &rw, int &rh) {
  cw = max(1, cw);
  ch = max(1, ch);
  if (cw <= ch) {
    rw = target_short;
    rh = static_cast<int>(roundf(static_cast<float>(target_short) *
                                  static_cast<float>(ch) /
                                  static_cast<float>(cw)));
  } else {
    rh = target_short;
    rw = static_cast<int>(roundf(static_cast<float>(target_short) *
                                  static_cast<float>(cw) /
                                  static_cast<float>(ch)));
  }
  float s_x = static_cast<float>(cap) / static_cast<float>(rw);
  float s_y = static_cast<float>(cap) / static_cast<float>(rh);
  float s = fminf(fminf(s_x, s_y), 1.0f);
  if (s < 1.0f) {
    rw = static_cast<int>(roundf(static_cast<float>(rw) * s));
    rh = static_cast<int>(roundf(static_cast<float>(rh) * s));
  }
  rw = max(1, rw);
  rh = max(1, rh);
}

// Fused bilinear sample + per-channel normalize + BT.601 BGR->GRAY collapse.
// Each thread writes one (b, dy, dx) of the 384x384 single-channel output.
// Block layout (32, 8, 1); grid (12, 48, B).
//
// Numerical identity: per-channel normalize with equal mean/std commutes
// with the BT.601 grayscale collapse (weights sum to 1), so we run the
// grayscale FIRST then a single scalar normalize -- this is bit-identical
// to the Rust reference (modulo floating-point assoc within one MAC chain).
__global__ __launch_bounds__(256)
void fused_formula_pre_kernel(const uint8_t * __restrict__ page_data,
                               int page_step, int page_rows, int page_cols,
                               const FormulaSubRect * __restrict__ d_subrects,
                               const int32_t * __restrict__ d_bbox,
                               float * __restrict__ d_output) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  int b  = blockIdx.z;
  if (dx >= kFormulaW || dy >= kFormulaH) return;

  FormulaSubRect rect = d_subrects[b];
  int sx0 = max(0, rect.x0);
  int sy0 = max(0, rect.y0);
  int sx1 = min(page_cols, rect.x0 + rect.w);
  int sy1 = min(page_rows, rect.y0 + rect.h);
  int sw = max(1, sx1 - sx0);
  int sh = max(1, sy1 - sy0);

  // Per-crop inner bbox (sub-rect-relative coords) computed by the bbox kernel.
  int cx = d_bbox[b * 4 + 0];
  int cy = d_bbox[b * 4 + 1];
  int cw = d_bbox[b * 4 + 2];
  int ch = d_bbox[b * 4 + 3];
  // Defensive clamp: a degenerate bbox would otherwise read out of bounds.
  cw = max(1, min(cw, sw - cx));
  ch = max(1, min(ch, sh - cy));

  int rw, rh;
  resize_short_thumbnail(cw, ch, kFormulaH, kFormulaW, rw, rh);

  int pad_x = (kFormulaW - rw) / 2;
  int pad_y = (kFormulaH - rh) / 2;

  int dst_off = b * kFormulaH * kFormulaW + dy * kFormulaW + dx;

  // Pad value: byte-domain 0 (PaddleX `ImageOps.expand(fill=0)`) post-normalize.
  const float pad_gray = (0.0f / 255.0f - kFormulaMean) / kFormulaStd;

  int in_x = dx - pad_x;
  int in_y = dy - pad_y;
  if (in_x < 0 || in_y < 0 || in_x >= rw || in_y >= rh) {
    d_output[dst_off] = pad_gray;
    return;
  }

  // Map resized-space (in_x, in_y) -> cropped src space (PIL/cv2 sample center).
  float scale_x = static_cast<float>(cw) / static_cast<float>(rw);
  float scale_y = static_cast<float>(ch) / static_cast<float>(rh);
  float src_x = (static_cast<float>(in_x) + 0.5f) * scale_x - 0.5f;
  float src_y = (static_cast<float>(in_y) + 0.5f) * scale_y - 0.5f;
  int x0 = static_cast<int>(floorf(src_x));
  int y0 = static_cast<int>(floorf(src_y));
  float fx = src_x - static_cast<float>(x0);
  float fy = src_y - static_cast<float>(y0);

  // Clamp into cropped region [0..cw-1] x [0..ch-1] (matches Rust crate's
  // PIL bilinear sample clamp), then offset into the page absolute coords.
  int xc0 = min(max(x0,     0), cw - 1) + cx + sx0;
  int xc1 = min(max(x0 + 1, 0), cw - 1) + cx + sx0;
  int yc0 = min(max(y0,     0), ch - 1) + cy + sy0;
  int yc1 = min(max(y0 + 1, 0), ch - 1) + cy + sy0;

  const uint8_t *row0 = page_data + yc0 * page_step;
  const uint8_t *row1 = page_data + yc1 * page_step;

  auto sample = [](const uint8_t *row, int gx) -> float3 {
    const uint8_t *p = row + gx * 3;
    // BGR byte order (matches page GpuImage layout).
    return make_float3(static_cast<float>(__ldg(p)),
                       static_cast<float>(__ldg(p + 1)),
                       static_cast<float>(__ldg(p + 2)));
  };
  float3 p00 = sample(row0, xc0);
  float3 p10 = sample(row0, xc1);
  float3 p01 = sample(row1, xc0);
  float3 p11 = sample(row1, xc1);

  float w00 = (1.0f - fx) * (1.0f - fy);
  float w10 = fx * (1.0f - fy);
  float w01 = (1.0f - fx) * fy;
  float w11 = fx * fy;

  float bb = w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x;
  float gg = w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y;
  float rr = w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z;

  // BT.601 grayscale of BGR.
  constexpr float W_B = 0.114f;
  constexpr float W_G = 0.587f;
  constexpr float W_R = 0.299f;
  float gray = W_B * bb + W_G * gg + W_R * rr;
  d_output[dst_off] = (gray / 255.0f - kFormulaMean) / kFormulaStd;
}

} // namespace

void cuda_fused_formula_pre(const GpuImage &page,
                            const FormulaSubRect *d_subrects,
                            int batch_size,
                            int32_t *d_bbox,
                            float *d_output,
                            cudaStream_t stream) {
  if (batch_size <= 0) return;

  const uint8_t *page_data = static_cast<const uint8_t *>(page.data);
  int page_step = static_cast<int>(page.step);
  int page_rows = page.rows;
  int page_cols = page.cols;

  // Phase 1: per-crop bbox (one block per crop, 256 threads).
  dim3 bbox_block(256, 1, 1);
  dim3 bbox_grid(1, 1, batch_size);
  formula_bbox_kernel<<<bbox_grid, bbox_block, 0, stream>>>(
      page_data, page_step, page_rows, page_cols,
      d_subrects, d_bbox);
  CUDA_CHECK(cudaGetLastError());

  // Phase 2: fused bilinear + normalize + grayscale into (B,1,384,384).
  dim3 pre_block(32, 8, 1);
  dim3 pre_grid((kFormulaW + pre_block.x - 1) / pre_block.x,
                (kFormulaH + pre_block.y - 1) / pre_block.y,
                batch_size);
  fused_formula_pre_kernel<<<pre_grid, pre_block, 0, stream>>>(
      page_data, page_step, page_rows, page_cols,
      d_subrects, d_bbox, d_output);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace turbo_ocr::formula::kernels
