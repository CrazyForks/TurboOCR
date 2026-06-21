// Fused preprocess kernels for the table-detection pipeline.
// Reads sub-rects of a single page GpuImage (BGR uint8), produces normalized
// CHW float tensors directly into TRT input buffers — no intermediate copies.
//
// Three kernels (all declared in turbo_ocr/kernels/kernels.h):
//   - cuda_fused_table_cls_pre          : resize-short(256) + center-crop(224)
//                                         + ImageNet normalize + CHW
//   - cuda_fused_slanext_pre            : ResizeByLong(488) preserve AR +
//                                         ImageNet normalize + bottom-right
//                                         pad to 488×488 + CHW
//   - cuda_fused_resize_normalize_layout (sub-rect overload) : resize sub-rect
//                                         to dst_w × dst_h, /255 normalize, CHW
//
// Channel order: BGR throughout (matches PaddleOCR img_mode and the existing
// det / layout kernels). Pad value for SLANeXt is `(0 - mean) / std` per
// PaddleX export.

#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/common/cuda_check.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace turbo_ocr::kernels {

// Bilinear sample within a sub-rect of a BGR uint8 image, normalize per channel.
// (sx_rel, sy_rel) are in source-pixel coords RELATIVE to the sub-rect's
// top-left. Clamp to sub-rect bounds.
__device__ __forceinline__ void bilinear_sample_norm(
    const unsigned char* __restrict__ src, int src_step,
    int rect_x, int rect_y, int rect_w, int rect_h,
    float sx_rel, float sy_rel,
    float mean0, float mean1, float mean2,
    float inv_std0, float inv_std1, float inv_std2, float inv_255,
    float& b_out, float& g_out, float& r_out) {
  int x0 = (int)floorf(sx_rel);
  int y0 = (int)floorf(sy_rel);
  float fx = sx_rel - x0;
  float fy = sy_rel - y0;
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  x0 = max(0, min(x0, rect_w - 1));
  x1 = max(0, min(x1, rect_w - 1));
  y0 = max(0, min(y0, rect_h - 1));
  y1 = max(0, min(y1, rect_h - 1));

  int ax0 = rect_x + x0;
  int ax1 = rect_x + x1;
  int ay0 = rect_y + y0;
  int ay1 = rect_y + y1;

  const unsigned char* row0 = src + ay0 * src_step;
  const unsigned char* row1 = src + ay1 * src_step;

  auto ldg3 = [](const unsigned char* base, int px) -> uchar3 {
    const unsigned char* p = base + px * 3;
    return make_uchar3(__ldg(p), __ldg(p + 1), __ldg(p + 2));
  };

  uchar3 p00 = ldg3(row0, ax0), p10 = ldg3(row0, ax1);
  uchar3 p01 = ldg3(row1, ax0), p11 = ldg3(row1, ax1);

  float w00 = (1.0f - fx) * (1.0f - fy);
  float w10 = fx * (1.0f - fy);
  float w01 = (1.0f - fx) * fy;
  float w11 = fx * fy;

  float b = w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x;
  float g = w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y;
  float r = w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z;

  b_out = (b * inv_255 - mean0) * inv_std0;
  g_out = (g * inv_255 - mean1) * inv_std1;
  r_out = (r * inv_255 - mean2) * inv_std2;
}

// =================================================================
// TableCls: resize-short(256) + center-crop(224) + ImageNet + CHW
// =================================================================
//
// Maths (mirrors PaddleX ResizeImage(short_size=256) + CropImage(224)):
//   s          = 256 / min(rect_w, rect_h)              // resize scale
//   R_w, R_h   = (s*rect_w, s*rect_h)                   // resized dims
//   crop_off_x = (R_w - 224) * 0.5                      // resized-px crop offset
//   For each dst pixel (dx, dy) in [0, 224):
//     rx_resized = dx + crop_off_x
//     sx_src     = (rx_resized + 0.5) / s - 0.5         // OpenCV INTER_LINEAR
//                = (dx + 0.5) / s + crop_off_x / s - 0.5
//   Pass `scale = 1/s` and `crop_off_x_src = (rect_w - 224*scale) * 0.5`
//   so kernel does: sx_rel = (dx + 0.5) * scale - 0.5 + crop_off_x_src

__global__ __launch_bounds__(256)
void table_cls_pre_kernel(
    const unsigned char* __restrict__ src, int src_step,
    int rect_x, int rect_y, int rect_w, int rect_h,
    float* __restrict__ dst_chw, int dst,
    float scale,
    float crop_off_x_src, float crop_off_y_src,
    float mean0, float mean1, float mean2,
    float inv_std0, float inv_std1, float inv_std2, float inv_255) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst || dy >= dst) return;

  float sx_rel = (dx + 0.5f) * scale - 0.5f + crop_off_x_src;
  float sy_rel = (dy + 0.5f) * scale - 0.5f + crop_off_y_src;

  float b, g, r;
  bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h,
                       sx_rel, sy_rel,
                       mean0, mean1, mean2,
                       inv_std0, inv_std1, inv_std2, inv_255,
                       b, g, r);

  int idx = dy * dst + dx;
  int plane = dst * dst;
  dst_chw[0 * plane + idx] = b;
  dst_chw[1 * plane + idx] = g;
  dst_chw[2 * plane + idx] = r;
}

void cuda_fused_table_cls_pre(const GpuImage& src,
                              int rect_x, int rect_y, int rect_w, int rect_h,
                              float* dst_chw, cudaStream_t stream) {
  constexpr int DST = 224;
  constexpr int SHORT = 256;
  float min_side = (float)::min(rect_w, rect_h);
  float scale = min_side / (float)SHORT;          // src px per resized px
  float crop_off_x_src = ((float)rect_w - DST * scale) * 0.5f;
  float crop_off_y_src = ((float)rect_h - DST * scale) * 0.5f;

  dim3 block(32, 8);
  dim3 grid((DST + block.x - 1) / block.x, (DST + block.y - 1) / block.y);
  table_cls_pre_kernel<<<grid, block, 0, stream>>>(
      (const unsigned char*)src.data, (int)src.step,
      rect_x, rect_y, rect_w, rect_h,
      dst_chw, DST,
      scale, crop_off_x_src, crop_off_y_src,
      0.485f, 0.456f, 0.406f,
      1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f, 1.0f / 255.0f);
  CUDA_CHECK(cudaGetLastError());
}

// =================================================================
// SLANeXt: ResizeByLong(488) preserve AR + ImageNet + bottom-right pad + CHW
// =================================================================
//
// Maths:
//   s        = 488 / max(rect_w, rect_h)
//   new_w    = round(rect_w * s); new_h = round(rect_h * s)
//   scale    = 1 / s
//   For dst pixel (dx, dy) in [0, 488):
//     if dx < new_w && dy < new_h:
//        sx_rel = (dx + 0.5) * scale - 0.5
//        sy_rel = (dy + 0.5) * scale - 0.5
//        sample + normalize from sub-rect
//     else:
//        write PAD_VALUE = (0 - mean) / std

__global__ __launch_bounds__(256)
void slanext_pre_kernel(
    const unsigned char* __restrict__ src, int src_step,
    int rect_x, int rect_y, int rect_w, int rect_h,
    float* __restrict__ dst_chw, int dst,
    int new_w, int new_h, float scale,
    float mean0, float mean1, float mean2,
    float inv_std0, float inv_std1, float inv_std2, float inv_255) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst || dy >= dst) return;

  int idx = dy * dst + dx;
  int plane = dst * dst;

  if (dx < new_w && dy < new_h) {
    float sx_rel = (dx + 0.5f) * scale - 0.5f;
    float sy_rel = (dy + 0.5f) * scale - 0.5f;
    float b, g, r;
    bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h,
                         sx_rel, sy_rel,
                         mean0, mean1, mean2,
                         inv_std0, inv_std1, inv_std2, inv_255,
                         b, g, r);
    dst_chw[0 * plane + idx] = b;
    dst_chw[1 * plane + idx] = g;
    dst_chw[2 * plane + idx] = r;
  } else {
    dst_chw[0 * plane + idx] = (-mean0) * inv_std0;
    dst_chw[1 * plane + idx] = (-mean1) * inv_std1;
    dst_chw[2 * plane + idx] = (-mean2) * inv_std2;
  }
}

void cuda_fused_slanext_pre(const GpuImage& src,
                            int rect_x, int rect_y, int rect_w, int rect_h,
                            float* dst_chw, cudaStream_t stream) {
  constexpr int DST = 488;
  float long_side = (float)::max(rect_w, rect_h);
  float s = (float)DST / long_side;
  int new_w = (int)lroundf((float)rect_w * s);
  int new_h = (int)lroundf((float)rect_h * s);
  if (new_w > DST) new_w = DST;
  if (new_h > DST) new_h = DST;
  if (new_w < 1) new_w = 1;
  if (new_h < 1) new_h = 1;
  float scale = 1.0f / s;                          // src px per resized px

  dim3 block(32, 8);
  dim3 grid((DST + block.x - 1) / block.x, (DST + block.y - 1) / block.y);
  slanext_pre_kernel<<<grid, block, 0, stream>>>(
      (const unsigned char*)src.data, (int)src.step,
      rect_x, rect_y, rect_w, rect_h,
      dst_chw, DST, new_w, new_h, scale,
      0.485f, 0.456f, 0.406f,
      1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f, 1.0f / 255.0f);
  CUDA_CHECK(cudaGetLastError());
}

// =================================================================
// Nemotron table-structure (YOLOX) preprocess: letterbox the sub-rect to a
// 1024×1024 canvas (resize-by-long-side, top-left, pad 114), RAW 0-255 BGR
// values (YOLOX takes unnormalized input), CHW. The top-left letterbox with
// scale = 1024/max(w,h) is exactly what decode_nemotron inverts via
// ratio = min(1024/h, 1024/w).
// =================================================================
__global__ __launch_bounds__(256)
void nemotron_pre_kernel(
    const unsigned char* __restrict__ src, int src_step,
    int rect_x, int rect_y, int rect_w, int rect_h,
    float* __restrict__ dst_chw, int dst,
    int new_w, int new_h, float scale) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst || dy >= dst) return;
  int idx = dy * dst + dx;
  int plane = dst * dst;
  if (dx < new_w && dy < new_h) {
    float sx_rel = (dx + 0.5f) * scale - 0.5f;
    float sy_rel = (dy + 0.5f) * scale - 0.5f;
    float b, g, r;
    bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h,
                         sx_rel, sy_rel,
                         0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                         b, g, r);
    dst_chw[0 * plane + idx] = b;
    dst_chw[1 * plane + idx] = g;
    dst_chw[2 * plane + idx] = r;
  } else {
    dst_chw[0 * plane + idx] = 114.0f;
    dst_chw[1 * plane + idx] = 114.0f;
    dst_chw[2 * plane + idx] = 114.0f;
  }
}

void cuda_fused_nemotron_pre(const GpuImage& src,
                             int rect_x, int rect_y, int rect_w, int rect_h,
                             float* dst_chw, cudaStream_t stream) {
  constexpr int DST = 1024;
  float long_side = (float)::max(rect_w, rect_h);
  float s = (float)DST / long_side;
  int new_w = (int)lroundf((float)rect_w * s);
  int new_h = (int)lroundf((float)rect_h * s);
  if (new_w > DST) new_w = DST;
  if (new_h > DST) new_h = DST;
  if (new_w < 1) new_w = 1;
  if (new_h < 1) new_h = 1;
  float scale = 1.0f / s;                          // src px per resized px

  dim3 block(32, 8);
  dim3 grid((DST + block.x - 1) / block.x, (DST + block.y - 1) / block.y);
  nemotron_pre_kernel<<<grid, block, 0, stream>>>(
      (const unsigned char*)src.data, (int)src.step,
      rect_x, rect_y, rect_w, rect_h,
      dst_chw, DST, new_w, new_h, scale);
  CUDA_CHECK(cudaGetLastError());
}

// =================================================================
// Sub-rect overload of layout preprocess (used by cell-det).
// Same maths as the page-wide version in src/kernels/kernels.cu:
//   resize sub-rect to dst_w × dst_h, /255, mean=0, std=1, BGR CHW.
// =================================================================

__global__ __launch_bounds__(256)
void layout_subrect_pre_kernel(
    const unsigned char* __restrict__ src, int src_step,
    int rect_x, int rect_y, int rect_w, int rect_h,
    float* __restrict__ dst_chw, int dst_h, int dst_w,
    float scale_x, float scale_y, float inv_255) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst_w || dy >= dst_h) return;

  float sx_rel = (dx + 0.5f) * scale_x - 0.5f;
  float sy_rel = (dy + 0.5f) * scale_y - 0.5f;
  float b, g, r;
  bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h,
                       sx_rel, sy_rel,
                       0.0f, 0.0f, 0.0f,
                       1.0f, 1.0f, 1.0f, inv_255,
                       b, g, r);
  int idx = dy * dst_w + dx;
  int plane = dst_h * dst_w;
  dst_chw[0 * plane + idx] = b;
  dst_chw[1 * plane + idx] = g;
  dst_chw[2 * plane + idx] = r;
}

void cuda_fused_resize_normalize_layout(const GpuImage& src,
                                        int rect_x, int rect_y,
                                        int rect_w, int rect_h,
                                        float* dst_chw, int dst_w, int dst_h,
                                        cudaStream_t stream) {
  float scale_x = (float)rect_w / (float)dst_w;
  float scale_y = (float)rect_h / (float)dst_h;
  dim3 block(32, 8);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  layout_subrect_pre_kernel<<<grid, block, 0, stream>>>(
      (const unsigned char*)src.data, (int)src.step,
      rect_x, rect_y, rect_w, rect_h,
      dst_chw, dst_h, dst_w, scale_x, scale_y, 1.0f / 255.0f);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace turbo_ocr::kernels
