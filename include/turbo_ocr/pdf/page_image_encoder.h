#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace turbo_ocr::pdf {

/// Image formats supported for page image encoding.
enum class PageImageFormat {
  Jpeg,
  Png,
  WebP,
};

/// Parse format string (case-insensitive). Returns Jpeg on unknown.
PageImageFormat parse_page_image_format(const char *s) noexcept;

/// Human-readable format name used in Content-Type and URL paths.
const char *page_image_format_name(PageImageFormat fmt) noexcept;
const char *page_image_content_type(PageImageFormat fmt) noexcept;

struct EncodeOptions {
  PageImageFormat format  = PageImageFormat::Jpeg;
  int             quality = 85;   // JPEG/WebP quality (1–100)
  int             max_side = 0;   // 0 = no resize; >0 = fit within max_side px
};

/// Encode a BGR cv::Mat to compressed bytes.
/// Uses libjpeg-turbo for JPEG (fastest path), OpenCV imencode for PNG/WebP.
/// Returns empty vector on failure.
[[nodiscard]] std::vector<uint8_t>
encode_page_image(const cv::Mat &bgr, const EncodeOptions &opts);

} // namespace turbo_ocr::pdf
