#pragma once
#include <stdexcept>
#include <string>

namespace turbo_ocr {

class OcrError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class ModelLoadError : public OcrError {
  using OcrError::OcrError;
};

class InferenceError : public OcrError {
  using OcrError::OcrError;
};

class CudaError : public OcrError {
  using OcrError::OcrError;
};

class PoolExhaustedError : public OcrError {
public:
  PoolExhaustedError() : OcrError("Pipeline pool exhausted (timeout)") {}
  explicit PoolExhaustedError(const std::string &msg) : OcrError(msg) {}
};

class ImageDecodeError : public OcrError {
  using OcrError::OcrError;
};

// Decoded/declared image dimensions exceed MAX_IMAGE_DIM. Distinct from
// ImageDecodeError so the routes can map it to the stable DIMENSIONS_TOO_LARGE
// code (same as the pre/post-decode sniffs) instead of IMAGE_DECODE_FAILED.
class ImageTooLargeError : public OcrError {
  using OcrError::OcrError;
};

class PdfRenderError : public OcrError {
  using OcrError::OcrError;
};

} // namespace turbo_ocr
