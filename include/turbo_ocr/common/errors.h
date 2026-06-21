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

// A blocking wait (e.g. dispatcher.submit_for_default()) exceeded its
// per-request deadline. Mirrors PoolExhaustedError so routes/gRPC map it to a
// stable timeout status (504 / DEADLINE_EXCEEDED). Lives here (not a GPU-only
// header) so the CPU build's catch clauses see it too.
class TimeoutError : public OcrError {
public:
  TimeoutError() : OcrError("Inference timed out") {}
  explicit TimeoutError(const std::string &msg) : OcrError(msg) {}
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
