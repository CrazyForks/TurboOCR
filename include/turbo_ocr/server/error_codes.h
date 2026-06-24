#pragma once

#include <array>
#include <cstddef>
#include <string_view>

// Single source of truth for the structured error codes both transports emit:
// HTTP returns them inside {"error":{"code":"..."}}, gRPC stamps them into the
// "x-error-code" trailing metadata. Each code carries the HTTP status and the
// gRPC StatusCode it maps to so the two surfaces stay in lockstep — change a
// mapping here and both adopt it.
//
// The gRPC status is stored as the canonical grpc::StatusCode integer value so
// this header has no dependency on <grpcpp/grpcpp.h> and compiles standalone.
// Callers that already include grpcpp can pass the int straight into
// grpc::Status / static_cast it to grpc::StatusCode; grpc_status_code() is a
// thin, optionally-compiled convenience for that cast.

namespace turbo_ocr::server {

enum class ErrorCode {
  kMissingImage,
  kMissingPdf,
  kMissingFile,
  kMissingHeader,
  kInvalidHeader,
  kInvalidJson,
  kInvalidMultipart,
  kInvalidDpi,
  kInvalidDimensions,
  kInvalidParameter,
  kBase64DecodeFailed,
  kImageDecodeFailed,
  kDimensionsTooLarge,
  kBodySizeMismatch,
  kEmptyBody,
  kEmptyBatch,
  kBatchTooLarge,
  kEmptyPdf,
  kPdfTooLarge,
  kPdfRenderFailed,
  kAutorotateDisabled,
  kLayoutDisabled,
  kNotReady,
  kServerBusy,
  kPdfNotAvailable,
  kInferenceError,
  kPageDecodeFailed,
  kInferenceTimeout,
};

// Canonical grpc::StatusCode integer values (grpc/status.h), inlined so this
// header stays free of the grpcpp dependency.
enum class GrpcStatusValue : int {
  kOk = 0,
  kCancelled = 1,
  kUnknown = 2,
  kInvalidArgument = 3,
  kDeadlineExceeded = 4,
  kNotFound = 5,
  kAlreadyExists = 6,
  kPermissionDenied = 7,
  kResourceExhausted = 8,
  kFailedPrecondition = 9,
  kAborted = 10,
  kOutOfRange = 11,
  kUnimplemented = 12,
  kInternal = 13,
  kUnavailable = 14,
  kDataLoss = 15,
  kUnauthenticated = 16,
};

struct ErrorCodeEntry {
  ErrorCode      code = ErrorCode::kInferenceError;
  std::string_view name;   // wire string ("MISSING_IMAGE", …)
  int            http_status = 500;   // HTTP status code (e.g. 400, 503, 504)
  GrpcStatusValue grpc_status = GrpcStatusValue::kInternal;
};

// One row per ErrorCode, declaration order matching the enum so an index lookup
// is O(1). HTTP/gRPC mappings reproduce the literals previously scattered
// across the routes, grpc_service.h, server_types.h and proto/ocr.proto.
inline constexpr std::array<ErrorCodeEntry, 28> kErrorCodeTable{{
    {ErrorCode::kMissingImage,       "MISSING_IMAGE",        400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kMissingPdf,         "MISSING_PDF",          400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kMissingFile,        "MISSING_FILE",         400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kMissingHeader,      "MISSING_HEADER",       400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidHeader,      "INVALID_HEADER",       400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidJson,        "INVALID_JSON",         400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidMultipart,   "INVALID_MULTIPART",    400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidDpi,         "INVALID_DPI",          400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidDimensions,  "INVALID_DIMENSIONS",   400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidParameter,   "INVALID_PARAMETER",    400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kBase64DecodeFailed, "BASE64_DECODE_FAILED", 400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kImageDecodeFailed,  "IMAGE_DECODE_FAILED",  400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kDimensionsTooLarge, "DIMENSIONS_TOO_LARGE", 400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kBodySizeMismatch,   "BODY_SIZE_MISMATCH",   400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kEmptyBody,          "EMPTY_BODY",           400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kEmptyBatch,         "EMPTY_BATCH",          400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kBatchTooLarge,      "BATCH_TOO_LARGE",      400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kEmptyPdf,           "EMPTY_PDF",            400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kPdfTooLarge,        "PDF_TOO_LARGE",        400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kPdfRenderFailed,    "PDF_RENDER_FAILED",    400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kAutorotateDisabled, "AUTOROTATE_DISABLED",  400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kLayoutDisabled,     "LAYOUT_DISABLED",      400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kNotReady,           "NOT_READY",            503, GrpcStatusValue::kUnavailable},
    {ErrorCode::kServerBusy,         "SERVER_BUSY",          503, GrpcStatusValue::kResourceExhausted},
    {ErrorCode::kPdfNotAvailable,    "PDF_NOT_AVAILABLE",    501, GrpcStatusValue::kUnimplemented},
    {ErrorCode::kInferenceError,     "INFERENCE_ERROR",      500, GrpcStatusValue::kInternal},
    {ErrorCode::kPageDecodeFailed,   "PAGE_DECODE_FAILED",   500, GrpcStatusValue::kInternal},
    {ErrorCode::kInferenceTimeout,   "INFERENCE_TIMEOUT",    504, GrpcStatusValue::kDeadlineExceeded},
}};

[[nodiscard]] inline constexpr const ErrorCodeEntry &error_entry(ErrorCode code) {
  // Index by enum value; the table is declared in enum order.
  return kErrorCodeTable[static_cast<std::size_t>(code)];
}

// Wire string for a code, e.g. "MISSING_IMAGE". The returned view points into
// the static table and is valid for the program lifetime.
[[nodiscard]] inline constexpr std::string_view error_code_str(ErrorCode code) {
  return error_entry(code).name;
}

[[nodiscard]] inline constexpr int error_http_status(ErrorCode code) {
  return error_entry(code).http_status;
}

// gRPC status as the canonical grpc::StatusCode integer value. Callers holding
// grpcpp can static_cast<grpc::StatusCode>(...) it.
[[nodiscard]] inline constexpr int error_grpc_status_value(ErrorCode code) {
  return static_cast<int>(error_entry(code).grpc_status);
}

} // namespace turbo_ocr::server

#ifdef GRPCPP_GRPCPP_H
// Only available where grpcpp is already in scope; keeps this header standalone
// for translation units that never touch gRPC.
namespace turbo_ocr::server {
[[nodiscard]] inline grpc::StatusCode error_grpc_status(ErrorCode code) {
  return static_cast<grpc::StatusCode>(error_grpc_status_value(code));
}
} // namespace turbo_ocr::server
#endif
