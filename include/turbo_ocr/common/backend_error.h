#pragma once
#include <stdexcept>

namespace turbo_ocr {

// An EXPLICITLY requested table/formula backend (named registry pick or inline
// spec) was null or not ready — distinct from "no table/formula in this image".
// Routes map it to 503 BACKEND_UNAVAILABLE so a failed-to-load backend can't
// masquerade as an empty-but-successful result.
struct BackendUnavailableError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

} // namespace turbo_ocr
