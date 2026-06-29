#pragma once

#include <cstddef>
#include <exception>
#include <format>
#include <optional>
#include <string>

#include <drogon/HttpRequest.h>
#include <drogon/HttpResponse.h>

namespace turbo_ocr::server {

// Resolved /ocr/pixels dimensions, or a 400 to emit. `ok()` is false when
// error_code is set; otherwise width/height/channels are valid.
struct PixelDims {
  int width = 0;
  int height = 0;
  int channels = 3;
  bool used_legacy_header = false;  // a request carried an X-* dim header
  std::string error_code;           // empty => ok
  std::string error;                // human-readable message
  [[nodiscard]] bool ok() const { return error_code.empty(); }
};

// Resolve width/height/channels from query params (preferred, consistent with
// the rest of the API) with the legacy X-Width/X-Height/X-Channels headers as a
// v2.3-compat fallback. A value supplied in BOTH that disagrees -> 400
// DIMENSION_CONFLICT (fail loud, never silently pick one). Shared verbatim by
// the GPU and CPU /ocr/pixels handlers so the contract can't drift between them.
[[nodiscard]] inline PixelDims
resolve_pixel_dims(const drogon::HttpRequestPtr &req) {
  PixelDims d;
  std::string parse_err, conflict_err;
  bool legacy = false;
  auto resolve = [&](const char *qname, const char *hname,
                     std::optional<int> &out) {
    const std::string q = req->getParameter(qname);
    const std::string h = req->getHeader(hname);
    std::optional<int> qv, hv;
    auto parse = [&](const std::string &s, std::optional<int> &o) -> bool {
      if (s.empty()) return true;
      try {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) return false;  // reject trailing junk ("12abc")
        o = v;
        return true;
      } catch (const std::exception &) { return false; }
    };
    if (!parse(q, qv) || !parse(h, hv)) {
      if (parse_err.empty())  // keep the first malformed param's message
        parse_err = std::string("Invalid ") + qname + " value";
      return;
    }
    if (qv && hv && *qv != *hv)
      conflict_err = std::format(
          "Conflicting {0}: query ?{0}={1} disagrees with {2} header {3}",
          qname, q, hname, h);
    if (hv) legacy = true;
    out = qv ? qv : hv;
  };
  std::optional<int> ow, oh, oc;
  resolve("width",    "X-Width",    ow);
  resolve("height",   "X-Height",   oh);
  resolve("channels", "X-Channels", oc);
  d.used_legacy_header = legacy;
  if (!parse_err.empty()) {
    d.error_code = "INVALID_DIMENSIONS";
    d.error = parse_err;
    return d;
  }
  if (!conflict_err.empty()) {
    d.error_code = "DIMENSION_CONFLICT";
    d.error = conflict_err;
    return d;
  }
  if (!ow || !oh) {
    d.error_code = "MISSING_DIMENSIONS";
    d.error = "Missing width/height (use ?width=&height= query params or the "
              "X-Width/X-Height headers)";
    return d;
  }
  d.width = *ow;
  d.height = *oh;
  d.channels = oc.value_or(3);
  if (d.width <= 0 || d.height <= 0 || (d.channels != 1 && d.channels != 3)) {
    d.error_code = "INVALID_DIMENSIONS";
    d.error = "Invalid dimensions or channels";
    return d;
  }
  return d;
}

// RFC 8594 deprecation signal for clients still passing the legacy X-* dim
// headers. Stamped on the success response so the migration to query params is
// discoverable without breaking the v2.3 contract. A custom advisory header
// carries the actionable hint (RFC 8594 leaves the body/link to the operator).
inline void stamp_pixel_dim_deprecation(const drogon::HttpResponsePtr &resp) {
  resp->addHeader("Deprecation", "true");
  resp->addHeader("X-Deprecation-Notice",
                  "X-Width/X-Height/X-Channels headers are deprecated; pass "
                  "?width=&height=&channels= query params instead");
}

}  // namespace turbo_ocr::server
