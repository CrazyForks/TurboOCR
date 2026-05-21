#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// SHA-256 via OpenSSL (always present — Drogon links it transitively).
#include <openssl/sha.h>

namespace turbo_ocr {

/// Compute SHA-256 of arbitrary bytes and return the result as a 64-char
/// lowercase hex string. Suitable for cache keys.
[[nodiscard]] inline std::string sha256_hex(const void *data, size_t len) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(static_cast<const unsigned char *>(data), len, digest);

  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(SHA256_DIGEST_LENGTH * 2);
  for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
    out += kHex[(digest[i] >> 4) & 0xF];
    out += kHex[digest[i] & 0xF];
  }
  return out;
}

} // namespace turbo_ocr
