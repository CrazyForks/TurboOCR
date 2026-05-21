#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace turbo_ocr::pdf {

/// Thread-safe LRU cache for encoded page images.
///
/// Key:   sha256_hex + ':' + page_idx + ':' + format + ':' + dpi
/// Value: compressed image bytes (JPEG/PNG/WebP)
///
/// Eviction policy: LRU by byte size.
/// Max capacity: IMG_CACHE_MB env var (default 1024 MiB).
///
/// Read-heavy pattern: shared_mutex allows concurrent readers (GET hits)
/// without blocking each other. Writers (cache fills) take exclusive locks.
class PageImageCache {
public:
  explicit PageImageCache(size_t max_bytes);

  /// Singleton — one cache shared by all routes.
  static PageImageCache &instance();

  /// Build a cache key from its components.
  static std::string make_key(const std::string &pdf_sha,
                               int page_idx,
                               const char *format,
                               int dpi);

  /// Look up encoded bytes. Returns nullopt on miss.
  [[nodiscard]] std::optional<std::vector<uint8_t>> get(const std::string &key) const;

  /// Insert encoded bytes. Evicts LRU entries as needed to stay within quota.
  void put(const std::string &key, std::vector<uint8_t> value);

  /// Current total cached bytes.
  [[nodiscard]] size_t current_bytes() const;

  /// Maximum cache capacity in bytes.
  [[nodiscard]] size_t max_bytes() const { return max_bytes_; }

private:
  using List = std::list<std::pair<std::string, std::vector<uint8_t>>>;
  using Map  = std::unordered_map<std::string, List::iterator>;

  mutable std::shared_mutex mutex_;
  List lru_;           // front = MRU, back = LRU
  Map  index_;
  size_t current_bytes_{0};
  size_t max_bytes_;

  // Must be called with exclusive lock held.
  void evict_to_fit(size_t incoming_bytes);
};

} // namespace turbo_ocr::pdf
