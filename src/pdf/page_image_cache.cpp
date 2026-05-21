#include "turbo_ocr/pdf/page_image_cache.h"

#include <cstdlib>
#include <format>
#include <mutex>
#include <shared_mutex>

namespace turbo_ocr::pdf {

namespace {
size_t default_max_bytes() {
  if (const char *env = std::getenv("IMG_CACHE_MB")) {
    int mb = std::atoi(env);
    if (mb > 0) return static_cast<size_t>(mb) * 1024 * 1024;
  }
  return 1024ULL * 1024 * 1024; // 1 GiB default
}
} // namespace

PageImageCache &PageImageCache::instance() {
  static PageImageCache cache(default_max_bytes());
  return cache;
}

PageImageCache::PageImageCache(size_t max_bytes) : max_bytes_(max_bytes) {}

std::string PageImageCache::make_key(const std::string &pdf_sha,
                                      int page_idx,
                                      const char *format,
                                      int dpi) {
  return std::format("{}:{}:{}:{}", pdf_sha, page_idx, format, dpi);
}

std::optional<std::vector<uint8_t>>
PageImageCache::get(const std::string &key) const {
  // Optimistic shared read — no write needed on a miss.
  {
    std::shared_lock lock(mutex_);
    auto it = index_.find(key);
    if (it == index_.end()) return std::nullopt;
    // Return a copy. We cannot promote to MRU here (need exclusive lock).
    // The "miss-but-already-hot" case just fails the MRU bump — acceptable.
    return it->second->second;
  }
}

void PageImageCache::put(const std::string &key, std::vector<uint8_t> value) {
  if (value.empty()) return;
  size_t incoming = value.size();
  if (incoming > max_bytes_) return; // single entry exceeds whole cache

  std::unique_lock lock(mutex_);

  // If key already present, remove old entry first.
  auto existing = index_.find(key);
  if (existing != index_.end()) {
    current_bytes_ -= existing->second->second.size();
    lru_.erase(existing->second);
    index_.erase(existing);
  }

  evict_to_fit(incoming);

  lru_.push_front({key, std::move(value)});
  index_[key] = lru_.begin();
  current_bytes_ += incoming;
}

size_t PageImageCache::current_bytes() const {
  std::shared_lock lock(mutex_);
  return current_bytes_;
}

void PageImageCache::evict_to_fit(size_t incoming_bytes) {
  // Evict from the LRU tail until we have room.
  while (!lru_.empty() && current_bytes_ + incoming_bytes > max_bytes_) {
    auto &back = lru_.back();
    current_bytes_ -= back.second.size();
    index_.erase(back.first);
    lru_.pop_back();
  }
}

} // namespace turbo_ocr::pdf
