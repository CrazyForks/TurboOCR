// GPU-only safety unit tests (C4 per-request timeout primitive, H4 sticky-CUDA
// fault classifier). Guarded: both headers pull in CUDA/TRT, so this file is
// empty in the CPU build and its tests run under the GPU build's ctest.
#include <catch_amalgamated.hpp>

#ifndef USE_CPU_ONLY

#include <future>
#include <thread>

#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/pipeline/pipeline_dispatcher.h"

using turbo_ocr::pipeline::get_with_timeout;

TEST_CASE("C4: get_with_timeout throws TimeoutError on deadline overrun", "[c4][timeout]") {
  // A promise that is never fulfilled models a wedged GPU worker: the wait must
  // give up at the deadline and throw (so the route maps it to 504), not block.
  std::promise<int> p;
  auto fut = p.get_future();
  REQUIRE_THROWS_AS(get_with_timeout(fut, /*timeout_ms=*/30), turbo_ocr::TimeoutError);
}

TEST_CASE("C4: get_with_timeout returns the value when it resolves in time", "[c4][timeout]") {
  std::promise<int> p;
  auto fut = p.get_future();
  std::thread producer([&p] {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    p.set_value(7);
  });
  REQUIRE(get_with_timeout(fut, /*timeout_ms=*/2000) == 7);
  producer.join();
}

TEST_CASE("H4: is_sticky_cuda_error flags context-poisoning faults only", "[h4][cuda]") {
  // Sticky faults poison the whole CUDA context -> must fail-fast for restart.
  REQUIRE(turbo_ocr::is_sticky_cuda_error(cudaErrorIllegalAddress));
  REQUIRE(turbo_ocr::is_sticky_cuda_error(cudaErrorLaunchFailure));
  REQUIRE(turbo_ocr::is_sticky_cuda_error(cudaErrorMisalignedAddress));
  // Recoverable / benign codes must NOT trip the fail-fast.
  REQUIRE_FALSE(turbo_ocr::is_sticky_cuda_error(cudaSuccess));
  REQUIRE_FALSE(turbo_ocr::is_sticky_cuda_error(cudaErrorInvalidValue));
}

#endif  // !USE_CPU_ONLY
