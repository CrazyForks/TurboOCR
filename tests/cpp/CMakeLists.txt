# NOT the canonical test build.
#
# The real C++ test target `turbo_ocr_tests` is defined in the ROOT
# CMakeLists.txt and is registered with CTest (`add_test(turbo_ocr_unit ...)`).
# It compiles the full suite — including the migration tests
# (test_server_config, test_dict_resolution, test_docassembly,
# test_language_paths, test_pdf_renderer_liveness) that this standalone file
# historically omitted, which made a build from here silently miss coverage.
#
# Build + run the tests from the repo root instead:
#   cmake -S . -B build -DNATIVE_ARCH=OFF && cmake --build build --target turbo_ocr_tests
#   ctest --test-dir build --output-on-failure
#
# This file is intentionally a no-op to avoid resurrecting the partial suite.
