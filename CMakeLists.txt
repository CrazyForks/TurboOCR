cmake_minimum_required(VERSION 3.20)

# Default to Release when the user didn't pick a build type. Single-config
# generators (Ninja/Make) leave CMAKE_BUILD_TYPE empty otherwise, which gives
# unoptimized builds.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Defaulting CMAKE_BUILD_TYPE to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

project(turbo_ocr VERSION 1.0.0 LANGUAGES CXX)

# /opt/cuda ships its own protobuf cmake config. When gRPC's
# find_dependency(Protobuf) resolves to it instead of the system install, its
# protobuf-targets.cmake tries to redefine targets the system FindProtobuf
# module already created ("Some (but not all) targets in this export set were
# already defined"). Exclude just that directory so protobuf + gRPC both come
# from one consistent (system) source; CUDAToolkit discovery is unaffected.
list(APPEND CMAKE_IGNORE_PATH "/opt/cuda/lib/cmake/protobuf")

# Protobuf discovery that works on BOTH distros we build on:
#   - Arch dev host: ships protobuf-config.cmake (v35 + libupb). CONFIG mode
#     is required there so the target set matches gRPC's config-mode protobuf
#     ("some but not all targets already defined" otherwise).
#   - Ubuntu Docker images (Dockerfile.cpu/gpu): apt libprotobuf-dev ships NO
#     cmake config, so CONFIG mode fails hard — but the legacy FindProtobuf
#     MODULE works (and gRPC there has no config either, so it uses the
#     pkg-config fallback below and never re-finds protobuf in config mode,
#     meaning no target clash). Prefer CONFIG, fall back to MODULE.
macro(turbo_find_protobuf)
  find_package(Protobuf CONFIG QUIET)
  if(NOT Protobuf_FOUND)
    find_package(Protobuf MODULE REQUIRED)
  endif()
endmacro()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Per-target optimization flags (applied below instead of global CMAKE_CXX_FLAGS)
option(NATIVE_ARCH "Use -march=native (faster but not portable)" ON)
if(NATIVE_ARCH)
  set(TURBO_OCR_CXX_FLAGS -O3 -march=native -Wall -Wextra -Wpedantic -Wno-unused-parameter)
else()
  set(TURBO_OCR_CXX_FLAGS -O3 -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# --- Target architecture detection (x86_64 / aarch64) ----------------------
# The binaries vendored in third_party/ (ONNX Runtime, PDFium) and bin/
# (fastpdf2png) are x86_64. On aarch64 the build fetches the matching
# ONNX Runtime release and PDFium build instead of linking the x86 copies.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _turbo_proc)
if(_turbo_proc MATCHES "aarch64|arm64")
  set(TURBO_ARCH "aarch64")
  set(TURBO_ORT_URL_ARCH "aarch64")
else()
  set(TURBO_ARCH "x64")
  set(TURBO_ORT_URL_ARCH "x64")
endif()
message(STATUS "Turbo OCR target arch: ${CMAKE_SYSTEM_PROCESSOR} -> ${TURBO_ARCH}")

# PDFium location. x86_64 uses the vendored copy; aarch64 expects an arm64
# PDFium unpacked to third_party/pdfium by scripts/install_pdfium.sh (which
# pulls pdfium-linux-arm64 from bblanchon/pdfium-binaries). Both layouts are
# {include/, lib/libpdfium.so}, so only the root differs.
set(TURBO_PDFIUM_DIR "${CMAKE_SOURCE_DIR}/third_party/pdfium"
    CACHE PATH "Root of the PDFium SDK (include/ + lib/)")

# Profile-Guided Optimization. Two-pass build on the SAME build dir:
#   1. cmake -DPGO_MODE=generate && build  →  run a representative workload
#   2. cmake -DPGO_MODE=use      && build  →  recompiles using the *.gcda profile
# Scoped to host C++ only ($<COMPILE_LANGUAGE:CXX>) — CUDA .cu objects are
# compiled by nvcc and left untouched. The .gcda files live next to the object
# files, so generate and use must share one build directory.
set(PGO_MODE "off" CACHE STRING "Profile-guided optimization: off|generate|use")
set_property(CACHE PGO_MODE PROPERTY STRINGS off generate use)
if(NOT PGO_MODE STREQUAL "off")
    # The executable is linked by the C++ compiler, but CMake prepends CUDA's
    # implicit link dirs — and since CUDA 13.2's nvcc host compiler is GCC 15,
    # that dir ships a 15.x libgcov that -lgcov would resolve before the C++
    # compiler's own. Put the actual C++ compiler's gcc lib dir first so the
    # libgcov runtime matches the instrumentation the compiler emits.
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgcov.a
        OUTPUT_VARIABLE _gcov_a OUTPUT_STRIP_TRAILING_WHITESPACE)
    get_filename_component(_gcc_libdir "${_gcov_a}" DIRECTORY)
    add_link_options("-L${_gcc_libdir}")
endif()
if(PGO_MODE STREQUAL "generate")
    message(STATUS "PGO: instrumented build (-fprofile-generate); gcov dir ${_gcc_libdir}")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fprofile-generate>)
    add_link_options(-fprofile-generate)
elseif(PGO_MODE STREQUAL "use")
    message(STATUS "PGO: profile-optimized build (-fprofile-use); gcov dir ${_gcc_libdir}")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-fprofile-use>
        $<$<COMPILE_LANGUAGE:CXX>:-fprofile-correction>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-profile>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-coverage-mismatch>)
    add_link_options(-fprofile-use)
endif()

option(USE_CPU_ONLY "Build CPU-only mode with ONNX Runtime" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
# Native builds have nothing in models/ (repo excludes the binaries), so
# fetch the PP-OCRv5 bundle from the pinned GitHub Release after build.
# Off-by-default for Docker, where fetch_release_models.sh runs as its own
# RUN layer; the container Dockerfiles set -DFETCH_MODELS=OFF explicitly.
option(FETCH_MODELS "After build, fetch OCR model bundles into ./models/" ON)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()
if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Drogon CONFIG REQUIRED)

# c-ares is a transitive dependency of gRPC's cmake config.  The apt
# package provides only pkg-config, not cmake config, so we create the
# imported target manually if cmake can't find it.
find_package(c-ares CONFIG QUIET)
if(NOT c-ares_FOUND AND NOT TARGET c-ares::cares)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(_CARES QUIET libcares)
    endif()
    add_library(c-ares::cares INTERFACE IMPORTED)
    if(_CARES_FOUND)
        set_target_properties(c-ares::cares PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_CARES_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${_CARES_LINK_LIBRARIES}")
    else()
        find_library(_CARES_LIB cares)
        if(_CARES_LIB)
            set_target_properties(c-ares::cares PROPERTIES
                INTERFACE_LINK_LIBRARIES "${_CARES_LIB}")
        endif()
    endif()
endif()

# Trantor's installed cmake targets on some distros reference a bare
# `c-ares_lib` instead of the modern `c-ares::cares` namespace. Provide a
# shim target so the linker resolves it. Harmless when Trantor is upgraded.
if(NOT TARGET c-ares_lib)
    add_library(c-ares_lib INTERFACE IMPORTED)
    find_library(_SHIM_CARES_LIB cares)
    if(_SHIM_CARES_LIB)
        set_target_properties(c-ares_lib PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_SHIM_CARES_LIB}")
    elseif(TARGET c-ares::cares)
        set_target_properties(c-ares_lib PROPERTIES
            INTERFACE_LINK_LIBRARIES "c-ares::cares")
    endif()
endif()

# --- Shared library: turbo_ocr_common ---
add_library(turbo_ocr_common STATIC
    src/detection/det_postprocess.cpp
    src/recognition/crop_utils.cpp
    src/recognition/ctc_decode.cpp
    src/decode/fast_png_decoder.cpp
    src/layout/reading_order.cpp
    src/layout/match_unsorted.cpp
    src/layout/child_blocks.cpp
    src/layout/text_line_cluster.cpp
    src/docassembly/repetition_guard.cpp
    src/docassembly/title_level.cpp
    src/docassembly/table_merge.cpp
    third_party/clipper/clipper.cpp
    third_party/simdutf/simdutf.cpp
)
target_include_directories(turbo_ocr_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/simdutf>
)
target_link_libraries(turbo_ocr_common PUBLIC ${OpenCV_LIBS})
target_compile_options(turbo_ocr_common PRIVATE ${TURBO_OCR_CXX_FLAGS})
add_library(turbo_ocr::common ALIAS turbo_ocr_common)

# --- C++ unit tests (Catch2, always built) ---
add_executable(turbo_ocr_tests
    tests/cpp/test_box.cpp
    tests/cpp/test_encoding.cpp
    tests/cpp/test_serialization.cpp
    tests/cpp/test_language_paths.cpp
    tests/cpp/test_pdf_renderer_liveness.cpp
    tests/cpp/test_ctc_decode.cpp
    tests/cpp/test_perspective.cpp
    tests/cpp/test_det_postprocess.cpp
    tests/cpp/test_image_dims.cpp
    tests/cpp/test_xy_cut.cpp
    tests/cpp/test_logger_ratelimit.cpp
    tests/cpp/test_server_config.cpp
    tests/cpp/test_docassembly.cpp
    src/render/pdf_renderer.cpp
    third_party/catch2/catch_amalgamated.cpp
)
target_link_libraries(turbo_ocr_tests PRIVATE turbo_ocr_common)
target_include_directories(turbo_ocr_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/catch2
    ${CMAKE_SOURCE_DIR}/third_party/cli11
)
target_compile_options(turbo_ocr_tests PRIVATE ${TURBO_OCR_CXX_FLAGS})

if(USE_CPU_ONLY)
    message(STATUS "Building CPU-ONLY mode (ONNX Runtime backend)")

    # Prefer system-installed ORT (Dockerfile installs to /usr/local/lib).
    # If not found, FetchContent downloads the same pinned tarball as
    # docker/Dockerfile.cpu so native CPU builds stay reproducible.
    # The vendored third_party/onnxruntime is x86_64. On x86_64 it is preferred;
    # on aarch64 it is the wrong arch, so trust only /usr/local (Docker installs
    # the right arch there) or a fresh arch-matched fetch.
    if(TURBO_ARCH STREQUAL "x64")
        set(_ort_tp_lib ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib)
        set(_ort_tp_inc ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include)
    else()
        set(_ort_tp_lib "")
        set(_ort_tp_inc "")
    endif()
    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS /usr/local/lib ${_ort_tp_lib})
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS /usr/local/include ${_ort_tp_inc})
    if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE_DIR)
        include(FetchContent)
        set(ORT_VERSION "1.22.0" CACHE STRING "ONNX Runtime version to fetch")
        # SHA256 of onnxruntime-linux-<arch>-${ORT_VERSION}.tgz. The x64 hash is
        # pinned; the aarch64 tarball has a different hash per release, so pass
        # -DORT_SHA256=... in the aarch64 build/CI (empty = fetch without check).
        if(TURBO_ARCH STREQUAL "x64")
            set(ORT_SHA256 "8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3"
                CACHE STRING "ONNX Runtime tarball SHA256")
        else()
            set(ORT_SHA256 "" CACHE STRING "ONNX Runtime aarch64 tarball SHA256 (set in CI)")
        endif()
        set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${TURBO_ORT_URL_ARCH}-${ORT_VERSION}.tgz")
        message(STATUS "ONNX Runtime not found; fetching v${ORT_VERSION} (${TURBO_ORT_URL_ARCH})")
        if(ORT_SHA256 STREQUAL "")
            message(WARNING "ORT_SHA256 unset for ${TURBO_ARCH}; fetching ONNX Runtime without hash verification")
            FetchContent_Declare(onnxruntime_prebuilt
                URL "${_ort_url}"
                SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
        else()
            FetchContent_Declare(onnxruntime_prebuilt
                URL "${_ort_url}"
                URL_HASH SHA256=${ORT_SHA256}
                SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
        endif()
        FetchContent_MakeAvailable(onnxruntime_prebuilt)
        find_library(ONNXRUNTIME_LIB onnxruntime
            HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib REQUIRED)
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include REQUIRED)
    endif()
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")

    # Protobuf + gRPC (same discovery logic as GPU build).
    turbo_find_protobuf()
    find_package(gRPC CONFIG QUIET)
    if(NOT gRPC_FOUND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GRPC REQUIRED grpc++ grpc)
        set(GRPC_LIBRARIES ${GRPC_LINK_LIBRARIES})
        find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
    else()
        set(GRPC_LIBRARIES gRPC::grpc++)
        get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)
    endif()

    # Proto codegen
    set(PROTO_SRC_DIR "${CMAKE_SOURCE_DIR}/proto")
    set(PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/proto_gen")
    file(MAKE_DIRECTORY ${PROTO_GEN_DIR})
    set(PROTO_FILE "${PROTO_SRC_DIR}/ocr.proto")
    set(PROTO_GEN_CC "${PROTO_GEN_DIR}/ocr.pb.cc")
    set(PROTO_GEN_H "${PROTO_GEN_DIR}/ocr.pb.h")
    set(GRPC_GEN_CC "${PROTO_GEN_DIR}/ocr.grpc.pb.cc")
    set(GRPC_GEN_H "${PROTO_GEN_DIR}/ocr.grpc.pb.h")

    add_custom_command(
        OUTPUT ${PROTO_GEN_CC} ${PROTO_GEN_H} ${GRPC_GEN_CC} ${GRPC_GEN_H}
        COMMAND protobuf::protoc
            --proto_path=${PROTO_SRC_DIR}
            --cpp_out=${PROTO_GEN_DIR}
            --grpc_out=${PROTO_GEN_DIR}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
            ${PROTO_FILE}
        DEPENDS ${PROTO_FILE}
        COMMENT "Generating protobuf/gRPC sources (CPU)"
    )

    # --- CPU library ---
    add_library(turbo_ocr_cpu STATIC
        src/engine/cpu_engine.cpp
        src/detection/cpu_paddle_det.cpp
        src/recognition/cpu_paddle_rec.cpp
        src/classification/cpu_paddle_cls.cpp
        src/classification/cpu_doc_orientation.cpp
        src/pipeline/cpu_ocr_pipeline.cpp
        src/layout/cpu_paddle_layout.cpp
        src/render/pdf_renderer.cpp
        src/pdf/pdf_text_layer.cpp
        src/pdf/page_image_encoder.cpp
    )
    # libjpeg-turbo: fast JPEG encode for inline page-image export.
    find_library(TURBOJPEG_LIB turbojpeg REQUIRED)
    target_include_directories(turbo_ocr_cpu PUBLIC
        ${TURBO_PDFIUM_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(turbo_ocr_cpu PUBLIC turbo_ocr_common ${ONNXRUNTIME_LIB}
        ${TURBO_PDFIUM_DIR}/lib/libpdfium.so pthread
        ${TURBOJPEG_LIB})
    target_compile_options(turbo_ocr_cpu PRIVATE ${TURBO_OCR_CXX_FLAGS})
    target_compile_definitions(turbo_ocr_cpu PUBLIC USE_CPU_ONLY)
    add_library(turbo_ocr::cpu ALIAS turbo_ocr_cpu)

    # CPU HTTP + gRPC server
    add_executable(paddle_cpu_server
        src/server/cpu_main.cpp
        src/routes/common_routes.cpp
        src/routes/pdf_routes.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(paddle_cpu_server PRIVATE
        ${PROTO_GEN_DIR}
        ${CMAKE_SOURCE_DIR}/third_party/cli11
    )
    target_link_libraries(paddle_cpu_server PRIVATE
        turbo_ocr_cpu
        Drogon::Drogon
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

else()
    message(STATUS "Building GPU mode (CUDA + TensorRT)")

    find_package(CUDAToolkit REQUIRED)

    set(TENSORRT_DIR "/usr/local/tensorrt" CACHE PATH "Path to TensorRT")

    # Protobuf + gRPC (see turbo_find_protobuf for the CONFIG/MODULE rationale).
    turbo_find_protobuf()
    find_package(gRPC CONFIG QUIET)
    if(NOT gRPC_FOUND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GRPC REQUIRED grpc++ grpc)
        set(GRPC_LIBRARIES ${GRPC_LINK_LIBRARIES})
        find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
    else()
        set(GRPC_LIBRARIES gRPC::grpc++)
        get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)
    endif()

    # Proto codegen
    set(PROTO_SRC_DIR "${CMAKE_SOURCE_DIR}/proto")
    set(PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/proto_gen")
    file(MAKE_DIRECTORY ${PROTO_GEN_DIR})
    set(PROTO_FILE "${PROTO_SRC_DIR}/ocr.proto")
    set(PROTO_GEN_CC "${PROTO_GEN_DIR}/ocr.pb.cc")
    set(PROTO_GEN_H "${PROTO_GEN_DIR}/ocr.pb.h")
    set(GRPC_GEN_CC "${PROTO_GEN_DIR}/ocr.grpc.pb.cc")
    set(GRPC_GEN_H "${PROTO_GEN_DIR}/ocr.grpc.pb.h")

    add_custom_command(
        OUTPUT ${PROTO_GEN_CC} ${PROTO_GEN_H} ${GRPC_GEN_CC} ${GRPC_GEN_H}
        COMMAND protobuf::protoc
            --proto_path=${PROTO_SRC_DIR}
            --cpp_out=${PROTO_GEN_DIR}
            --grpc_out=${PROTO_GEN_DIR}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
            ${PROTO_FILE}
        DEPENDS ${PROTO_FILE}
        COMMENT "Generating protobuf/gRPC sources"
    )

    # CUDA
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Compute capabilities to bake into the GPU binary. Default targets
    # sm_120 (Blackwell consumer, RTX 50-series) only — the full multi-arch
    # fat binary balloons to ~12.5 GB and PTX-JIT cold-start adds 10-15 s.
    # Override via -DCMAKE_CUDA_ARCHITECTURES="86;89;120" for other GPUs.
    # Reference SMs:
    #   75   Turing            (T4, RTX 20, GTX 16)
    #   80   Ampere DC         (A100)
    #   86   Ampere consumer   (RTX 30, A40, A10)
    #   89   Ada Lovelace      (RTX 40, L40, L4)
    #   90   Hopper            (H100, H200, GH200)              [needs CUDA 12.0+]
    #   100  Blackwell DC      (B100, B200, B300, GB200)        [needs CUDA 12.8+]
    #   120  Blackwell consumer (RTX 50-series)                 [needs CUDA 12.8+]
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "Comma-separated CUDA SMs to build for")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo --expt-relaxed-constexpr")

    # --- GPU library ---
    add_library(turbo_ocr_gpu STATIC
        src/engine/trt_engine.cpp
        src/engine/onnx_to_trt.cpp
        src/detection/paddle_det.cpp
        src/recognition/paddle_rec.cpp
        src/classification/paddle_cls.cpp
        src/classification/doc_orientation.cpp
        src/lang_cls/script_id.cpp
        src/layout/paddle_layout.cpp
        src/pipeline/ocr_pipeline.cpp
        src/kernels/kernels.cu
        src/kernels/jfa.cu
        src/render/pdf_renderer.cpp
        src/pdf/pdf_text_layer.cpp
        src/pdf/page_image_encoder.cpp
        # CUA router + table + formula stages (referenced by ocr_pipeline's
        # dispatch_router_). Pulled in here so paddle_highspeed_cpp links;
        # the operator's prior in-progress integration left these source
        # files on disk but not on the build target.
        src/router/cua_router.cpp
        src/router/region_features.cpp
        src/router/routing_plan.cpp
        src/table/table_stage.cpp
        src/table/table_cls.cpp
        src/table/table_cell.cpp
        src/table/slanext.cpp
        src/table/slanext_dict.cpp
        src/table/slanext_postprocess.cpp
        src/table/cell_matcher.cpp
        src/table/html_reconstruct.cpp
        src/table/kernels/table_kernels.cu
        src/formula/formulanet.cpp
        src/formula/formula_tokenizer.cpp
        src/formula/kernels/formula_kernels.cu
    )
    target_include_directories(turbo_ocr_gpu PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_DIR}/include
        ${PROTO_GEN_DIR}
        ${TURBO_PDFIUM_DIR}/include
    )
    target_link_directories(turbo_ocr_gpu PUBLIC ${TENSORRT_DIR}/lib)
    # Link libpdfium.so directly for in-process text-layer extraction.
    # The shared lib is vendored in third_party/pdfium/lib and also placed
    # in bin/ by scripts/install_fastpdf2png.sh (so native dev + Docker
    # builds both resolve it at runtime). Docker additionally copies it to
    # /usr/lib via ldconfig in Dockerfile.gpu:49.
    # libjpeg-turbo: CPU JPEG encode fallback for inline page-image export
    # (nvJPEG handles the GPU default path; already linked below).
    find_library(TURBOJPEG_LIB turbojpeg REQUIRED)
    message(STATUS "libjpeg-turbo: ${TURBOJPEG_LIB}")
    target_link_libraries(turbo_ocr_gpu PUBLIC
        turbo_ocr_common
        nvinfer
        nvinfer_plugin
        nvonnxparser
        nvjpeg
        cudart
        ${TURBO_PDFIUM_DIR}/lib/libpdfium.so
        ${TURBOJPEG_LIB}
    )
    target_compile_options(turbo_ocr_gpu PRIVATE ${TURBO_OCR_CXX_FLAGS})
    add_library(turbo_ocr::gpu ALIAS turbo_ocr_gpu)

    # GPU tests link to GPU library so CUDA-dependent tests can be added here.
    target_link_libraries(turbo_ocr_tests PRIVATE turbo_ocr_gpu CUDA::cudart)
    target_include_directories(turbo_ocr_tests PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_DIR}/include
    )

    # GPU HTTP + gRPC server (shared pipeline pool)
    add_executable(paddle_highspeed_cpp
        src/server/main.cpp
        src/routes/common_routes.cpp
        src/routes/image_routes.cpp
        src/routes/pdf_routes.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(paddle_highspeed_cpp PRIVATE
        ${PROTO_GEN_DIR}
        ${CMAKE_SOURCE_DIR}/third_party/cli11
    )
    target_link_libraries(paddle_highspeed_cpp PRIVATE
        turbo_ocr_gpu
        Drogon::Drogon
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

    # Legacy standalone gRPC server removed — gRPC now runs from the main
    # binary via start_grpc_server() in grpc_service.h.

    # Benchmark tools
    add_executable(grpc_bench
        tools/grpc_bench.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(grpc_bench PRIVATE ${PROTO_GEN_DIR})
    target_link_libraries(grpc_bench PRIVATE
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

    add_executable(grpc_burst
        tools/grpc_burst.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(grpc_burst PRIVATE ${PROTO_GEN_DIR})
    target_link_libraries(grpc_burst PRIVATE
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

    # diff_ccl: side-by-side comparator for CPU CCL=1 vs GPU CCL=2 outputs.
    # Loads an image, runs both detection paths, prints box counts, IoU stats,
    # and missed/extra boxes. Used to drive the GPU JFA pipeline accuracy.
    add_executable(diff_ccl tools/diff_ccl.cpp)
    target_include_directories(diff_ccl PRIVATE include)
    target_link_libraries(diff_ccl PRIVATE
        turbo_ocr_gpu
        ${OpenCV_LIBS}
    )
    set_target_properties(diff_ccl PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # PDF-only static-engine throughput benchmark.
    add_executable(pdf_static_bench tools/pdf_static_bench.cpp)
    target_include_directories(pdf_static_bench PRIVATE include)
    target_link_libraries(pdf_static_bench PRIVATE
        turbo_ocr_gpu
        ${OpenCV_LIBS}
    )
    set_target_properties(pdf_static_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Regression: nvJPEG handler-thread -> dispatcher-worker race
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_nvjpeg_race.cpp)
        add_executable(test_nvjpeg_race tools/test_nvjpeg_race.cpp)
        target_include_directories(test_nvjpeg_race PRIVATE include)
        target_link_libraries(test_nvjpeg_race PRIVATE
            CUDA::cudart
            CUDA::nvjpeg
            ${OpenCV_LIBS}
        )
    endif()
endif()

# --- Interprocedural / link-time optimization (Release only) --------------
# Skipped for turbo_ocr_tests: Catch2's amalgamated TU is large and LTO
# linking adds significant time with no runtime payoff for unit tests.
include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_err)
if(_ipo_ok AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Enabling IPO/LTO for Release targets")
    set_property(TARGET turbo_ocr_common PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    if(USE_CPU_ONLY)
        set_property(TARGET turbo_ocr_cpu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_property(TARGET paddle_cpu_server PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        set_property(TARGET turbo_ocr_gpu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_property(TARGET paddle_highspeed_cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
elseif(NOT _ipo_ok)
    message(STATUS "IPO/LTO not supported by this toolchain: ${_ipo_err}")
endif()

# --- Install rules --------------------------------------------------------
# Minimal layout so `cmake --install build --prefix /usr/local` works.
if(USE_CPU_ONLY)
    install(TARGETS paddle_cpu_server RUNTIME DESTINATION bin)
else()
    install(TARGETS paddle_highspeed_cpp RUNTIME DESTINATION bin)
endif()
install(DIRECTORY models/ DESTINATION share/turbo-ocr/models OPTIONAL)
install(DIRECTORY scripts/ DESTINATION share/turbo-ocr/scripts USE_SOURCE_PERMISSIONS)

# --- Fetch OCR model bundles for native builds ----------------------------
# Uses det.onnx as a sentinel — ninja/make skip the step once it exists.
# Docker builds explicitly pass -DFETCH_MODELS=OFF because the Dockerfile
# runs fetch_release_models.sh as its own cacheable RUN layer.
if(FETCH_MODELS)
    set(MODELS_SENTINEL "${CMAKE_SOURCE_DIR}/models/det.onnx")
    add_custom_command(
        OUTPUT ${MODELS_SENTINEL}
        COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/fetch_release_models.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Fetching PP-OCRv5 models from pinned GitHub Release"
        VERBATIM
    )
    add_custom_target(fetch_models ALL DEPENDS ${MODELS_SENTINEL})
    if(USE_CPU_ONLY)
        add_dependencies(paddle_cpu_server fetch_models)
    else()
        add_dependencies(paddle_highspeed_cpp fetch_models)
    endif()
endif()
