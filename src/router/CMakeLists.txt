file(GLOB ROUTER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
if(NOT ROUTER_SOURCES)
    return()
endif()

add_library(turbo_ocr_router STATIC ${ROUTER_SOURCES})
target_include_directories(turbo_ocr_router PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(turbo_ocr_router PUBLIC turbo_ocr_common)
target_compile_options(turbo_ocr_router PRIVATE ${TURBO_OCR_CXX_FLAGS})
add_library(turbo_ocr::router ALIAS turbo_ocr_router)

# Router unit tests — separate executable so it builds without dragging in
# turbo_ocr_gpu (the root turbo_ocr_tests target links CUDA via the GPU lib).
file(GLOB ROUTER_TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/tests/router/*.cpp")
if(ROUTER_TEST_SOURCES)
    add_executable(turbo_ocr_router_tests
        ${ROUTER_TEST_SOURCES}
        ${CMAKE_SOURCE_DIR}/third_party/catch2/catch_amalgamated.cpp
    )
    target_include_directories(turbo_ocr_router_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/catch2
    )
    target_link_libraries(turbo_ocr_router_tests PRIVATE
        turbo_ocr_router
        turbo_ocr_common
    )
    target_compile_options(turbo_ocr_router_tests PRIVATE ${TURBO_OCR_CXX_FLAGS})
endif()
