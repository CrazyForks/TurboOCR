file(GLOB VLM_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
if(NOT VLM_SOURCES)
    return()
endif()

add_library(turbo_ocr_vlm STATIC ${VLM_SOURCES})
target_include_directories(turbo_ocr_vlm PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party)
find_package(CURL REQUIRED)
target_link_libraries(turbo_ocr_vlm PUBLIC
    turbo_ocr_common
    CURL::libcurl)
target_compile_options(turbo_ocr_vlm PRIVATE ${TURBO_OCR_CXX_FLAGS})
add_library(turbo_ocr::vlm ALIAS turbo_ocr_vlm)
