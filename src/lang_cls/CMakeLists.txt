# Per-text-line script classifier. Reuses the rec/cls perspective-warp
# kernel (cuda_batch_roi_warp) since its (pixel/255 - 0.5)/0.5 RGB CHW
# output is bit-identical to the script_id training-recipe normalization.
file(GLOB LANG_CLS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
if(NOT LANG_CLS_SOURCES)
    return()
endif()

add_library(turbo_ocr_lang_cls STATIC ${LANG_CLS_SOURCES})
target_include_directories(turbo_ocr_lang_cls PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party)
target_link_directories(turbo_ocr_lang_cls PUBLIC ${TENSORRT_DIR}/lib)
target_link_libraries(turbo_ocr_lang_cls PUBLIC
    turbo_ocr_common
    turbo_ocr_gpu
    nvinfer nvinfer_plugin nvonnxparser cudart)
target_compile_options(turbo_ocr_lang_cls PRIVATE ${TURBO_OCR_CXX_FLAGS})
add_library(turbo_ocr::lang_cls ALIAS turbo_ocr_lang_cls)
