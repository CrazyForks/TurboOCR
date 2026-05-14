file(GLOB TABLE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu")
if(NOT TABLE_SOURCES)
    return()
endif()

add_library(turbo_ocr_table STATIC ${TABLE_SOURCES})
target_include_directories(turbo_ocr_table PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_DIR}/include)
target_link_directories(turbo_ocr_table PUBLIC ${TENSORRT_DIR}/lib)
target_link_libraries(turbo_ocr_table PUBLIC
    turbo_ocr_common
    turbo_ocr_gpu
    nvinfer nvinfer_plugin nvonnxparser cudart)
target_compile_options(turbo_ocr_table PRIVATE ${TURBO_OCR_CXX_FLAGS})
set_target_properties(turbo_ocr_table PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_library(turbo_ocr::table ALIAS turbo_ocr_table)
