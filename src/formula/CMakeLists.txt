file(GLOB FORMULA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu")
if(NOT FORMULA_SOURCES)
    return()
endif()

add_library(turbo_ocr_formula STATIC ${FORMULA_SOURCES})
target_include_directories(turbo_ocr_formula PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party)
target_link_directories(turbo_ocr_formula PUBLIC ${TENSORRT_DIR}/lib)
target_link_libraries(turbo_ocr_formula PUBLIC
    turbo_ocr_common
    turbo_ocr_gpu
    nvinfer nvinfer_plugin nvonnxparser cudart)
target_compile_options(turbo_ocr_formula PRIVATE ${TURBO_OCR_CXX_FLAGS})
set_target_properties(turbo_ocr_formula PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_library(turbo_ocr::formula ALIAS turbo_ocr_formula)
